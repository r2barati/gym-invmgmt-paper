"""Export per-period rollouts for the project-page simulation player.

Replays the benchmark's own episodes (one canonical seed) on every main
scenario (the 22 non-MARL rows) and writes the state matrices the page
animates (inventory, shipments, demand, backlog, profit), one JSON file per
scenario plus an index.

Optimization and heuristic policies always run. Trained policies run when their
checkpoints are in data/models/ (see download_weights.sh); otherwise they are
skipped, or the script fails with --require-ml.

Usage (from the repository root):
    python docs/scripts/export_rollouts.py --out docs/data/rollouts
    python docs/scripts/export_rollouts.py --require-ml --verify   # as run in CI

With --verify, each replay is checked against results/cache_v2 and the largest
deviation per policy is written to index.json.
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (ROOT, os.path.join(ROOT, "agents"), os.path.join(ROOT, "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_benchmarks import (  # noqa: E402
    _cache_path,
    _env_kwargs,
    _extract_kpis,
    _make_agent,
    build_all_scenarios,
    scenario_id,
)

from gym_invmgmt import CoreEnv  # noqa: E402

# (agent id, family shown on the page). Learned policies need released checkpoints.
AGENTS = [
    ("Oracle", "Upper bound"),
    ("MSSP-I", "Optimization"),
    ("DLP", "Optimization"),
    ("Newsvendor-I", "Heuristic"),
    ("(s,S)", "Heuristic"),
    ("PPO-Transformer", "Learned"),
    ("PPO-MLP", "Learned"),
    ("DAgger-G", "Learned"),
]
LEARNED = {a for a, fam in AGENTS if fam == "Learned"}
DEFAULT = "A_Core|base|trend+seasonal+shock|GW:False|BL:True|MARL:False"


def _r(a, nd=1):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def slug(key):
    return re.sub(r"[^a-z0-9]+", "-", key.lower().replace("+", "-")).strip("-")


def rollout(agent_id, family, scfg, seed):
    from mssp_agent import RollingHorizonMSSPAgent  # needs PuLP, so imported only when replaying

    env = CoreEnv(**_env_kwargs(scfg))
    try:
        agent = _make_agent(agent_id, env, scfg, seed)
    except ImportError:  # learned policies need torch and stable-baselines3
        if agent_id not in LEARNED:
            raise
        agent = None
    if agent is None:
        return None
    obs, _ = env.reset(seed=seed)
    done, t = False, 0
    while not done:
        action = agent.get_action(t) if isinstance(agent, RollingHorizonMSSPAgent) else agent.get_action(obs, t)
        obs, _, term, trunc, _ = env.step(action)
        done, t = term or trunc, t + 1
    T = env.period
    k = _extract_kpis(env)
    return {
        "agent": agent_id,
        "family": family,
        "kpis": {"profit": round(k["Profit"], 1), "fill_rate": round(k["FillRate"], 4),
                 "avg_inv": round(k["AvgInv"], 1)},
        "X": _r(env.X[: T + 1]),          # on-hand inventory per node index
        "R": _r(env.R[:T]),               # filled replenishment per reorder link
        "S": _r(env.S[:T]),               # shipments per network edge
        "D": _r(env.D[:T]),               # customer demand per retail link
        "U": _r(env.U[:T]),               # backlog (or lost sales) per retail link
        "P": _r(np.cumsum(env.P[:T].sum(axis=1)), 1),  # cumulative profit
    }


def scenario_meta(scfg, seed):
    """Network layout and settings the page needs to draw a scenario."""
    env = CoreEnv(**_env_kwargs(scfg))
    net = env.network
    return {
        "scenario": scenario_id(scfg),
        "seed": seed,
        "periods": env.num_periods,
        "backlog": bool(scfg["backlog"]),
        "nodes": [int(n) for n in env.graph.nodes],
        "node_index": {int(n): int(i) for n, i in net.node_map.items()},
        "levels": {k: [int(x) for x in v] for k, v in net.levels.items()},
        "reorder_links": [[int(a), int(b)] for a, b in net.reorder_links],
        "retail_links": [[int(a), int(b)] for a, b in net.retail_links],
        "network_links": [[int(a), int(b)] for a, b in net.network_links],
        "lead_times": {f"{a}-{b}": int(env.graph.edges[a, b].get("L", 0)) for a, b in net.reorder_links},
    }


def export(scfg, seed):
    meta = scenario_meta(scfg, seed)
    runs = [r for r in (rollout(a, fam, scfg, seed) for a, fam in AGENTS) if r is not None]
    return {"meta": meta, "runs": runs}


def matches_cache(agent_id, cached, replay):
    """Whether a replayed profit reproduces the benchmark's cached profit.

    Optimization and heuristic policies must match to rounding. Learned policies may
    drift slightly with the torch / stable-baselines3 version, so they get a 1%
    relative tolerance.
    """
    diff = abs(cached - replay)
    return diff / max(abs(cached), 1.0) <= 0.01 if agent_id in LEARNED else diff <= 0.5


def verify(key, seed, runs, caches, worst):
    """Check each replay against the cache; record the largest deviation per policy in `worst`."""
    for r in runs:
        c = caches[r["agent"]]
        row = c[(c["ScenarioKey"] == key) & (c["Seed"] == seed)]
        if row.empty:
            raise SystemExit(f"No cached result for {r['agent']} on {key}, seed {seed}")
        cached, replay = float(row["Profit"].iloc[0]), r["kpis"]["profit"]
        diff = abs(cached - replay)
        w = worst.setdefault(r["agent"], {"max_abs": 0.0, "max_rel": 0.0})
        w["max_abs"] = max(w["max_abs"], round(diff, 2))
        w["max_rel"] = max(w["max_rel"], round(diff / max(abs(cached), 1.0), 5))
        if not matches_cache(r["agent"], cached, replay):
            raise SystemExit(f"{r['agent']} on {key}: replay profit {replay} != cached {cached:.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data", "rollouts"))
    ap.add_argument("--seed", type=int, default=42)  # first canonical benchmark seed
    ap.add_argument("--require-ml", action="store_true", help="fail if a learned policy's checkpoint is missing")
    ap.add_argument("--verify", action="store_true", help="check every replay against results/cache_v2")
    args = ap.parse_args()

    caches = {a: pd.read_csv(_cache_path(a)) for a, _ in AGENTS} if args.verify else None
    os.makedirs(args.out, exist_ok=True)
    index, worst = [], {}
    for scfg in build_all_scenarios():
        if scfg["marl"]:
            continue
        key = scenario_id(scfg)
        data = export(scfg, args.seed)
        missing = LEARNED - {r["agent"] for r in data["runs"]}
        if missing and args.require_ml:
            raise SystemExit(f"Missing checkpoints for {sorted(missing)}; run download_weights.sh first")
        if args.verify:
            verify(key, args.seed, data["runs"], caches, worst)
        with open(os.path.join(args.out, slug(key) + ".json"), "w") as f:
            json.dump(data, f, separators=(",", ":"))
        index.append({"key": key, "file": slug(key) + ".json"})
        print(key, {r["agent"]: r["kpis"]["profit"] for r in data["runs"]}, flush=True)
    with open(os.path.join(args.out, "index.json"), "w") as f:
        out = {"default": DEFAULT, "agents": [a for a, _ in AGENTS], "learned": sorted(LEARNED), "seed": args.seed,
               "scenarios": index}
        if args.verify:
            out["verification"] = worst  # largest |replay - cached| profit per policy
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
