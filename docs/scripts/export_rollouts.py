"""Export per-period rollouts for the project-page simulation player.

Runs five policies that need no RL checkpoints on every main benchmark scenario
(the 22 non-MARL rows) and writes the state matrices the page animates
(inventory, pipeline, shipments, demand, backlog, profit), one JSON file per
scenario plus an index.

Usage (from the repository root):
    python docs/scripts/export_rollouts.py --out docs/data/rollouts
"""

import argparse
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (ROOT, os.path.join(ROOT, "agents"), os.path.join(ROOT, "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

from mssp_agent import RollingHorizonMSSPAgent  # noqa: E402
from run_benchmarks import _env_kwargs, _extract_kpis, _make_agent, build_all_scenarios, scenario_id  # noqa: E402

from gym_invmgmt import CoreEnv  # noqa: E402

AGENTS = ["Oracle", "MSSP-I", "Newsvendor-I", "(s,S)", "DLP"]
DEFAULT = "A_Core|base|trend+seasonal+shock|GW:False|BL:True|MARL:False"


def _r(a, nd=1):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def slug(key):
    return re.sub(r"[^a-z0-9]+", "-", key.lower().replace("+", "-")).strip("-")


def rollout(agent_id, scfg, seed):
    env = CoreEnv(**_env_kwargs(scfg))
    agent = _make_agent(agent_id, env, scfg, seed)
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
        "kpis": {"profit": round(k["Profit"], 1), "fill_rate": round(k["FillRate"], 4),
                 "avg_inv": round(k["AvgInv"], 1)},
        "X": _r(env.X[: T + 1]),          # on-hand inventory per node index
        "R": _r(env.R[:T]),               # filled replenishment per reorder link
        "S": _r(env.S[:T]),               # shipments per network edge
        "D": _r(env.D[:T]),               # customer demand per retail link
        "U": _r(env.U[:T]),               # backlog (or lost sales) per retail link
        "P": _r(np.cumsum(env.P[:T].sum(axis=1)), 1),  # cumulative profit
    }


def export(scfg, seed):
    env = CoreEnv(**_env_kwargs(scfg))
    net = env.network
    meta = {
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
    return {"meta": meta, "runs": [rollout(a, scfg, seed) for a in AGENTS]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data", "rollouts"))
    ap.add_argument("--seed", type=int, default=42)  # first canonical benchmark seed
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    index = []
    for scfg in build_all_scenarios():
        if scfg["marl"]:
            continue
        key = scenario_id(scfg)
        data = export(scfg, args.seed)
        with open(os.path.join(args.out, slug(key) + ".json"), "w") as f:
            json.dump(data, f, separators=(",", ":"))
        index.append({"key": key, "file": slug(key) + ".json"})
        print(key, {r["agent"]: r["kpis"]["profit"] for r in data["runs"]}, flush=True)
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump({"default": DEFAULT, "agents": AGENTS, "seed": args.seed, "scenarios": index}, f, indent=1)


if __name__ == "__main__":
    main()
