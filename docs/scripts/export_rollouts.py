"""Export per-period rollouts for the project-page animation.

Runs a few weight-free agents (no RL checkpoints needed) on one canonical
benchmark scenario and writes the state matrices the page animates
(inventory, pipeline, shipments, demand, backlog, profit) to JSON.

Usage (from the repository root):
    python docs/scripts/export_rollouts.py --out docs/data/rollouts.json
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for p in (ROOT, os.path.join(ROOT, "agents"), os.path.join(ROOT, "benchmarks")):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_benchmarks import _env_kwargs, _extract_kpis, _make_agent, build_all_scenarios, scenario_id  # noqa: E402

from gym_invmgmt import CoreEnv  # noqa: E402

AGENTS = ["Oracle", "(s,S)", "Newsvendor-I", "DLP"]
SCENARIO = "A_Core|base|trend+seasonal+shock|GW:False|BL:True|MARL:False"


def _r(a, nd=1):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


def rollout(agent_id, scfg, seed):
    env = CoreEnv(**_env_kwargs(scfg))
    agent = _make_agent(agent_id, env, scfg, seed)
    obs, _ = env.reset(seed=seed)
    done, t = False, 0
    while not done:
        obs, _, term, trunc, _ = env.step(agent.get_action(obs, t))
        done, t = term or trunc, t + 1
    T = env.period
    k = _extract_kpis(env)
    return {
        "agent": agent_id,
        "kpis": {"profit": round(k["Profit"], 1), "fill_rate": round(k["FillRate"], 4),
                 "avg_inv": round(k["AvgInv"], 1)},
        "X": _r(env.X[: T + 1]),          # on-hand inventory per node index
        "Y": _r(env.Y[: T + 1]),          # pipeline per reorder link
        "R": _r(env.R[:T]),               # filled replenishment per reorder link
        "S": _r(env.S[:T]),               # shipments per network edge
        "D": _r(env.D[:T]),               # customer demand per retail link
        "U": _r(env.U[:T]),               # standing backlog per retail link
        "P": _r(np.cumsum(env.P[:T].sum(axis=1)), 1),  # cumulative profit
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data", "rollouts.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scfg = next(s for s in build_all_scenarios() if scenario_id(s) == SCENARIO)
    env = CoreEnv(**_env_kwargs(scfg))
    net = env.network
    meta = {
        "scenario": SCENARIO,
        "seed": args.seed,
        "periods": env.num_periods,
        "nodes": [int(n) for n in env.graph.nodes],
        "node_index": {int(n): int(i) for n, i in net.node_map.items()},
        "levels": {k: [int(x) for x in v] for k, v in net.levels.items()},
        "reorder_links": [[int(a), int(b)] for a, b in net.reorder_links],
        "retail_links": [[int(a), int(b)] for a, b in net.retail_links],
        "network_links": [[int(a), int(b)] for a, b in net.network_links],
        "lead_times": {f"{a}-{b}": int(env.graph.edges[a, b].get("L", 0)) for a, b in net.reorder_links},
    }
    runs = [rollout(a, scfg, args.seed) for a in AGENTS]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "runs": runs}, f, separators=(",", ":"))
    for r in runs:
        print(r["agent"], r["kpis"])


if __name__ == "__main__":
    main()
