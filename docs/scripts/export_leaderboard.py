"""Export the canonical benchmark CSV as compact JSON for the project-page leaderboard.

Usage (from the repository root):
    python docs/scripts/export_leaderboard.py --out docs/data/leaderboard.json
"""

import argparse
import json
import math
import os
import sys

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "benchmarks"))

from run_benchmarks import ABLATION_AGENT_IDS, ALL_AGENT_IDS, _agent_prefix  # noqa: E402

FAMILY = {
    "Oracle": "Oracle",
    "MSSP": "OR", "MSSP-I": "OR", "DLP": "OR", "DLP-I": "OR",
    "Newsvendor": "Heuristic", "(s,S)": "Heuristic", "ExpSmoothing": "Heuristic",
    "Newsvendor-I": "Heuristic", "(s,S)-I": "Heuristic", "ExpSmoothing-I": "Heuristic",
    "EchelonApprox": "Heuristic", "EchelonApprox-I": "Heuristic",
    "GNN-IL": "Imitation", "DAgger-G": "Imitation", "DAgger-B": "Imitation",
    "LLM-Policy-C": "LLM",
}
METRICS = {"Profit": "profit", "Profit_Std": "profit_std", "FillRate": "fill_rate", "AvgInv": "avg_inv",
           "CVaR5": "cvar5", "BullwhipRatio": "bullwhip", "Time_Sec": "time_sec"}


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "benchmark_final_merged.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data", "leaderboard.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    scenarios = [
        {"key": r.ScenarioKey, "block": r.Block, "network": r.Network, "demand": r.Demand,
         "goodwill": bool(r.Goodwill), "backlog": bool(r.Backlog), "marl": bool(r.MARL)}
        for r in df.itertuples()
    ]
    agents = []
    for aid in ALL_AGENT_IDS + ["LLM-Policy-C"]:
        pre = _agent_prefix(aid)
        if f"{pre}_Profit" not in df.columns:
            continue
        rows = []
        for _, r in df.iterrows():
            rec = {}
            for col, key in METRICS.items():
                name = f"Time_Sec_{pre}" if col == "Time_Sec" else f"{pre}_{col}"
                rec[key] = _num(r.get(name))
            rows.append(rec)
        agents.append({
            "id": aid,
            "family": FAMILY.get(aid, "RL"),
            "informed": not (aid.endswith("-B") or aid in {"MSSP", "DLP", "Newsvendor", "(s,S)", "ExpSmoothing",
                                                           "EchelonApprox", "DAgger-B"}),
            "role": "bound" if aid == "Oracle" else ("ablation" if aid in ABLATION_AGENT_IDS else
                                                     ("llm" if aid.startswith("LLM") else "primary")),
            "results": rows,
        })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"source": os.path.relpath(args.csv, ROOT), "scenarios": scenarios, "agents": agents}, f,
                  separators=(",", ":"))
    print(f"{len(agents)} agents x {len(scenarios)} scenarios -> {args.out}")


if __name__ == "__main__":
    main()
