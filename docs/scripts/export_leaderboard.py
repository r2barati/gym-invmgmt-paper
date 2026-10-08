"""Export benchmark results as compact JSON for the project page.

Reads the canonical merged CSV (per-scenario means) and the per-agent caches
(one row per scenario and seed) and writes:

- scenario metadata and per-scenario means for every metric the page shows,
- per-seed profits, paired with the Oracle's on the same demand seeds, so the
  page can compute bootstrap confidence intervals and performance profiles,
- mean cost decomposition (profit = revenue - procurement - holding - backlog
  penalty - operating - fixed ordering),
- a one-line description of every agent configuration.

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

from run_benchmarks import ABLATION_AGENT_IDS, ALL_AGENT_IDS, _agent_prefix, _cache_path  # noqa: E402

FAMILY = {
    "Oracle": "Oracle",
    "MSSP": "OR", "MSSP-I": "OR", "DLP": "OR", "DLP-I": "OR",
    "Newsvendor": "Heuristic", "(s,S)": "Heuristic", "ExpSmoothing": "Heuristic",
    "Newsvendor-I": "Heuristic", "(s,S)-I": "Heuristic", "ExpSmoothing-I": "Heuristic",
    "EchelonApprox": "Heuristic", "EchelonApprox-I": "Heuristic",
    "GNN-IL": "Imitation", "DAgger-G": "Imitation", "DAgger-B": "Imitation",
    "LLM-Policy-C": "LLM",
}
BLIND = {"MSSP", "DLP", "Newsvendor", "(s,S)", "ExpSmoothing", "EchelonApprox", "DAgger-B"}

# One line per method, written from the agent implementations in agents/.
DESCRIPTIONS = {
    "Oracle": "Solves one full-horizon linear program with the realized demand known in advance. "
              "A perfect-information upper bound, not a deployable policy.",
    "MSSP": "Rolling-horizon multi-stage stochastic program. Samples demand scenarios (SAA) with "
            "non-anticipativity constraints and re-solves every period.",
    "DLP": "Rolling-horizon deterministic linear program over a 10-period window, using point "
           "demand forecasts and re-solved every period.",
    "Newsvendor": "Base-stock target per node from the newsvendor critical ratio b / (b + h) over "
                  "lead-time demand, with a normal approximation.",
    "(s,S)": "Reorder point s = lead-time demand plus safety stock; when inventory falls below s, "
             "order up to S = s + EOQ.",
    "ExpSmoothing": "Holt double exponential smoothing (level and trend) forecast, with safety stock "
                    "from the forecast error.",
    "EchelonApprox": "Clark-Scarf-inspired echelon base-stock policy: targets from cumulative "
                     "lead time and a critical ratio.",
    "PPO-MLP": "PPO with a multilayer-perceptron policy on engineered per-node and global features.",
    "PPO-GNN": "PPO with a directed, edge-conditioned message-passing graph network over the "
               "supply network.",
    "PPO-Transformer": "PPO with a transformer encoder that treats each node as a token.",
    "ST-PPO": "PPO with spatio-temporal attention over (node, period) tokens from a 4-period history.",
    "Residual": "PPO learns a per-link correction of up to 50% on top of the Newsvendor heuristic's order.",
    "SAC": "Soft Actor-Critic with a multilayer-perceptron policy on engineered features.",
    "GNN-IL": "Behavioral cloning of Oracle decisions into the PPO-GNN architecture, then PPO fine-tuning.",
    "DAgger-G": "DAgger imitation of the Oracle (iterative dataset aggregation) with the PPO-GNN architecture.",
    "PPO-MLP-v1": "Observation ablation of PPO-MLP: 3 basic features per node instead of the enhanced set.",
    "PPO-MLP-raw": "Observation ablation of PPO-MLP: the raw 71-dimensional observation with no "
                   "engineered features. Trained and evaluated on the base network only.",
    "LLM-Policy-C": "A local LLM (Qwen2.5-1.5B-Instruct) is queried once per episode for strategy "
                    "multipliers that set a bounded base-stock policy.",
}
INFORMED_NOTE = "Informed: reads the true mean of the demand process."
BLIND_NOTE = "Blind: estimates demand from realized history only."

METRICS = {"Profit": "profit", "Profit_Std": "profit_std", "FillRate": "fill_rate", "AvgInv": "avg_inv",
           "CVaR5": "cvar5", "Time_Sec": "time_sec"}
COSTS = {"Revenue": "revenue", "ProcurementCost": "procurement", "HoldingCost": "holding",
         "BacklogPenalty": "backlog", "OperatingCost": "operating", "FixedOrderingCost": "fixed"}


def _num(v, nd=4):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, nd)


def describe(aid):
    base = aid[:-2] if aid.endswith("-I") or aid.endswith("-B") else aid
    if aid in ("DAgger-B",):
        base = "DAgger-G"
    text = DESCRIPTIONS.get(aid) or DESCRIPTIONS[base]
    if aid in ("Oracle", "PPO-MLP-v1", "PPO-MLP-raw", "LLM-Policy-C"):
        return text
    if aid.endswith("-B") and base in DESCRIPTIONS:
        return text + " Blind variant: its features use a moving average of realized demand instead of the true mean."
    return text + " " + (BLIND_NOTE if aid in BLIND else INFORMED_NOTE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "results", "benchmark_final_merged.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "data", "leaderboard.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    keys = df["ScenarioKey"].tolist()
    scenarios = [
        {"key": r.ScenarioKey, "block": r.Block, "network": r.Network, "demand": r.Demand,
         "goodwill": bool(r.Goodwill), "backlog": bool(r.Backlog), "marl": bool(r.MARL)}
        for r in df.itertuples()
    ]
    oracle_cache = pd.read_csv(_cache_path("Oracle"))
    seeds = sorted(int(s) for s in oracle_cache["Seed"].unique())

    agents = []
    for aid in ALL_AGENT_IDS + ["LLM-Policy-C"]:
        pre = _agent_prefix(aid)
        if f"{pre}_Profit" not in df.columns:
            continue
        rows = []
        for _, r in df.iterrows():
            rec = {}
            for col, key in METRICS.items():
                rec[key] = _num(r.get(f"Time_Sec_{pre}" if col == "Time_Sec" else f"{pre}_{col}"))
            rows.append(rec)

        cache = pd.read_csv(_cache_path(aid))
        for col in COSTS:
            if col not in cache.columns:
                cache[col] = 0.0
        by = cache.set_index(["ScenarioKey", "Seed"])
        per_seed, costs = [], []
        for k in keys:
            if k in by.index.get_level_values(0):
                sub = by.loc[k]
                per_seed.append([_num(sub.loc[s, "Profit"], 1) if s in sub.index else None for s in seeds])
                costs.append({v: _num(sub[c].mean(), 1) for c, v in COSTS.items()})
            else:
                per_seed.append(None)
                costs.append(None)
        # The per-seed cache must reproduce the merged means it was merged into.
        for i, (r, ps) in enumerate(zip(rows, per_seed)):
            if ps is not None and r["profit"] is not None:
                mean = sum(ps) / len(ps)
                assert abs(mean - r["profit"]) < 0.05 + 1e-6 * abs(mean), (aid, keys[i], mean, r["profit"])

        agents.append({
            "id": aid,
            "family": FAMILY.get(aid, "RL"),
            "informed": aid not in BLIND and not aid.endswith("-B"),
            "role": "bound" if aid == "Oracle" else ("ablation" if aid in ABLATION_AGENT_IDS else
                                                     ("llm" if aid.startswith("LLM") else "primary")),
            "desc": describe(aid),
            "results": rows,
            "seeds": per_seed,
            "costs": costs,
        })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"source": os.path.relpath(args.csv, ROOT), "seeds": seeds, "scenarios": scenarios,
                   "agents": agents}, f, separators=(",", ":"))
    print(f"{len(agents)} agents x {len(scenarios)} scenarios x {len(seeds)} seeds -> {args.out}")


if __name__ == "__main__":
    main()
