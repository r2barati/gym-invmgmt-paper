#!/usr/bin/env python3
"""Extract meaningful aggregate tables from benchmark_final_merged.csv for appendix."""
import csv, statistics, sys, json
from collections import defaultdict

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANONICAL_CSV = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "results", "benchmark_final_merged.csv")
)

CSV = os.environ.get('BENCHMARK_CSV', None)
if not CSV:
    CSV = CANONICAL_CSV
if not os.path.exists(CSV):
    raise FileNotFoundError(
        f"Could not find benchmark CSV at {CSV}. "
        "Set BENCHMARK_CSV to intentionally use a non-canonical artifact."
    )

# Core agents for the paper
AGENTS = [
    "Oracle", "MSSP", "GNN_V3", "PPO_Transformer", "ST_PPO", "Residual",
    "PPO_MLP", "SAC", "DAgger_G", "DAgger_B", "Newsvendor", "sS",
    "ExpSmoothing", "Newsvendor_I", "sS_I", "ExpSmooth_I", "DLP",
    "GNN_IL",
]

# Read all rows (exclude MARL supplementary — headline tables use 22-scenario main grid)
with open(CSV) as f:
    reader = csv.DictReader(f)
    all_rows = list(reader)
rows = [r for r in all_rows if r.get('MARL', 'False') != 'True']

if len(all_rows) > 0:
    required_cols = [f"{agent}_Profit" for agent in AGENTS]
    missing = [c for c in required_cols if c not in all_rows[0]]
    if missing:
        print(f"\n[ERROR] Missing required columns in CSV: {missing}")
        print("Run all paper-table agents, then --merge, before generating appendix tables.")
        sys.exit(1)

print(f"Total scenarios: {len(rows)} main ({len(all_rows)} including MARL)")
print(f"Total columns: {len(all_rows[0]) if all_rows else 0}")
print()

# ──────────────────────────────────────────────────────────────
# 1. FULL SCENARIO MATRIX — Summary Performance Table
# ──────────────────────────────────────────────────────────────
print("="*80)
print("TABLE 1: AGGREGATE PERFORMANCE ACROSS ALL SCENARIOS")
print("="*80)
print(f"{'Agent':<18} {'Mean Profit':>12} {'Std':>8} {'Mean FR':>8} {'Mean Inv':>10} {'Mean Unful':>10} {'Mean CVaR5':>12} {'N':>4}")

for agent in AGENTS:
    prof_key = f"{agent}_Profit"
    fr_key = f"{agent}_FillRate"
    inv_key = f"{agent}_AvgInv"
    unf_key = f"{agent}_Unfulfilled"
    cvar_key = f"{agent}_CVaR5"
    
    profits = []
    fill_rates = []
    inventories = []
    unfulfilled = []
    cvars = []
    
    for r in rows:
        try:
            p = float(r.get(prof_key, ''))
            profits.append(p)
        except (ValueError, TypeError):
            continue
        try: fill_rates.append(float(r.get(fr_key, '')))
        except: pass
        try: inventories.append(float(r.get(inv_key, '')))
        except: pass
        try: unfulfilled.append(float(r.get(unf_key, '')))
        except: pass
        try: cvars.append(float(r.get(cvar_key, '')))
        except: pass
    
    if profits:
        mp = statistics.mean(profits)
        sp = statistics.stdev(profits) if len(profits) > 1 else 0
        mfr = statistics.mean(fill_rates) if fill_rates else 0
        mi = statistics.mean(inventories) if inventories else 0
        mu = statistics.mean(unfulfilled) if unfulfilled else 0
        mc = statistics.mean(cvars) if cvars else 0
        print(f"{agent:<18} {mp:>12.1f} {sp:>8.1f} {mfr:>8.3f} {mi:>10.1f} {mu:>10.1f} {mc:>12.1f} {len(profits):>4}")

print()

# ──────────────────────────────────────────────────────────────
# 2. OPTIMALITY GAP TABLE — per agent
# ──────────────────────────────────────────────────────────────
print("="*80)
print("TABLE 2: MEAN OPTIMALITY GAP (% below Oracle)")
print("="*80)

OPT_AGENTS = []
for col in rows[0].keys():
    if col.endswith("_OptGap_Pct") and not col.startswith("Oracle"):
        agent_name = col.replace("_OptGap_Pct", "")
        OPT_AGENTS.append(agent_name)

print(f"{'Agent':<20} {'Mean Gap%':>10} {'Std Gap%':>10} {'Min Gap%':>10} {'Max Gap%':>10} {'N':>4}")
for agent in sorted(OPT_AGENTS):
    col = f"{agent}_OptGap_Pct"
    gaps = []
    for r in rows:
        try:
            g = float(r.get(col, ''))
            gaps.append(g)
        except:
            pass
    if gaps:
        print(f"{agent:<20} {statistics.mean(gaps):>10.2f} {(statistics.stdev(gaps) if len(gaps)>1 else 0):>10.2f} {min(gaps):>10.2f} {max(gaps):>10.2f} {len(gaps):>4}")

print()

# ──────────────────────────────────────────────────────────────
# 3. BULLWHIP RATIO — per agent
# ──────────────────────────────────────────────────────────────
print("="*80)
print("TABLE 3: BULLWHIP RATIO BY AGENT")
print("="*80)

print(f"{'Agent':<18} {'Mean BWR':>10} {'Std BWR':>10} {'N':>4}")
for agent in AGENTS:
    bwr_key = f"{agent}_BullwhipRatio"
    bwrs = []
    for r in rows:
        try:
            b = float(r.get(bwr_key, ''))
            bwrs.append(b)
        except:
            pass
    if bwrs:
        print(f"{agent:<18} {statistics.mean(bwrs):>10.2f} {(statistics.stdev(bwrs) if len(bwrs)>1 else 0):>10.2f} {len(bwrs):>4}")

print()

# ──────────────────────────────────────────────────────────────
# 4. COMPUTE TIME TABLE — per agent
# ──────────────────────────────────────────────────────────────
print("="*80)
print("TABLE 4: COMPUTATIONAL COST (seconds per scenario)")  
print("="*80)

print(f"{'Agent':<18} {'Mean Time(s)':>12} {'Std':>10} {'N':>4}")
for agent in AGENTS:
    time_key = f"Time_Sec_{agent}"
    times = []
    for r in rows:
        try:
            t = float(r.get(time_key, ''))
            times.append(t)
        except:
            pass
    if times:
        print(f"{agent:<18} {statistics.mean(times):>12.4f} {(statistics.stdev(times) if len(times)>1 else 0):>10.4f} {len(times):>4}")

print()

# ──────────────────────────────────────────────────────────────
# 5. SCENARIO BREAKDOWN by Demand Type
# ──────────────────────────────────────────────────────────────
print("="*80)
print("TABLE 5: PPO-GNN & ORACLE PROFIT BY DEMAND TYPE")
print("="*80)

KEY_AGENTS = ["Oracle", "GNN_V3", "PPO_Transformer", "MSSP", "Residual"]
demand_data = defaultdict(lambda: defaultdict(list))

for r in rows:
    demand = r.get("Demand", "")
    for agent in KEY_AGENTS:
        try:
            p = float(r.get(f"{agent}_Profit", ''))
            demand_data[demand][agent].append(p)
        except:
            pass

for demand in sorted(demand_data.keys()):
    print(f"\nDemand: {demand}")
    for agent in KEY_AGENTS:
        vals = demand_data[demand][agent]
        if vals:
            print(f"  {agent:<12}: Mean={statistics.mean(vals):>8.1f}, Std={statistics.stdev(vals) if len(vals)>1 else 0:>6.1f}, N={len(vals)}")

print()

# ──────────────────────────────────────────────────────────────
# 6. Empirical Oracle gaps and VSS
# ──────────────────────────────────────────────────────────────
print("="*80)
print("TABLE 6: EMPIRICAL ORACLE GAP & STOCHASTIC SOLUTION VALUE")
print("="*80)

for label, col in [
    ("Oracle gap vs MSSP", "OracleGap_MSSP"),
    ("Oracle gap vs MSSP-I", "OracleGap_MSSP_I"),
    ("VSS blind", "VSS_Blind"),
    ("VSS informed", "VSS_Informed"),
]:
    vals = []
    for r in rows:
        try:
            vals.append(float(r.get(col, "")))
        except Exception:
            pass
    if vals:
        print(
            f"{label}: Mean={statistics.mean(vals):.2f}, "
            f"Std={statistics.pstdev(vals):.2f}, Min={min(vals):.2f}, Max={max(vals):.2f}"
        )

print()

# ──────────────────────────────────────────────────────────────
# 7. COST DECOMPOSITION — Revenue, HoldingCost, BacklogPenalty, ProcurementCost
# ──────────────────────────────────────────────────────────────
print("="*80)
print("TABLE 7: COST DECOMPOSITION BY AGENT (aggregated means)")
print("="*80)

DECOMP_AGENTS = ["Oracle", "MSSP", "GNN_V3", "PPO_Transformer", "ST_PPO", "Residual", "PPO_MLP"]
components = ["Revenue", "HoldingCost", "BacklogPenalty", "ProcurementCost", "OperatingCost"]
print(f"{'Agent':<12}", end="")
for c in components:
    print(f" {c:>16}", end="")
print()

for agent in DECOMP_AGENTS:
    print(f"{agent:<12}", end="")
    for comp in components:
        key = f"{agent}_{comp}"
        vals = []
        for r in rows:
            try: vals.append(float(r.get(key, '')))
            except: pass
        if vals:
            print(f" {statistics.mean(vals):>16.1f}", end="")
        else:
            print(f" {'N/A':>16}", end="")
    print()

print()

# ──────────────────────────────────────────────────────────────
# 8. PER-SCENARIO FULL TABLE (for the full results appendix)
# ──────────────────────────────────────────────────────────────
print("="*80)
print("TABLE 8: FULL SCENARIO × AGENT PROFIT MATRIX")
print("="*80)

FULL_AGENTS = ["Oracle", "MSSP", "DLP", "GNN_V3", "PPO_Transformer", "ST_PPO", "Residual", "PPO_MLP", "SAC"]
header = f"{'Scenario':<55}"
for a in FULL_AGENTS:
    header += f" {a:>10}"
print(header)

for r in rows:
    key = r.get("ScenarioKey", "")
    # Shorten key
    short = key.replace("A_Core|", "").replace("|MARL:False", "")
    line = f"{short:<55}"
    for a in FULL_AGENTS:
        try:
            p = float(r.get(f"{a}_Profit", ''))
            line += f" {p:>10.1f}"
        except:
            line += f" {'—':>10}"
    print(line)
