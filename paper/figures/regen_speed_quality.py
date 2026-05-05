"""Regenerate Figure 5a: Speed–Quality Pareto Frontier with improved label visibility."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANONICAL_CSV = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..", "..", "results", "benchmark_final_merged.csv")
)

CSV_PATH = os.environ.get('BENCHMARK_CSV', None)
if not CSV_PATH:
    CSV_PATH = CANONICAL_CSV
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Could not find benchmark CSV at {CSV_PATH}. "
        "Set BENCHMARK_CSV to intentionally use a non-canonical artifact."
    )
df = pd.read_csv(CSV_PATH)

# Agents to plot with their display names
agents = {
    'Oracle': ('Oracle_Profit', 'Time_Sec_Oracle'),
    'MSSP': ('MSSP_Profit', 'Time_Sec_MSSP'),
    'DLP': ('DLP_Profit', 'Time_Sec_DLP'),
    'Newsvendor': ('Newsvendor_Profit', 'Time_Sec_Newsvendor'),
    '(s,S)': ('sS_Profit', 'Time_Sec_sS'),
    'ExpSmooth': ('ExpSmoothing_Profit', 'Time_Sec_ExpSmoothing'),
    'DAgger-B': ('DAgger_B_Profit', 'Time_Sec_DAgger_B'),
    'DAgger-G': ('DAgger_G_Profit', 'Time_Sec_DAgger_G'),
    'Residual': ('Residual_Profit', 'Time_Sec_Residual'),
    'PPO-MLP': ('PPO_MLP_Profit', 'Time_Sec_PPO_MLP'),
    'PPO-GNN': ('GNN_V3_Profit', 'Time_Sec_GNN_V3'),
    'PPO-Transformer': ('PPO_Transformer_Profit', 'Time_Sec_PPO_Transformer'),
    'ST-PPO': ('ST_PPO_Profit', 'Time_Sec_ST_PPO'),
    'SAC': ('SAC_Profit', 'Time_Sec_SAC'),
}

# Strict column guard for every plotted profit/timing pair.
required_cols = ['MARL']
for profit_col, time_col in agents.values():
    required_cols.extend([profit_col, time_col])
missing = [c for c in required_cols if c not in df.columns]
if missing:
    import sys
    print(f"\n[ERROR] Missing required columns in CSV: {missing}")
    print("Run all speed-quality agents, then --merge, before generating figures.")
    sys.exit(1)

# Exclude MARL supplementary scenarios — headline figures use the 22-scenario main grid
df = df[df['MARL'] != True]
expected_rows = len(df)

# Calculate mean profit and mean inference time across all scenarios
results = {}
for name, (profit_col, time_col) in agents.items():
    paired = df[[profit_col, time_col]].dropna()
    if len(paired) != expected_rows:
        import sys
        print(
            f"\n[ERROR] {name} has {len(paired)}/{expected_rows} paired profit/time rows.",
            file=sys.stderr,
        )
        sys.exit(1)
    mean_profit = paired[profit_col].mean()
    mean_time_ms = paired[time_col].mean() * 1000  # convert to ms
    if mean_time_ms <= 0:
        import sys
        print(f"\n[ERROR] {name} has non-positive mean timing: {mean_time_ms}", file=sys.stderr)
        sys.exit(1)
    results[name] = (mean_time_ms, mean_profit)

print("Agent results (time_ms, profit):")
for name, (t, p) in sorted(results.items(), key=lambda x: x[1][0]):
    print(f"  {name:12s}: {t:10.1f} ms, ${p:10.1f}")

# Define styling by paradigm
paradigm_style = {
    'Oracle':     {'color': '#FFD700', 'marker': '*', 'size': 250, 'paradigm': 'Oracle'},
    'MSSP':       {'color': '#E74C3C', 'marker': 'o', 'size': 150, 'paradigm': 'Exact OR'},
    'DLP':        {'color': '#E74C3C', 'marker': 's', 'size': 120, 'paradigm': 'Exact OR'},
    'Newsvendor': {'color': '#95A5A6', 'marker': 'D', 'size': 100, 'paradigm': 'Heuristic'},
    '(s,S)':      {'color': '#95A5A6', 'marker': '^', 'size': 100, 'paradigm': 'Heuristic'},
    'ExpSmooth':  {'color': '#95A5A6', 'marker': 'v', 'size': 100, 'paradigm': 'Heuristic'},
    'DAgger-B':   {'color': '#F39C12', 'marker': 'P', 'size': 140, 'paradigm': 'Imitation'},
    'DAgger-G':   {'color': '#F39C12', 'marker': 'X', 'size': 140, 'paradigm': 'Imitation'},
    'Residual':   {'color': '#2ECC71', 'marker': 's', 'size': 150, 'paradigm': 'Hybrid RL'},
    'PPO-MLP':    {'color': '#3498DB', 'marker': 'o', 'size': 120, 'paradigm': 'Standard RL'},
    'PPO-GNN':    {'color': '#8E44AD', 'marker': 'h', 'size': 180, 'paradigm': 'Advanced RL'},
    'PPO-Transformer': {'color': '#8E44AD', 'marker': 'p', 'size': 150, 'paradigm': 'Advanced RL'},
    'ST-PPO':     {'color': '#8E44AD', 'marker': 'H', 'size': 150, 'paradigm': 'Advanced RL'},
    'SAC':        {'color': '#3498DB', 'marker': 'D', 'size': 120, 'paradigm': 'Standard RL'},
}

fig, ax = plt.subplots(figsize=(10, 7))

# Plot each agent
for name, (time_ms, profit) in results.items():
    style = paradigm_style.get(name, {'color': 'gray', 'marker': 'o', 'size': 100})
    ax.scatter(time_ms, profit, c=style['color'], marker=style['marker'],
               s=style['size'], edgecolors='black', linewidths=0.8,
               zorder=5, label=f"{name}")

# Smart label placement with adjustText-style offsets
label_offsets = {
    'Oracle':     (0, 80),
    'MSSP':       (15, -90),
    'DLP':        (15, 70),
    'Newsvendor': (15, -80),
    '(s,S)':      (15, 60),
    'ExpSmooth':  (-60, -80),
    'DAgger-B':   (15, 70),
    'DAgger-G':   (-70, -70),
    'Residual':   (-80, 80),
    'PPO-MLP':    (15, -70),
    'PPO-GNN':    (20, 100),
    'PPO-Transformer': (-80, -70),
    'ST-PPO':     (15, -90),
    'SAC':        (15, -70),
}

for name, (time_ms, profit) in results.items():
    dx, dy = label_offsets.get(name, (15, 0))
    style = paradigm_style.get(name, {'color': 'gray'})
    ax.annotate(name, (time_ms, profit), fontsize=8.5, fontweight='bold',
                xytext=(dx, dy), textcoords='offset points',
                color=style['color'],
                arrowprops=dict(arrowstyle='->', color=style['color'],
                                lw=0.8, connectionstyle='arc3,rad=0.1'),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=style['color'], alpha=0.85))

# Desirable direction arrow
ax.annotate('', xy=(0.08, 0.92), xytext=(0.25, 0.72),
            xycoords='axes fraction', textcoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='#27AE60', lw=2.5))
ax.text(0.17, 0.82, 'Better\n(Fast & Profitable)', transform=ax.transAxes,
        fontsize=10, fontweight='bold', color='#27AE60', ha='center',
        fontstyle='italic')

ax.set_xscale('log')
ax.set_xlabel('Average Episode Inference Time (ms) — Log Scale', fontsize=11, fontweight='bold')
ax.set_ylabel('Average Episode Profit ($)', fontsize=11, fontweight='bold')
ax.set_title('Performance vs. Inference Cost (Speed–Quality Frontier)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.tick_params(labelsize=10)

# Paradigm legend
from matplotlib.lines import Line2D
paradigm_colors = {
    'Oracle': '#FFD700', 'Exact OR': '#E74C3C', 'Heuristic': '#95A5A6',
    'Imitation': '#F39C12', 'Hybrid RL': '#2ECC71',
    'Standard RL': '#3498DB', 'Advanced RL': '#8E44AD'
}
legend_elements = [Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=c, markersize=10,
                          markeredgecolor='black', label=p)
                   for p, c in paradigm_colors.items()]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9,
          framealpha=0.9, edgecolor='#ccc', title='Paradigm',
          title_fontsize=10)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "D_speed_vs_quality.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nSaved to: {out_path}")
plt.close()
