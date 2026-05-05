#!/usr/bin/env python3
"""
Regenerate appendix figures from the canonical benchmark CSV.

Source: results/benchmark_final_merged.csv unless BENCHMARK_CSV is set.
Output: paper/figures/B1_*.png, B2_*.png, B3_*.png, E4_*.png, T4_*.png

All figures use transparent backgrounds for LaTeX compatibility.
"""

import csv
import os
import sys
import numpy as np

# ── Ensure matplotlib works in headless mode ──
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch

# ── Paths ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CANONICAL_CSV = os.path.abspath(
    os.path.join(SCRIPT_DIR, '..', '..', 'results', 'benchmark_final_merged.csv')
)

CSV_PATH = os.environ.get('BENCHMARK_CSV', None)
if not CSV_PATH:
    CSV_PATH = CANONICAL_CSV
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Could not find benchmark CSV at {CSV_PATH}. "
        "Set BENCHMARK_CSV to intentionally use a non-canonical artifact."
    )
OUT_DIR = SCRIPT_DIR  # Same directory as the script

# ── Core agents for figures (one per paradigm) ──
# Oracle, OR-exact, OR-heuristic, AdvancedRL(GNN), AdvancedRL(Transformer),
# Hybrid, ImitationLearning
CORE_AGENTS = ['Oracle', 'MSSP', 'ExpSmoothing', 'GNN_V3', 'PPO_Transformer', 'ST_PPO', 'DAgger_B']
CORE_LABELS = ['Oracle', 'MSSP', 'ExpSmooth', 'PPO-GNN', 'PPO-T', 'ST-PPO', 'DAgger-B']
CORE_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#2D6A4F', '#52B788',
               '#E07A5F', '#7B68EE']

# Smaller set for focused figures
FOCUS_AGENTS = ['Oracle', 'MSSP', 'GNN_V3', 'PPO_Transformer', 'Residual', 'DAgger_B']
FOCUS_LABELS = ['Oracle', 'MSSP', 'PPO-GNN', 'PPO-T', 'Residual', 'DAgger-B']
FOCUS_COLORS = ['#2E86AB', '#A23B72', '#2D6A4F', '#52B788', '#E07A5F', '#7B68EE']

FIGURE_REQUIRED_COLUMNS = sorted({
    'MARL', 'Network', 'Demand', 'Goodwill', 'Backlog',
    *(f'{agent}_Profit' for agent in set(CORE_AGENTS + FOCUS_AGENTS)),
    *(f'{agent}_AvgInv' for agent in FOCUS_AGENTS),
    *(f'{agent}_Unfulfilled' for agent in FOCUS_AGENTS),
    *(f'{agent}_FillRate' for agent in FOCUS_AGENTS),
})

# ── Global plot style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 200,
})


def load_data(include_marl=False):
    """Load and parse the benchmark CSV.

    By default, excludes the 4 supplementary MARL scenarios so that
    headline averages match the paper's 22-scenario main grid.
    """
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    total = len(rows)
    if not include_marl:
        rows = [r for r in rows if r.get('MARL', 'False') != 'True']
    print(f"Loaded {len(rows)}/{total} scenarios from {os.path.basename(CSV_PATH)}"
          f"{'' if include_marl else ' (excluding MARL supplementary)'}")
          
    # Strict column guard for every metric used by generated figures.
    if len(rows) > 0:
        missing = [c for c in FIGURE_REQUIRED_COLUMNS if c not in rows[0]]
        if missing:
            print(f"\n[ERROR] Missing required columns in CSV: {missing}")
            print("Run all figure agents, then --merge, before generating figures.")
            sys.exit(1)
            
    return rows


def safe_float(val, default=np.nan):
    """Safely convert to float."""
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (ValueError, TypeError):
        return default


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 1: Scenario-Level Performance Heatmap (B1)
# ══════════════════════════════════════════════════════════════════════
def gen_heatmap(rows):
    """Full 26-scenario × 7-agent heatmap with optimality gap coloring."""
    agents = CORE_AGENTS
    labels = CORE_LABELS

    # Build data matrix
    scenario_labels = []
    profit_matrix = []
    for r in rows:
        # Build readable label
        net = r['Network']
        dem = r['Demand'].replace('trend+seasonal+shock', 'T+S+Sh')\
                          .replace('trend+seasonal', 'T+S')\
                          .replace('M5_volatile', 'M5')
        gw = 'GW' if r['Goodwill'] == 'True' else 'NoGW'
        bl = 'BL' if r['Backlog'] == 'True' else 'LS'
        scenario_labels.append(f"{net}·{dem}·{gw}·{bl}")

        row_vals = []
        for a in agents:
            row_vals.append(safe_float(r.get(f'{a}_Profit', ''), np.nan))
        profit_matrix.append(row_vals)

    profit = np.array(profit_matrix)
    oracle_col = profit[:, 0]  # Oracle is first column

    # Compute optimality gap (%)
    gap = np.zeros_like(profit)
    for j in range(profit.shape[1]):
        gap[:, j] = np.where(oracle_col > 0,
                             (1 - profit[:, j] / oracle_col) * 100, 0)
    gap = np.clip(gap, 0, 100)

    # Sort scenarios by block
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 16))

    # Custom colormap: green (low gap) → yellow → red (high gap)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'gap', ['#d4edda', '#fff3cd', '#f8d7da', '#c62828'], N=256)

    im = ax.imshow(gap, cmap=cmap, aspect='auto', vmin=0, vmax=100)

    # Annotate cells with profit values
    for i in range(len(scenario_labels)):
        for j in range(len(agents)):
            val = profit[i, j]
            if np.isnan(val):
                txt = '—'
            else:
                txt = f'{val:.0f}'
            text_color = 'white' if gap[i, j] > 60 else 'black'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color=text_color)

    # Find best non-Oracle per row and add green border
    for i in range(len(scenario_labels)):
        non_oracle = profit[i, 1:]  # exclude Oracle
        if not np.all(np.isnan(non_oracle)):
            best_j = np.nanargmax(non_oracle) + 1  # +1 for Oracle offset
            rect = plt.Rectangle((best_j - 0.5, i - 0.5), 1, 1,
                                 linewidth=2.5, edgecolor='#28a745',
                                 facecolor='none')
            ax.add_patch(rect)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_yticks(range(len(scenario_labels)))
    ax.set_yticklabels(scenario_labels, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Optimality Gap (%)', fontsize=11)

    ax.set_title('Scenario-Level Performance Heatmap\n'
                 '(Values = Profit $, Color = Gap from Oracle, '
                 'Green Border = Best Non-Oracle)',
                 fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'B1_scenario_heatmap.png')
    fig.savefig(out, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 2: Model Recommendation Matrix (B2)
# ══════════════════════════════════════════════════════════════════════
def gen_recommendation(rows):
    """4-panel recommendation matrix by operating condition axis."""
    agents = FOCUS_AGENTS[1:]   # Exclude Oracle
    labels = FOCUS_LABELS[1:]
    colors = FOCUS_COLORS[1:]

    axes_config = [
        ('By Demand', 'Demand', None),
        ('By Topology', 'Network', None),
        ('By Goodwill', 'Goodwill', {'True': 'With Goodwill', 'False': 'No Goodwill'}),
        ('By Backlog', 'Backlog', {'True': 'Backlog', 'False': 'Lost Sales'}),
    ]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    for idx, (title, key, rename) in enumerate(axes_config):
        ax = axs[idx]
        # Group scenarios by this axis
        groups = {}
        for r in rows:
            g = r[key]
            if rename:
                g = rename.get(g, g)
            if g not in groups:
                groups[g] = []
            groups[g].append(r)

        group_names = sorted(groups.keys())
        x = np.arange(len(group_names))
        width = 0.12
        n = len(agents)

        all_panel_means = []
        for i, (agent, label, color) in enumerate(zip(agents, labels, colors)):
            means = []
            for gn in group_names:
                vals = [safe_float(r.get(f'{agent}_Profit', ''))
                        for r in groups[gn]]
                vals = [v for v in vals if not np.isnan(v)]
                means.append(np.mean(vals) if vals else 0)
            all_panel_means.extend(means)
            offset = (i - n/2 + 0.5) * width
            bars = ax.bar(x + offset, means, width, label=label, color=color,
                          edgecolor='white', linewidth=0.5)

            # Mark best per group
            for gj, gn in enumerate(group_names):
                all_means = []
                for a2 in agents:
                    v2 = [safe_float(r.get(f'{a2}_Profit', ''))
                          for r in groups[gn]]
                    v2 = [v for v in v2 if not np.isnan(v)]
                    all_means.append(np.mean(v2) if v2 else 0)
                if i == np.argmax(all_means):
                    ax.text(x[gj] + offset, means[gj] + 20, 'BEST',
                            ha='center', va='bottom', fontsize=7,
                            fontweight='bold', color='#28a745')

        # Shorten demand labels
        display_names = []
        for gn in group_names:
            dn = gn.replace('trend+seasonal+shock', 'T+S+Shock')\
                   .replace('trend+seasonal', 'T+Seasonal')\
                   .replace('M5_volatile', 'M5 Volatile')
            display_names.append(dn)

        ax.set_xticks(x)
        ax.set_xticklabels(display_names, fontsize=9)
        ax.set_ylabel('Profit ($)')
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.legend(fontsize=7, ncol=2, loc='best')
        finite_means = [v for v in all_panel_means if np.isfinite(v)]
        if finite_means:
            ymin = min(finite_means)
            ymax = max(finite_means)
            pad = max((ymax - ymin) * 0.12, 50.0)
            ax.set_ylim(min(0, ymin - pad), ymax + pad)
        ax.axhline(0, color='#444444', linewidth=0.8, alpha=0.5)

    fig.suptitle('Model Recommendation by Operating Condition\n'
                 '(□ = Best Agent)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'B2_recommendation_matrix.png')
    fig.savefig(out, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 3: Goodwill Impact (B3)
# ══════════════════════════════════════════════════════════════════════
def gen_goodwill(rows):
    """Matched-pair bar chart: No-Goodwill vs With-Goodwill per agent."""
    agents = FOCUS_AGENTS
    labels = FOCUS_LABELS
    bar_colors = ['#4A6FA5', '#C25B56']  # blue = NoGW, red = GW

    # Match each goodwill row to the same topology/demand/backlog row with
    # goodwill disabled. Stationary, M5, and MARL rows are intentionally not
    # included because they have no goodwill counterpart in the scenario grid.
    no_gw_index = {
        (r['Network'], r['Demand'], r['Backlog']): r
        for r in rows
        if r['Goodwill'] == 'False' and r.get('MARL', 'False') != 'True'
    }
    pairs = []
    for gw_row in rows:
        if gw_row['Goodwill'] != 'True':
            continue
        key = (gw_row['Network'], gw_row['Demand'], gw_row['Backlog'])
        if key in no_gw_index:
            pairs.append((no_gw_index[key], gw_row))

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(agents))
    width = 0.32

    means_no, means_gw = [], []
    for agent in agents:
        vn = [safe_float(r0.get(f'{agent}_Profit', '')) for r0, _ in pairs]
        vn = [v for v in vn if not np.isnan(v)]
        vg = [safe_float(r1.get(f'{agent}_Profit', '')) for _, r1 in pairs]
        vg = [v for v in vg if not np.isnan(v)]
        means_no.append(np.mean(vn) if vn else 0)
        means_gw.append(np.mean(vg) if vg else 0)

    bars1 = ax.bar(x - width/2, means_no, width, label='No Goodwill',
                   color=bar_colors[0], edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, means_gw, width, label='With Goodwill',
                   color=bar_colors[1], edgecolor='white', linewidth=0.5)

    # Add % change annotations
    for i, (mn, mg) in enumerate(zip(means_no, means_gw)):
        if mn > 0:
            pct = (mg - mn) / mn * 100
            color = '#28a745' if pct > 0 else '#dc3545'
            sign = '+' if pct > 0 else ''
            ax.text(x[i] + width/2, mg + 15, f'{sign}{pct:.0f}%',
                    ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_ylabel('Mean Profit ($)', fontsize=12)
    ax.set_title('Matched Impact of Endogenous Goodwill',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'B3_goodwill_impact.png')
    fig.savefig(out, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 4: Operational Decomposition (E4)
# ══════════════════════════════════════════════════════════════════════
def gen_decomposition(rows):
    """3-panel: Profit, AvgInventory, Unfulfilled — horizontal bars."""
    agents = FOCUS_AGENTS
    labels = FOCUS_LABELS
    colors = FOCUS_COLORS

    metrics = [
        ('Profit ($)', '_Profit', '$'),
        ('Avg Inventory (units)', '_AvgInv', ''),
        ('Unfulfilled Demand (units)', '_Unfulfilled', ''),
    ]

    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    for mi, (title, suffix, prefix) in enumerate(metrics):
        ax = axs[mi]
        means = []
        for agent in agents:
            vals = [safe_float(r.get(f'{agent}{suffix}', '')) for r in rows]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(np.mean(vals) if vals else 0)

        y = np.arange(len(agents))
        bars = ax.barh(y, means, color=colors, edgecolor='white', linewidth=0.5)

        # Annotate
        for i, (bar, val) in enumerate(zip(bars, means)):
            ax.text(bar.get_width() + max(means)*0.01, bar.get_y() + bar.get_height()/2,
                    f'{prefix}{val:.0f}', va='center', fontsize=9,
                    fontweight='bold', color='white' if val < 0 else colors[i])

        ax.set_yticks(y)
        ax.set_yticklabels(labels if mi == 0 else [''] * len(labels))
        ax.set_title(title, fontweight='bold', fontsize=11)
        ax.invert_yaxis()

    fig.suptitle('Operational Decomposition — How Agents Achieve Profit',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'E4_reward_decomposition.png')
    fig.savefig(out, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
#  FIGURE 5: Fill Rate Distribution (T4)
# ══════════════════════════════════════════════════════════════════════
def gen_fill_rate(rows):
    """Violin plot of scenario-level mean fill-rate distributions."""
    agents = FOCUS_AGENTS
    labels = FOCUS_LABELS
    colors = FOCUS_COLORS

    fig, ax = plt.subplots(figsize=(12, 6))

    # Collect scenario-level mean fill rates per agent from the merged CSV.
    data = []
    for agent in agents:
        vals = [safe_float(r.get(f'{agent}_FillRate', '')) for r in rows]
        vals = [v for v in vals if not np.isnan(v)]
        data.append(vals if vals else [0])

    positions = np.arange(len(agents))
    parts = ax.violinplot(data, positions=positions, showmeans=False,
                          showmedians=False, showextrema=False)

    # Color each violin
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.75)
        pc.set_edgecolor('white')
        pc.set_linewidth(1)

    # Add median markers and annotations
    for i, d in enumerate(data):
        med = np.median(d)
        ax.scatter(i, med, color='white', s=30, zorder=5, edgecolors='black',
                   linewidth=1)
        ax.text(i, med + 0.015, f'{med:.2f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    # Perfect service line
    ax.axhline(y=1.0, color='#52B788', linestyle='--', alpha=0.5,
               label='Perfect Service')

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontweight='bold')
    ax.set_ylabel('Fill Rate (higher = better)', fontsize=12)
    ax.set_title('Scenario-Mean Fill Rate Distribution by Agent',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0.45, 1.08)
    ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, 'T4_fill_rate_distribution.png')
    fig.savefig(out, dpi=200, transparent=True, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {out}")


# ══════════════════════════════════════════════════════════════════════
#  TABLE DATA: Cost Decomposition, Compute Times, Oracle Gap/VSS Analogue
# ══════════════════════════════════════════════════════════════════════
def compute_tables(rows):
    """Compute corrected table values from the canonical CSV."""
    print("\n" + "="*70)
    print("TABLE DATA (from canonical CSV — use these to audit main.tex)")
    print("="*70)

    # ── Cost Decomposition ──
    # Note: Use GNN_V3 for cost columns (PPO-GNN data)
    COST_AGENTS = ['Oracle', 'MSSP', 'GNN_V3', 'PPO_Transformer', 'Residual', 'DAgger_B']
    COST_LABELS = ['Oracle', 'MSSP', 'PPO-GNN', 'PPO-T', 'Residual', 'DAgger-B']
    print("\n── Table H: Cost Decomposition ──")
    print(f"{'Agent':15s} {'Revenue':>10s} {'Holding':>10s} {'Backlog':>10s} {'Procurement':>14s} {'Operating':>12s}")
    print("-" * 75)
    for agent, label in zip(COST_AGENTS, COST_LABELS):
        rev = np.nanmean([safe_float(r.get(f'{agent}_Revenue', '')) for r in rows])
        hold = np.nanmean([safe_float(r.get(f'{agent}_HoldingCost', '')) for r in rows])
        back = np.nanmean([safe_float(r.get(f'{agent}_BacklogPenalty', '')) for r in rows])
        proc = np.nanmean([safe_float(r.get(f'{agent}_ProcurementCost', '')) for r in rows])
        oper = np.nanmean([safe_float(r.get(f'{agent}_OperatingCost', '')) for r in rows])
        print(f"{label:15s} {rev:10.0f} {hold:10.0f} {back:10.0f} {proc:14.0f} {oper:12.0f}")

    # ── Compute Times ──
    print("\n── Table K: Compute Times ──")
    all_agents_time = ['Oracle', 'sS', 'ExpSmoothing', 'Newsvendor',
                       'PPO_MLP', 'GNN_V3', 'ST_PPO', 'Residual', 'SAC',
                       'DAgger_B', 'DAgger_G', 'DLP', 'MSSP']
    all_labels_time = ['Oracle', '(s,S)', 'ExpSmoothing', 'Newsvendor',
                       'PPO-MLP', 'PPO-GNN', 'ST-PPO', 'Residual', 'SAC',
                       'DAgger-B', 'DAgger-G', 'DLP', 'MSSP']

    mssp_time = np.nanmean([safe_float(r.get('Time_Sec_MSSP', '')) for r in rows])
    print(f"{'Agent':15s} {'Mean Time(s)':>12s} {'Speedup':>10s}")
    print("-" * 40)
    times_list = []
    for agent, label in zip(all_agents_time, all_labels_time):
        t = np.nanmean([safe_float(r.get(f'Time_Sec_{agent}', '')) for r in rows])
        speedup = mssp_time / t if t > 0 else float('inf')
        times_list.append((t, label, speedup))

    # Sort by time
    times_list.sort(key=lambda x: x[0])
    for t, label, speedup in times_list:
        sp_str = f"{speedup:,.0f}x" if speedup < 1e6 else "∞"
        print(f"{label:15s} {t:12.3f} {sp_str:>10s}")

    # ── Empirical Oracle gap / VSS analogue ──
    print("\n── Table L: Empirical Oracle gap / VSS analogue ──")
    metrics = {
        'Oracle gap vs MSSP ($)': np.array([
            safe_float(r.get('OracleGap_MSSP', '')) for r in rows
        ]),
        'Oracle gap vs MSSP-I ($)': np.array([
            safe_float(r.get('OracleGap_MSSP_I', '')) for r in rows
        ]),
        'VSS analogue blind ($)': np.array([
            safe_float(r.get('VSS_Blind', '')) for r in rows
        ]),
        'VSS analogue informed ($)': np.array([
            safe_float(r.get('VSS_Informed', '')) for r in rows
        ]),
    }
    print(f"{'Metric':10s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    for label, values in metrics.items():
        values = values[np.isfinite(values)]
        print(
            f"{label:24s} {np.mean(values):8.2f} {np.std(values):8.2f} "
            f"{np.min(values):8.2f} {np.max(values):8.2f}"
        )

    # ── Summary performance by paradigm ──
    print("\n── Overall Performance (% of Oracle) ──")
    all_display = [('MSSP', 'MSSP'), ('DLP', 'DLP'),
                   ('ExpSmoothing', 'ExpSmooth'), ('sS', '(s,S)'),
                   ('Newsvendor', 'Newsvendor'),
                   ('GNN_V3', 'PPO-GNN'), ('PPO_Transformer', 'PPO-Transformer'),
                   ('ST_PPO', 'ST-PPO'),
                   ('Residual', 'Residual'), ('SAC', 'SAC'),
                   ('DAgger_B', 'DAgger-B'), ('DAgger_G', 'DAgger-G')]
    for agent, label in sorted(all_display,
                                key=lambda x: -np.nanmean(
                                    [safe_float(r.get(f'{x[0]}_Profit',''))/safe_float(r.get('Oracle_Profit',''),1)*100
                                     for r in rows])):
        pcts = [safe_float(r.get(f'{agent}_Profit', '')) /
                safe_float(r.get('Oracle_Profit', ''), 1) * 100
                for r in rows]
        pcts = [p for p in pcts if not np.isnan(p)]
        print(f"  {label:15s}: {np.mean(pcts):5.1f}% of Oracle")


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 70)
    print("Regenerating appendix figures from canonical benchmark data")
    print("=" * 70)

    rows = load_data()

    print("\nGenerating figures...")
    gen_heatmap(rows)
    gen_recommendation(rows)
    gen_goodwill(rows)
    gen_decomposition(rows)
    gen_fill_rate(rows)

    compute_tables(rows)

    print("\nOK All figures regenerated from canonical benchmark data.")
    print("   Next: use the printed values only as an audit aid for matching-scope tables.")
