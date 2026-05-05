"""
Regenerate Figure 2: Enriched Dual-Track Timeline — v5 PHOTOREALISTIC.
Premium visual effects:
  - Radial-gradient milestone markers with specular highlights
  - Drop shadows on all markers and text
  - Frosted-glass reproducibility gap
  - Gradient-stroked timeline spines
  - Subtle background texture / vignette
  - Dimensional era brackets with glow
  - "Ours" star with outer glow
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, Circle, Shadow
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors
import numpy as np

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(-2, 29.5)
ax.set_ylim(-6.5, 7.5)
ax.axis('off')

fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)

# ═══════════════════════════════════════════════════════════
# HELPER: Draw a marker with drop shadow + specular highlight
# ═══════════════════════════════════════════════════════════
def draw_marker(ax, x, y, marker, color, size, edge_color='white', edge_w=1.5):
    """Draw marker with soft drop shadow and subtle specular."""
    # Drop shadow (offset, larger, blurred)
    shadow_effects = [pe.withSimplePatchShadow(
        offset=(0.6, -0.6), shadow_rgbFace='#1C2833', alpha=0.18)]
    ax.plot(x, y, marker, color=color, markersize=size, zorder=5,
            markeredgecolor=edge_color, markeredgewidth=edge_w,
            path_effects=shadow_effects)
    # Specular highlight (small lighter dot offset top-left)
    if marker == 'o':
        highlight_color = mcolors.to_rgba(color, alpha=0.0)
        r = size / 220
        highlight = Circle((x - r * 0.4, y + r * 0.4), r * 0.35,
                           facecolor='white', edgecolor='none',
                           alpha=0.35, zorder=6)
        ax.add_patch(highlight)

def shadow_text(ax, x, y, text, **kwargs):
    """Text with soft drop shadow."""
    effects = [pe.withStroke(linewidth=3, foreground='white'),
               pe.withSimplePatchShadow(offset=(0.4, -0.4),
                                         shadow_rgbFace='#2C3E50',
                                         alpha=0.08)]
    ax.text(x, y, text, path_effects=effects, **kwargs)

# ═══════════════════════════════════════════════════════════
# TOP TRACK TITLE
# ═══════════════════════════════════════════════════════════
shadow_text(ax, -1.2, 6.8, 'Solution Methods', fontsize=19,
            fontweight='bold', style='italic', color='#1B2631')

# ═══════════════════════════════════════════════════════════
# TOP TRACK: Solution Methods — gradient spine
# ═══════════════════════════════════════════════════════════
y_top = 3.5

# Draw gradient line (dark → medium teal, segmented)
n_seg = 200
xs = np.linspace(-0.5, 27, n_seg)
for i in range(n_seg - 1):
    frac = i / n_seg
    r = 0.106 + frac * (0.173 - 0.106)
    g = 0.310 + frac * (0.553 - 0.310)
    b = 0.447 + frac * (0.522 - 0.447)
    ax.plot(xs[i:i+2], [y_top, y_top], color=(r, g, b), lw=3.5, zorder=1,
            solid_capstyle='round')

# Soft glow behind the line
ax.plot([-0.5, 27], [y_top, y_top], color='#5DADE2', lw=8, alpha=0.08, zorder=0)

top_milestones = [
    (0.0,  '1913', 'EOQ', True),
    (1.9,  '1951', '(s,S)\nPolicies', False),
    (3.4,  '1957', 'Bellman\nDP', False),
    (5.2,  '1958', 'Min-Max\nNewsvendor', True),
    (7.2,  '1960', 'Base-Stock\nOptimality', True),
    (9.5,  '1995', 'Sim.\nOptimization', True),
    (12.0, '2006', 'Robust\nOptimization', True),
    (13.5, '2011', 'DAgger\n(IL)', False),
    (14.8, '2015', 'DQN', False),
    (16.0, '2017', 'DQN\nBeer Game', False),
    (17.2, '2017', 'PPO', False),
    (18.5, '2018', 'SAC /\nResidual RL', False),
    (20.0, '2019', 'Data-Driven\nE2E', True),
    (22.0, '2021', 'MSSP\nvs. RL', True),
    (24.0, '2023', 'DL at\nScale', True),
    (25.5, '2024', 'LLM\nAgents', False),
    (27.0, '2025', 'GNN\nMARL', True),
]

for x, year, label, is_major in top_milestones:
    color = '#1B4F72' if is_major else '#5DADE2'
    size = 14 if is_major else 10
    draw_marker(ax, x, y_top, 'o', color, size)
    # Year above — with subtle shadow
    shadow_text(ax, x, y_top + 0.65, year,
                fontsize=13.5 if is_major else 11.5,
                fontweight='bold' if is_major else 'normal',
                ha='center', color='#1B4F72' if is_major else '#2E86C1')
    # Label below
    shadow_text(ax, x, y_top - 0.65, label,
                fontsize=12 if is_major else 10.5,
                ha='center', va='top', color='#2C3E50',
                fontweight='bold' if is_major else 'normal')

# ═══════════════════════════════════════════════════════════
# ERA BRACKETS with subtle glow
# ═══════════════════════════════════════════════════════════
era_y = y_top + 1.8

def draw_era(ax, x1, x2, label, color, bold=False):
    # Glow behind bracket
    ax.plot([x1, x2], [era_y - 0.15, era_y - 0.15],
            color=color, lw=6, alpha=0.07, solid_capstyle='round', zorder=0)
    ax.annotate('', xy=(x1, era_y - 0.15), xytext=(x2, era_y - 0.15),
                arrowprops=dict(arrowstyle='|-|', color=color, lw=1.3))
    shadow_text(ax, (x1 + x2) / 2, era_y + 0.2, label,
                fontsize=11, ha='center', color=color, style='italic',
                fontweight='bold' if bold else 'normal')

draw_era(ax, 0, 6.5, 'Classical OR', '#85929E')
draw_era(ax, 9, 12, 'Modern OR', '#85929E')
draw_era(ax, 13.5, 18.5, 'Deep RL Revolution', '#2E86C1', bold=True)
draw_era(ax, 20, 27.5, 'Applied RL for Supply Chains', '#148F77', bold=True)

# ═══════════════════════════════════════════════════════════
# REPRODUCIBILITY GAP — frosted glass effect
# ═══════════════════════════════════════════════════════════
# Layered translucent rectangles for depth
for i, (alpha, pad) in enumerate([(0.03, 0.6), (0.06, 0.4), (0.12, 0.3)]):
    gap = FancyBboxPatch((19 - i * 0.1, -5.5 - i * 0.05),
                          7 + i * 0.2, 8.2 + i * 0.1,
                          boxstyle=f"round,pad={pad}",
                          facecolor='#D5D8DC', edgecolor='none',
                          alpha=alpha, zorder=-1 + i * 0.1)
    ax.add_patch(gap)

# Main frosted panel
gap_rect = FancyBboxPatch((19, -5.5), 7, 8.2,
                           boxstyle="round,pad=0.3",
                           facecolor='#F2F3F4', edgecolor='#D5D8DC',
                           alpha=0.45, linewidth=0.8, zorder=0)
ax.add_patch(gap_rect)

# Inner subtle border for glass edge
gap_inner = FancyBboxPatch((19.15, -5.35), 6.7, 7.9,
                            boxstyle="round,pad=0.25",
                            facecolor='none', edgecolor='white',
                            alpha=0.3, linewidth=0.6, zorder=0.5)
ax.add_patch(gap_inner)

shadow_text(ax, 22.5, -0.3, 'Reproducibility\nGap', fontsize=16,
            ha='center', color='#ABB2B9', style='italic',
            fontweight='bold', alpha=0.7)

# ═══════════════════════════════════════════════════════════
# BOTTOM TRACK: Benchmarking Infrastructure
# ═══════════════════════════════════════════════════════════
y_bot = -3.5
shadow_text(ax, -1.2, -1.5, 'Benchmarking\nInfrastructure', fontsize=19,
            fontweight='bold', style='italic', color='#148F77')

# Gradient spine for bottom track
n_seg = 200
xs = np.linspace(5, 27, n_seg)
for i in range(n_seg - 1):
    frac = i / n_seg
    r = 0.078 + frac * (0.055 - 0.078)
    g = 0.561 + frac * (0.386 - 0.561)
    b = 0.467 + frac * (0.318 - 0.467)
    ax.plot(xs[i:i+2], [y_bot, y_bot], color=(r, g, b), lw=3.5, zorder=1,
            solid_capstyle='round')

# Soft glow behind bottom line
ax.plot([5, 27], [y_bot, y_bot], color='#1ABC9C', lw=8, alpha=0.07, zorder=0)

# Dashed pre-1960 line with shadow
ax.plot([2, 5], [y_bot, y_bot], color='#148F77', lw=2.5, linestyle='--', zorder=1,
        path_effects=[pe.withSimplePatchShadow(offset=(0.3, -0.3),
                       shadow_rgbFace='#0E6251', alpha=0.1)])

bot_milestones = [
    (2.5,  '', 'Bespoke\nSimulators', False),
    (5.5,  '1960', 'Beer\nGame', False),
    (8.0,  '2016', 'OpenAI\nGym', False),
    (10.5, '2019', 'ORL', False),
    (12.5, '2020', 'OR-Gym', True),
    (14.5, '2020', 'MARO', False),
    (16.5, '2022', 'Deep IM', True),
    (18.5, '2022', 'DRL\nInv.', False),
    (21.0, '2023', 'MABIM\n(MARL)', True),
    (24.0, '2024', 'InvAgent\n(LLMs)', True),
]

for x, year, label, is_major in bot_milestones:
    color = '#148F77'
    size = 14 if is_major else 10
    draw_marker(ax, x, y_bot, 'D', color, size)
    shadow_text(ax, x, y_bot - 0.7, year,
                fontsize=12 if is_major else 10.5,
                fontweight='bold' if is_major else 'normal',
                ha='center', color=color)
    shadow_text(ax, x, y_bot - 1.4, label,
                fontsize=11.5 if is_major else 10,
                ha='center', va='top', color='#2C3E50',
                fontweight='bold' if is_major else 'normal')

# ═══════════════════════════════════════════════════════════
# "OURS" STAR with outer glow
# ═══════════════════════════════════════════════════════════
# Outer glow rings
for r_mult, a in [(36, 0.04), (33, 0.07), (31, 0.10)]:
    ax.plot(27.0, y_bot, '*', color='#1ABC9C', markersize=r_mult,
            alpha=a, zorder=4)

# Main star with shadow
ax.plot(27.0, y_bot, '*', color='#148F77', markersize=30, zorder=5,
        markeredgecolor='#0E6251', markeredgewidth=1.5,
        path_effects=[pe.withSimplePatchShadow(
            offset=(0.8, -0.8), shadow_rgbFace='#0E6251', alpha=0.2)])

shadow_text(ax, 27.0, y_bot - 0.7, '2026', fontsize=13, fontweight='bold',
            ha='center', color='#0E6251')

# "Ours" badge with shadow
badge = FancyBboxPatch((25.5, y_bot - 2.65), 3.0, 1.15,
                        boxstyle='round,pad=0.2',
                        facecolor='#D5F5E3', edgecolor='#148F77',
                        alpha=0.85, linewidth=1.2, zorder=4)
badge_shadow = FancyBboxPatch((25.6, y_bot - 2.75), 3.0, 1.15,
                               boxstyle='round,pad=0.2',
                               facecolor='#1C2833', edgecolor='none',
                               alpha=0.08, zorder=3)
ax.add_patch(badge_shadow)
ax.add_patch(badge)
shadow_text(ax, 27.0, y_bot - 1.4, 'gym-invmgmt\n(Ours)', fontsize=12,
            ha='center', va='top', color='#0E6251', fontweight='bold')

# Dashed vertical connecting line with shadow
ax.plot([27.0, 27.0], [y_bot + 0.4, y_top - 0.4], '--', color='#ABB2B9',
        lw=1.2, zorder=0,
        path_effects=[pe.withSimplePatchShadow(
            offset=(0.3, -0.3), shadow_rgbFace='#2C3E50', alpha=0.06)])

plt.tight_layout()
import os
out_path = os.path.join(os.path.dirname(__file__), "literature_timeline.png")
fig.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
print(f"Saved to: {out_path}")
plt.close()
