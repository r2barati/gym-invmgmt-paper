#!/usr/bin/env python3
"""Regenerate goodwill_dynamics.png — numerically exact trace of Eq. 5."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Parameters from Eq. 5 ──
gamma_drop = 0.90
gamma_rec  = 1.01
s_min      = 0.2
s_max      = 2.0
s0         = 1.0

# ── Design a stockout scenario (60-step horizon to match Fig 3a scale) ──
T = 60
stockout = np.zeros(T, dtype=bool)
stockout[5:13]  = True   # First stockout: 8 consecutive periods
stockout[45:49] = True   # Second stockout: 4 consecutive periods

# ── Compute exact trajectory ──
s = np.zeros(T + 1)
s[0] = s0
for t in range(T):
    if stockout[t]:
        s[t + 1] = max(s_min, gamma_drop * s[t])
    else:
        s[t + 1] = min(s_max, gamma_rec * s[t])

time = np.arange(T + 1)

# ── Publication-quality styling (400 DPI, matches existing figures) ──
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 14,
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
})

# Near-square aspect ratio to match Fig 3a (~1.07:1)
fig, ax = plt.subplots(figsize=(6, 5.6), dpi=400)

# ── Stockout shaded bands ──
in_stockout = False
start = 0
for t in range(T):
    if stockout[t] and not in_stockout:
        start = t
        in_stockout = True
    elif not stockout[t] and in_stockout:
        ax.axvspan(start - 0.5, t - 0.5, alpha=0.15, color='#D32F2F',
                   zorder=0)
        mid = (start + t) / 2
        ax.text(mid, 1.12, 'Stockout', ha='center', va='bottom',
                fontsize=11, color='#B71C1C', fontweight='bold')
        in_stockout = False
if in_stockout:
    ax.axvspan(start - 0.5, T - 0.5, alpha=0.15, color='#D32F2F', zorder=0)
    mid = (start + T) / 2
    ax.text(mid, 1.12, 'Stockout', ha='center', va='bottom',
            fontsize=11, color='#B71C1C', fontweight='bold')

# ── Main trajectory ──
ax.plot(time, s, color='#1B2A4A', linewidth=2.2, zorder=3)

# ── Reference lines ──
ax.axhline(y=1.0, linestyle='--', color='#757575', linewidth=1.0, zorder=1)
ax.text(T + 1, 1.0, 'Baseline', va='center', fontsize=10, color='#616161')
ax.axhline(y=s_min, linestyle='--', color='#D32F2F', linewidth=1.0, zorder=1)
ax.text(T + 1, s_min, 'Floor', va='center', fontsize=10, color='#C62828')

# ── Rate annotations ──
drop_t = 9
ax.annotate(r'$\times 0.90\,/\,\mathrm{step}$',
            xy=(drop_t, s[drop_t]), xytext=(drop_t + 8, s[drop_t] + 0.15),
            fontsize=11, color='#1B2A4A',
            arrowprops=dict(arrowstyle='->', color='#1B2A4A', lw=1.2))

rec_t = 30
ax.annotate(r'$\times 1.01\,/\,\mathrm{step}$',
            xy=(rec_t, s[rec_t]), xytext=(rec_t + 6, s[rec_t] + 0.15),
            fontsize=11, color='#1B2A4A',
            arrowprops=dict(arrowstyle='->', color='#1B2A4A', lw=1.2))

# ── Axes ──
ax.set_xlabel(r'Time Step $t$', fontsize=14)
ax.set_ylabel(r'Sentiment $s_t$', fontsize=14)
ax.set_xlim(-1, T + 5)
ax.set_ylim(0.0, 1.25)
ax.set_xticks(np.arange(0, T + 1, 10))
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
ax.grid(True, alpha=0.25, linewidth=0.5)

fig.tight_layout()
import os
out = os.path.join(os.path.dirname(__file__), "goodwill_dynamics.png")
fig.savefig(out, dpi=400, bbox_inches='tight', facecolor='white')
print(f'Saved: {out}')
print(f'\nKey values:')
print(f'  After 8-step stockout from s≈1.05: s={s[13]:.4f}')
print(f'  Recovery at t=30 (~17 steps): s={s[30]:.4f}')
print(f'  Recovery at t=45 (~32 steps): s={s[45]:.4f}')
print(f'  After 4-step stockout from s≈{s[45]:.3f}: s={s[49]:.4f}')
print(f'  Final at t={T}: s={s[T]:.4f}')

