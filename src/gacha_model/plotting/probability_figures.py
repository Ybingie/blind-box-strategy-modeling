import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from .style import COLORS, clean_axis, save_figure

def make_probability_figure(cfg, result, run):
    records = result.records
    x = np.array([r['delta'] * 100 for r in records])
    fixed = np.array([r['fixed_min_cost'] for r in records])
    dynamic = np.array([r['dynamic_cost'] for r in records])
    pairs = np.array([r['pair_continue'] for r in records])
    (fig, axes) = plt.subplots(1, 2, figsize=(10.2, 3.7), gridspec_kw={'width_ratios': [1.2, 1]})
    axes[0].plot(
        x,
        fixed,
        color=COLORS['Fixed k=3'],
        linestyle='--',
        label='Optimized fixed'
    )
    axes[0].plot(
        x,
        dynamic,
        color=COLORS['Dynamic stopping'],
        label='Dynamic stopping'
    )
    axes[0].set_ylabel('Expected net cost (CNY)')
    axes[0].legend(loc='upper right')
    for j in (0, 25, 50):
        r = records[j]
        axes[0].scatter(x[j], fixed[j], color=COLORS['Fixed k=3'], s=27, zorder=4)
        axes[0].annotate(
            f"$k^*={r['fixed_best_k']}$",
            (x[j], fixed[j]),
            xytext=(5, 10),
            textcoords='offset points',
            fontsize=9
        )
    axes[0].set_ylim(min(dynamic) - 1.4, max(fixed) + 2.8)
    axes[1].step(x, pairs, where='mid', color=COLORS['Dynamic stopping'], linewidth=2)
    axes[1].set_ylabel('Continue states among 28 pairs')
    axes[1].set_ylim(-1, 29)
    axes[1].yaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    axes[1].scatter([0], [pairs[25]], color=COLORS['Dynamic stopping'], s=30, zorder=4)
    axes[1].annotate(
        f'Current inputs: {pairs[25]}/28',
        (0, pairs[25]),
        xytext=(0.6, 6.5),
        textcoords='data',
        fontsize=9,
        arrowprops={
            'arrowstyle': '-',
            'color': '#89939E',
            'linewidth': 0.8
        }
    )
    for (ax, panel) in zip(axes, ('(a)', '(b)')):
        clean_axis(ax)
        ax.axvline(0, color='#A8AFB8', linestyle=':', linewidth=1)
        ax.set_xlim(-2.6, 2.6)
        ax.set_xticks([-2.5, -1.25, 0, 1.25, 2.5])
        ax.set_xlabel(
            'Probability shift (percentage points)'
        )
        ax.text(0, 1.03, panel, transform=ax.transAxes, fontsize=11, fontweight='bold')
    fig.tight_layout(w_pad=2.4)
    return [
        save_figure(
            fig,
            run.output_root,
            'main',
            'F15',
            'probability_sensitivity',
            'Probability sensitivity',
            'Finite probability scenarios; not estimated product probabilities',
            run.png_dpi
        )
    ]
