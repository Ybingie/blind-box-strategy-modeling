import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ..uncertainty import load_price_samples
from .style import COLORS, clean_axis, save_figure

def make_data_figures(cfg, collect, sims, robustness, run):
    out = []
    samples = load_price_samples(cfg)
    rng = np.random.default_rng(cfg.base_seed + 101)
    y = np.arange(cfg.n)
    medians = np.asarray(cfg.second_hand_prices)
    resale = np.asarray(cfg.expected_resale_values)
    ci = np.asarray(robustness.median_ci)
    xmax = max((max(x) for x in samples)) + 22
    (fig, ax) = plt.subplots(figsize=(8.7, 5.0))
    for (i, values) in enumerate(samples):
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax.scatter(values, i + jitter, s=16, color='#9DB1C5', alpha=0.48, linewidths=0, zorder=1)
        ax.plot(ci[i], [i, i], color=COLORS['Fixed k=3'], lw=4, solid_capstyle='round', zorder=2)
        ax.text(xmax - 1, i, f'n={len(values)}', va='center', ha='right', fontsize=7.6, color='#56606B')
    ax.scatter(
        medians,
        y,
        marker='D',
        s=54,
        color=COLORS['Fixed k=3'],
        edgecolor='white',
        linewidth=0.7,
        label='Sample median and 95% bootstrap interval',
        zorder=4
    )
    ax.scatter(
        resale,
        y,
        marker='|',
        s=150,
        linewidth=2.2,
        color=COLORS['Dynamic stopping'],
        label='Expected resale value of a duplicate',
        zorder=4
    )
    ax.set_xlim(0, xmax)
    ax.set_yticks(y, cfg.short_roles)
    ax.invert_yaxis()
    ax.set_xlabel('Market price or resale value (CNY)')
    ax.legend(ncol=2, loc='lower right')
    clean_axis(ax, 'x')
    ax.set_title('Market prices vary substantially across the eight designs', loc='left')
    fig.tight_layout()
    out.append(
        save_figure(
            fig,
            run.output_root,
            'main',
            'F01',
            'price_samples',
            'Data collection and analysis',
            'Show price observations, medians and uncertainty',
            run.png_dpi
        )
    )
    draws = sims.draws['Draw until complete']
    p95 = float(np.quantile(draws, 0.95))
    cutoff = 80
    (fig, ax) = plt.subplots(figsize=(8.0, 4.8))
    bins = np.arange(draws.min() - 0.5, cutoff + 1.5)
    kept = draws <= cutoff
    ax.hist(
        draws[kept],
        bins=bins,
        weights=np.full(np.sum(kept), 1 / len(draws)),
        color=COLORS['Draw until complete'],
        alpha=0.76,
        edgecolor='white',
        linewidth=0.35
    )
    ax.axvline(
        collect.expected_draws,
        color=COLORS['dark'],
        ls='--',
        lw=1.5,
        label=f'Theoretical mean {collect.expected_draws:.2f}  draws'
    )
    ax.axvline(p95, color=COLORS['orange'], ls=':', lw=2, label=f'P95 = {p95:.0f}  draws')
    ax.text(
        0.98,
        0.94,
        f'> {cutoff}  draws, tail probability: {np.mean(draws > cutoff):.2%}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=8.5,
        bbox={
            'boxstyle': 'round,pad=.3',
            'fc': 'white',
            'ec': '#D8DDE3'
        }
    )
    ax.set_xlabel('Draws required for the first complete set')
    ax.set_ylabel('Simulated probability')
    ax.legend(loc='upper center', ncol=2)
    ax.set_title('Collect-all stopping has a pronounced right tail', loc='left')
    clean_axis(ax)
    inset = inset_axes(ax, width='39%', height='42%', loc='center right', borderpad=1.25)
    xs = np.sort(draws)
    survival = 1 - np.arange(1, len(xs) + 1) / len(xs)
    inset.step(
        xs,
        np.maximum(survival, 1 / len(xs)),
        where='post',
        color=COLORS['Draw until complete'],
        lw=1.25
    )
    inset.set_yscale('log')
    inset.set_xlim(10, np.quantile(draws, 0.999))
    inset.set_ylim(0.001, 1)
    inset.set_title('Tail survival probability $P(T>t)$', fontsize=8)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.15)
    inset.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    out.append(
        save_figure(
            fig,
            run.output_root,
            'main',
            'F02',
            'collect_draws_tail',
            'Model solution and results',
            'Show the collect-all distribution and survival tail',
            run.png_dpi
        )
    )
    costs = sims.costs['Draw until complete']
    (fig, ax) = plt.subplots(figsize=(7.2, 4.5))
    hb = ax.hexbin(draws, costs, gridsize=30, mincnt=1, cmap='Reds')
    fig.colorbar(hb, ax=ax, label='Simulation count')
    ax.set_xlabel('Number of draws')
    ax.set_ylabel('Final net cost (CNY)')
    clean_axis(ax, None)
    fig.tight_layout()
    out.append(
        save_figure(
            fig,
            run.output_root,
            'appendix',
            'A01',
            'collect_draw_cost',
            'Appendix',
            'Inspect collect-all draw count and cost',
            run.png_dpi
        )
    )
    return out
