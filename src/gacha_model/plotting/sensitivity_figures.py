import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from .style import COLORS, clean_axis, save_figure

def _draw_one_way_group(series, _title, code, slug, run):
    (fig, axes) = plt.subplots(1, 3, figsize=(11.6, 3.9), constrained_layout=True)
    handles = None
    for (panel, (ax, item)) in enumerate(zip(axes, series)):
        (fixed_line,) = ax.plot(
            item.values,
            item.fixed_min_cost,
            color=COLORS['Fixed k=2'],
            marker='o',
            markevery=max(1, item.values.size // 6),
            label='Optimized fixed-stop'
        )
        (dynamic_line,) = ax.plot(
            item.values,
            item.dynamic_cost,
            color=COLORS['Dynamic stopping'],
            marker='s',
            markevery=max(1, item.values.size // 6),
            label='Dynamic strategy'
        )
        ax.axvline(item.baseline, color='#D95F59', lw=1.1, ls=':')
        ax.scatter(
            [item.baseline],
            [np.interp(item.baseline, item.values, item.dynamic_cost)],
            marker='*',
            s=75,
            color='#D95F59',
            edgecolor='white',
            zorder=5
        )
        ax.set_xlabel(item.label)
        if panel == 0:
            ax.set_ylabel('Minimum expected net cost (CNY)')
        clean_axis(ax, 'y')
        twin = ax.twinx()
        (k_line,) = twin.step(
            item.values,
            item.fixed_best_k,
            where='mid',
            color='#697782',
            lw=1.25,
            ls='--',
            alpha=0.82,
            label='Optimal fixed $k^*$'
        )
        ticks = np.unique(item.fixed_best_k)
        twin.set_yticks(ticks)
        twin.set_ylim(max(-0.4, ticks.min() - 0.8), ticks.max() + 0.8)
        twin.spines['top'].set_visible(False)
        twin.spines['right'].set_color('#A8AFB8')
        twin.tick_params(axis='y', colors='#697782', labelsize=8)
        if panel == 2:
            twin.set_ylabel('Optimal fixed $k^*$', color='#697782')
        ax.text(
            0.02,
            0.96,
            f'({chr(97 + panel)})',
            transform=ax.transAxes,
            ha='left',
            va='top',
            weight='bold'
        )
        ax.set_title(item.label.split(' (')[0], loc='left', fontsize=10)
        handles = (
            [fixed_line, dynamic_line, k_line],
            ['Optimized fixed-stop', 'Dynamic strategy', 'Optimal fixed $k^*$']
        )
    fig.legend(*handles, loc='upper center', bbox_to_anchor=(0.5, -0.015), ncol=3)
    return save_figure(
        fig,
        run.output_root,
        'main',
        code,
        slug,
        'Sensitivity analysis',
        'Compare optimized costs and fixed counts under one-way perturbations',
        run.png_dpi
    )

def make_sensitivity_figures(cfg, sensitivity, run):
    out = []
    by_key = {item.key: item for item in sensitivity.one_way}
    out.append(
        _draw_one_way_group(
            [by_key['c'], by_key['eta'], by_key['lambda']],
            'One-way sensitivity to core decision parameters',
            'F10',
            'core_parameter_sensitivity',
            run
        )
    )
    out.append(
        _draw_one_way_group(
            [by_key['gamma'], by_key['d'], by_key['h']],
            'One-way sensitivity to transaction frictions',
            'F11',
            'friction_parameter_sensitivity',
            run
        )
    )
    extent = [
        sensitivity.c_grid.min(),
        sensitivity.c_grid.max(),
        sensitivity.eta_grid.min(),
        sensitivity.eta_grid.max()
    ]
    (fig, axes) = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    kvals = np.arange(np.min(sensitivity.fixed_best_k), np.max(sensitivity.fixed_best_k) + 1)
    kcmap = ListedColormap(plt.cm.YlGnBu(np.linspace(0.28, 0.92, len(kvals))))
    knorm = BoundaryNorm(np.r_[kvals - 0.5, kvals[-1] + 0.5], kcmap.N)
    im1 = axes[0].imshow(
        sensitivity.fixed_best_k,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap=kcmap,
        norm=knorm,
        interpolation='nearest'
    )
    im2 = axes[1].imshow(
        sensitivity.dynamic_pair_continue,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap='BuPu',
        interpolation='nearest'
    )
    axes[0].contour(
        sensitivity.c_grid,
        sensitivity.eta_grid,
        sensitivity.fixed_best_k,
        levels=np.arange(kvals.min() + 0.5, kvals.max() + 0.5, 1),
        colors='white',
        linewidths=0.7,
        alpha=0.85
    )
    levels = [5, 10, 15, 20, 25]
    cs = axes[1].contour(
        sensitivity.c_grid,
        sensitivity.eta_grid,
        sensitivity.dynamic_pair_continue,
        levels=levels,
        colors='white',
        linewidths=0.75,
        alpha=0.8
    )
    axes[1].clabel(cs, fontsize=6.5, fmt='%d')
    for ax in axes:
        ax.scatter(
            [cfg.blind_price],
            [cfg.resale_success_prob],
            marker='*',
            s=100,
            color='#E9827D',
            edgecolor='white'
        )
        ax.annotate(
            'Baseline',
            (cfg.blind_price, cfg.resale_success_prob),
            xytext=(7, 7),
            textcoords='offset points',
            fontsize=7.5,
            color='#38404A'
        )
        ax.set_xlabel('Blind-draw price $c$ (CNY)')
        ax.set_ylabel('Resale success probability $\\eta$')
    axes[0].text(-0.14, 1.04, '(a)', transform=axes[0].transAxes, weight='bold', fontsize=12)
    axes[1].text(-0.14, 1.04, '(b)', transform=axes[1].transAxes, weight='bold', fontsize=12)
    axes[0].set_title('Fixed strategy: regions of optimal $k$', loc='left')
    axes[1].set_title('Dynamic strategy: number of Continue states', loc='left')
    fig.colorbar(im1, ax=axes[0], label='Optimal $k$', ticks=kvals)
    fig.colorbar(im2, ax=axes[1], label='Continue states (out of 28)')
    out.append(
        save_figure(
            fig,
            run.output_root,
            'main',
            'F12',
            'fixed_dynamic_sensitivity',
            'Sensitivity analysis',
            'Compare joint sensitivity to c and eta',
            run.png_dpi
        )
    )
    order = np.argsort(np.abs(sensitivity.tornado_change))
    (fig, ax) = plt.subplots(figsize=(7.5, 4.5))
    vals = sensitivity.tornado_change[order]
    labels = np.asarray(sensitivity.tornado_labels)[order]
    ax.barh(
        labels,
        vals,
        color=[COLORS['Draw until complete'] if v > 0 else COLORS['Dynamic stopping'] for v in vals]
    )
    ax.axvline(0, color='#7B8794', lw=1)
    ax.set_xlabel('Change in minimum fixed-strategy cost (CNY)')
    clean_axis(ax, 'x')
    fig.tight_layout()
    out.append(
        save_figure(
            fig,
            run.output_root,
            'appendix',
            'A08',
            'tornado',
            'Appendix',
            'Compare local parameter sensitivity',
            run.png_dpi
        )
    )
    return out

def make_uncertainty_figure(cfg, robustness, run):
    (fig, axes) = plt.subplots(
        1,
        2,
        figsize=(11.2, 5.0),
        constrained_layout=True,
        gridspec_kw={'width_ratios': [0.8, 1.65]}
    )
    axes[0].bar(
        robustness.fixed_k_values.astype(str),
        robustness.fixed_k_probabilities,
        color=COLORS['Dynamic stopping']
    )
    axes[0].set_xlabel('Optimal fixed draw count $k$ in bootstrap samples')
    axes[0].set_ylabel('Proportion')
    axes[0].set_ylim(0, 1)
    axes[0].set_title('Price-sample robustness of the fixed optimum', loc='left', fontsize=10, weight='bold')
    for (x, p) in zip(robustness.fixed_k_values.astype(str), robustness.fixed_k_probabilities):
        axes[0].text(x, p + 0.025, f'{p:.1%}', ha='center', fontsize=8)
    clean_axis(axes[0], 'y')
    records = sorted(
        (r for r in robustness.pair_records if r['classification'] == 'Boundary-sensitive'),
        key=lambda r: r['continue_probability']
    )
    y = np.arange(len(records))
    probs = np.asarray([r['continue_probability'] for r in records])
    colors = [
        COLORS['Dynamic stopping'] if r['baseline_action'] == 'Continue' else COLORS['Draw until complete'] for r in records
    ]
    axes[1].axvspan(0.1, 0.9, color='#F7E7C6', alpha=0.55)
    axes[1].scatter(probs, y, c=colors, s=34, zorder=3)
    axes[1].axvline(0.5, color='#7B8794', lw=1, ls='--')
    axes[1].set_yticks(y, [r['pair'] for r in records], fontsize=7)
    axes[1].set_xlim(-0.03, 1.03)
    axes[1].set_xlabel('Proportion choosing Continue across price resamples')
    axes[1].set_ylabel('Pair of acquired item indices')
    axes[1].axvline(0.1, color='#A7AEB6', lw=0.8, ls=':')
    axes[1].axvline(0.9, color='#A7AEB6', lw=0.8, ls=':')
    axes[1].set_title(
        f'Showing only {len(records)}  boundary-sensitive states',
        loc='left',
        fontsize=10,
        weight='bold'
    )
    axes[1].text(
        0.02,
        0.02,
        'Colors show baseline actions; all other states remain stable',
        transform=axes[1].transAxes,
        fontsize=7.6,
        color='#56606B'
    )
    clean_axis(axes[1], 'x')
    return [
        save_figure(
            fig,
            run.output_root,
            'main',
            'F13',
            'price_bootstrap_robustness',
            'Sensitivity analysis',
            'Assess fixed and dynamic robustness using price bootstrap',
            run.png_dpi
        )
    ]
