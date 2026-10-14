import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from .style import COLORS, clean_axis, save_figure

def _ecdf(x):
    v = np.sort(np.asarray(x))
    return (v, np.arange(1, len(v) + 1) / len(v))

def make_risk_figures(cfg, fixed, collect, dynamic, sims, summaries, run):
    out = []
    names = list(sims.costs)
    means = np.array([summaries[n].mean for n in names])
    p95 = np.array([summaries[n].p95 for n in names])
    y = np.arange(len(names))
    colors = [COLORS[n] for n in names]
    (fig, (ax, ax_far)) = plt.subplots(
        1,
        2,
        figsize=(9.0, 4.5),
        sharey=True,
        constrained_layout=True,
        gridspec_kw={'width_ratios': [2.5, 1.0], 'wspace': 0.06}
    )
    for target in (ax, ax_far):
        for i in range(len(names)):
            target.plot([means[i], p95[i]], [i, i], color=colors[i], lw=4, alpha=0.65)
        target.scatter(means, y, color=colors, s=60, zorder=3)
        target.scatter(p95, y, color=colors, marker='D', s=45, zorder=3)
        clean_axis(target, 'x')
    ax.set_xlim(340, 430)
    ax_far.set_xlim(520, max(960, p95.max() + 20))
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel('Mean and P95 (CNY; main strategies)')
    ax_far.set_xlabel('Mean and P95 (CNY; collect-all)')
    ax.scatter([], [], color='#56606B', s=60, label='Mean')
    ax.scatter([], [], color='#56606B', marker='D', s=45, label='P95')
    ax.legend(ncol=2, loc='lower right')
    for (i, n) in enumerate(names):
        target = ax_far if n == 'Draw until complete' else ax
        target.annotate(
            f'{means[i]:.1f}',
            (means[i], i),
            xytext=(-4, -13),
            textcoords='offset points',
            ha='center',
            fontsize=7.2,
            color=colors[i]
        )
        target.annotate(
            f'{p95[i]:.1f}',
            (p95[i], i),
            xytext=(4, 6),
            textcoords='offset points',
            fontsize=7.2,
            color=colors[i]
        )
        if n != 'Draw until complete':
            s = summaries[n]
            ax.text(
                428.5,
                i,
                f'>Budget {100 * s.exceed_budget:.1f}%\n[{100 * s.exceed_ci_low:.1f}, {100 * s.exceed_ci_high:.1f}]',
                ha='right',
                va='center',
                fontsize=6.7,
                color='#56606B'
            )
    ax.spines['right'].set_visible(False)
    ax_far.spines['left'].set_visible(False)
    ax_far.tick_params(axis='y', left=False, labelleft=False)
    d = 0.018
    kwargs = dict(transform=ax.transAxes, color='#56606B', clip_on=False, lw=1.1)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    kwargs.update(transform=ax_far.transAxes)
    ax_far.plot((-d, +d), (-d, +d), **kwargs)
    ax_far.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    out.append(
        save_figure(
            fig,
            run.output_root,
            'main',
            'F08',
            'strategy_mean_p95',
            'Monte Carlo risk analysis',
            'Compare mean and P95 across five strategies',
            run.png_dpi
        )
    )
    random_names = ['Fixed k=2', 'Fixed k=3', 'Dynamic stopping', 'Draw until complete']
    local_names = random_names[:3]
    (fig, (ax, ax_far)) = plt.subplots(
        1,
        2,
        figsize=(9.0, 4.8),
        constrained_layout=True,
        gridspec_kw={'width_ratios': [2.5, 1.0], 'wspace': 0.22}
    )
    rng = np.random.default_rng(cfg.base_seed + 111)
    for (target, group) in ((ax, local_names), (ax_far, ['Draw until complete'])):
        data = [sims.costs[n] for n in group]
        parts = target.violinplot(
            data,
            positions=np.arange(1, len(group) + 1),
            widths=0.72,
            showmeans=False,
            showmedians=False,
            showextrema=False
        )
        for (body, name) in zip(parts['bodies'], group):
            verts = body.get_paths()[0].vertices
            center = np.mean(verts[:, 0])
            verts[:, 0] = np.maximum(verts[:, 0], center)
            body.set_facecolor(COLORS[name])
            body.set_edgecolor(COLORS[name])
            body.set_alpha(0.55)
        target.boxplot(
            data,
            positions=np.arange(1, len(group) + 1) - 0.1,
            widths=0.13,
            showfliers=False,
            patch_artist=True,
            boxprops={
                'facecolor': 'white',
                'edgecolor': '#56606B'
            },
            whiskerprops={
                'color': '#56606B'
            },
            capprops={
                'color': '#56606B'
            },
            medianprops={
                'color': '#222222',
                'linewidth': 1.5
            }
        )
        for (pos, name, values) in zip(np.arange(1, len(group) + 1), group, data):
            take = rng.choice(len(values), size=min(450, len(values)), replace=False)
            jitter = rng.uniform(-0.34, -0.18, size=len(take))
            target.scatter(pos + jitter, values[take], s=7, alpha=0.18, color=COLORS[name], linewidths=0)
        target.axhline(
            cfg.budget,
            color=COLORS['orange'],
            ls='--',
            lw=1.2,
            label=f'Budget {cfg.budget:.0f} CNY'
        )
        target.set_xticks(range(1, len(group) + 1), group)
        clean_axis(target)
    ax.set_ylabel('Final net cost (CNY)')
    ax.set_title('Main randomized strategies', fontsize=11)
    ax_far.set_title('Collect-all strategy (separate scale)', fontsize=11)
    ax.legend(loc='upper left')
    out.append(
        save_figure(
            fig,
            run.output_root,
            'main',
            'F09',
            'random_strategy_raincloud',
            'Monte Carlo risk analysis',
            'Compare randomized cost distributions',
            run.png_dpi
        )
    )
    risks = np.array([summaries[n].exceed_budget for n in names]) * 100
    (fig, ax) = plt.subplots(figsize=(7.4, 4.7))
    for (i, n) in enumerate(names):
        ax.scatter(means[i], risks[i], s=95, color=COLORS[n], label=n)
        if n == 'Draw until complete':
            ax.annotate(n, (means[i], risks[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Mean net cost (CNY)')
    ax.set_ylabel(f'Probability above {cfg.budget:.0f}  CNY (%)')
    ax.legend(fontsize=8, ncol=2)
    clean_axis(ax)
    fig.tight_layout()
    zoom = inset_axes(ax, width='46%', height='43%', loc='center right', borderpad=1.1)
    for (i, n) in enumerate(names[:-1]):
        zoom.scatter(means[i], risks[i], s=45, color=COLORS[n])
    zoom.set_xlim(min(means[:-1]) - 2, max(means[:-1]) + 2)
    zoom.set_ylim(-0.8, max(risks[:-1]) + 1.8)
    zoom.set_title('Zoomed view of main strategies', fontsize=8)
    zoom.tick_params(labelsize=7)
    zoom.grid(True, color='#E5E9ED', lw=0.7)
    out.append(
        save_figure(
            fig,
            run.output_root,
            'appendix',
            'A10',
            'cost_risk',
            'Appendix',
            'Show the mean-cost and exceedance-risk tradeoff',
            run.png_dpi
        )
    )
    theory = np.array(
        [cfg.direct_cost, fixed.total_cost[2], fixed.total_cost[3], dynamic.exact_cost, collect.expected_cost]
    )
    residuals = means - theory
    standard_errors = np.array(
        [
            np.std(sims.costs[n], ddof=1) / np.sqrt(
                len(sims.costs[n])
            ) if len(sims.costs[n]) > 1 else 0.0 for n in names
        ]
    )
    ci95 = 1.96 * standard_errors
    (fig, ax) = plt.subplots(figsize=(7.7, 4.6))
    x = np.arange(len(names))
    ax.axhline(0, color='#7B8794', ls='--', lw=1.3, label='Theoretical value')
    for (i, n) in enumerate(names):
        ax.errorbar(
            x[i],
            residuals[i],
            yerr=ci95[i],
            fmt='o',
            ms=8,
            capsize=4,
            color=COLORS[n],
            ecolor=COLORS[n],
            elinewidth=1.5
        )
        offset = 7 if residuals[i] >= 0 else -14
        ax.annotate(
            f'{residuals[i]:+.3f}',
            (x[i], residuals[i]),
            xytext=(0, offset),
            textcoords='offset points',
            ha='center',
            fontsize=7.8,
            color=COLORS[n]
        )
    ax.set_xticks(x, names, rotation=12)
    ax.set_ylabel('Simulation mean $-$ theoretical expectation (CNY)')
    ax.text(
        0.99,
        0.03,
        'Error bars: 95% Monte Carlo CIs for simulation means',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=8,
        color='#56606B'
    )
    clean_axis(ax)
    fig.tight_layout()
    out.append(
        save_figure(
            fig,
            run.output_root,
            'appendix',
            'A11',
            'theory_mc',
            'Appendix',
            'Compare theoretical expectations and simulation means',
            run.png_dpi
        )
    )
    (fig, ax) = plt.subplots(figsize=(7.2, 4.4))
    for n in ['Fixed k=2', 'Fixed k=3']:
        (x, yv) = _ecdf(sims.costs[n])
        ax.plot(x, yv, color=COLORS[n], label=n)
    ax.set_xlabel('Final net cost (CNY)')
    ax.set_ylabel('Cumulative probability')
    ax.legend()
    clean_axis(ax)
    fig.tight_layout()
    out.append(
        save_figure(
            fig,
            run.output_root,
            'appendix',
            'A05',
            'fixed_ecdf',
            'Appendix',
            'Compare fixed two- and three-draw cost distributions',
            run.png_dpi
        )
    )
    x = sims.costs['Dynamic stopping']
    running = np.cumsum(x) / np.arange(1, len(x) + 1)
    (fig, ax) = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(np.arange(1, len(x) + 1), running, color=COLORS['Dynamic stopping'])
    ax.axhline(dynamic.exact_cost, color=COLORS['dark'], ls='--')
    ax.set_xscale('log')
    ax.set_xlabel('Simulation count (log scale)')
    ax.set_ylabel('Running mean (CNY)')
    clean_axis(ax)
    fig.tight_layout()
    out.append(
        save_figure(
            fig,
            run.output_root,
            'appendix',
            'A06',
            'mc_convergence',
            'Appendix',
            'Check convergence of the dynamic simulation mean',
            run.png_dpi
        )
    )
    budgets = np.linspace(330, 650, 65)
    (fig, ax) = plt.subplots(figsize=(7.5, 4.5))
    for n in ['Fixed k=2', 'Fixed k=3', 'Dynamic stopping', 'Draw until complete']:
        curve = [100 * np.mean(sims.costs[n] > b) for b in budgets]
        ax.plot(budgets, curve, color=COLORS[n], label=n)
    ax.set_xlabel('Budget threshold (CNY)')
    ax.set_ylabel('Probability over budget (%)')
    ax.legend(ncol=2)
    clean_axis(ax)
    fig.tight_layout()
    out.append(
        save_figure(
            fig,
            run.output_root,
            'appendix',
            'A07',
            'budget_exceedance',
            'Appendix',
            'Show exceedance across budget thresholds',
            run.png_dpi
        )
    )
    return out
