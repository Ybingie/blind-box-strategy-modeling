import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from ..validation import is_report_baseline
from .style import COLORS, clean_axis, save_figure


def make_state_layers(cfg, dynamic, run):
    """Show all state layers when changed inputs no longer fit the baseline tree."""
    owned_counts = np.arange(cfg.n + 1)
    continuing = np.array([
        sum(r['owned_count'] == owned and r['action'] == 'Continue' for r in dynamic.state_records)
        for owned in owned_counts
    ])
    stopping = np.array([
        sum(r['owned_count'] == owned and r['action'] == 'Stop' for r in dynamic.state_records)
        for owned in owned_counts
    ])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(owned_counts, continuing, color=COLORS['Dynamic stopping'], label='Continue')
    ax.bar(owned_counts, stopping, bottom=continuing, color=COLORS['green'], label='Stop')
    ax.set_xlabel('Number of distinct acquired designs')
    ax.set_ylabel('Number of states')
    ax.set_xticks(owned_counts)
    ax.set_title('Dynamic actions by state layer under the current inputs', loc='left')
    ax.legend()
    clean_axis(ax)
    fig.tight_layout()
    return save_figure(fig, run.output_root, 'main', 'F06', 'dynamic_state_evolution',
                       'Price-aware dynamic stopping', 'Current-input dynamic state layers', run.png_dpi)

def make_model_figures(cfg, fixed, collect, dynamic, sims, robustness, run):
    out = []
    pair_continue = sum(r['action'] == 'Continue' for r in dynamic.pair_records)
    (fig, ax) = plt.subplots(figsize=(11.6, 3.45))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.15)
    ax.axis('off')
    ax.text(
        0.35,
        3.88,
        'From a common baseline to state-dependent decisions',
        fontsize=13,
        weight='bold',
        color=COLORS['dark'],
        va='top'
    )
    panels = [
        (
            0.45,
            '01  Boundary baselines',
            'Is blind drawing worthwhile?',
            'Secondary market vs. collect all',
            'No process information',
            COLORS['Secondary market only'],
            '#F2F4F6'
        ),
        (
            4.25,
            '02  Fixed stopping',
            'How many draws should be planned?',
            f'Optimal fixed count: $k^*={fixed.best_k}$',
            'Track only the draw count $k$',
            COLORS['Fixed k=3'],
            '#E9F2F8'
        ),
        (
            8.05,
            '03  Dynamic stopping',
            'Should the next draw be taken now?',
            '$G(U)>c$ : Continue',
            'Track the missing-item set $U$',
            COLORS['Dynamic stopping'],
            '#F0EAF6'
        )
    ]
    for (idx, (x, title, question, result, info, edge, fill)) in enumerate(panels):
        box = FancyBboxPatch(
            (x, 0.72),
            3.15,
            2.55,
            boxstyle='round,pad=.08,rounding_size=.10',
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.7
        )
        ax.add_patch(box)
        ax.text(x + 0.22, 2.98, title, fontsize=10.6, weight='bold', color=edge, va='top')
        ax.text(x + 0.22, 2.53, question, fontsize=9.0, color=COLORS['dark'], va='top')
        ax.text(x + 1.575, 1.82, result, fontsize=10.0, color=COLORS['dark'], ha='center', va='center')
        ax.plot([x + 0.35, x + 2.8], [1.3, 1.3], color=edge, lw=1.15, alpha=0.75)
        ax.text(x + 1.575, 1.02, info, fontsize=8.1, color='#56606B', ha='center', va='center')
        if idx == 1:
            xx = np.linspace(x + 0.65, x + 2.5, 30)
            yy = 1.47 + 0.16 * (xx - (x + 1.68)) ** 2
            ax.plot(xx, yy, color=edge, lw=1.6)
            ax.scatter([x + 1.68], [1.47], s=25, color=COLORS['Draw until complete'], zorder=3)
        elif idx == 2:
            ax.plot([x + 1.58, x + 1.58], [1.43, 1.7], color=edge, lw=1.4)
            ax.plot([x + 1.58, x + 1.15], [1.43, 1.31], color=edge, lw=1.4)
            ax.plot([x + 1.58, x + 2.01], [1.43, 1.31], color=edge, lw=1.4)
    for (x0, x1) in [(3.6, 4.18), (7.4, 7.98)]:
        ax.annotate(
            '',
            xy=(x1, 1.98),
            xytext=(x0, 1.98),
            arrowprops={
                'arrowstyle': '-|>',
                'lw': 1.6,
                'color': '#88939E'
            }
        )
    ax.text(
        7.7,
        3.48,
        'A fixed $k$ ignores item identity',
        ha='center',
        fontsize=8.0,
        color=COLORS['orange'],
        bbox={
            'boxstyle': 'round,pad=.25',
            'fc': '#FFF4EA',
            'ec': COLORS['orange']
        }
    )
    badge_x = 8.18
    badge_w = 3.42
    badge = FancyBboxPatch(
        (badge_x, 0.16),
        badge_w,
        0.4,
        boxstyle='round,pad=.06,rounding_size=.12',
        facecolor='#EAF5ED',
        edgecolor=COLORS['green'],
        linewidth=1.2
    )
    ax.add_patch(badge)
    ax.text(
        badge_x + badge_w / 2,
        0.36,
        'Bellman check: identical actions in all 256 states',
        ha='center',
        va='center',
        fontsize=7.5,
        color='#39774C',
        weight='bold'
    )
    ax.text(0.55, 0.32, 'State information', fontsize=8.3, color='#56606B', va='center')
    ax.annotate(
        '',
        xy=(7.95, 0.32),
        xytext=(1.72, 0.32),
        arrowprops={
            'arrowstyle': '->',
            'lw': 1.2,
            'color': '#A5ADB5'
        }
    )
    ax.text(2.05, 0.14, 'None', fontsize=7.6, color='#56606B')
    ax.text(5.25, 0.14, '$k$', fontsize=8.0, color='#56606B')
    ax.text(7.62, 0.14, '$U$', fontsize=8.0, color='#56606B')
    fig.tight_layout()
    out.append(
        save_figure(
            fig,
            run.output_root,
            'main',
            'F00',
            'model_progression',
            'Model development',
            'Summarize the progression from baselines to Bellman verification',
            run.png_dpi
        )
    )
    (fig, ax) = plt.subplots(figsize=(7.8, 4.7))
    ax.fill_between(
        fixed.k,
        robustness.fixed_cost_ci_low,
        robustness.fixed_cost_ci_high,
        color=COLORS['Fixed k=3'],
        alpha=0.16,
        label='95% price-bootstrap interval'
    )
    ax.plot(
        fixed.k,
        robustness.fixed_cost_median,
        color=COLORS['Fixed k=3'],
        lw=1.2,
        ls='--',
        label='Bootstrap median curve'
    )
    ax.plot(fixed.k, fixed.total_cost, marker='o', color=COLORS['dark'], lw=2, label='Baseline-price curve')
    ax.scatter(
        [2],
        [fixed.total_cost[2]],
        s=78,
        color=COLORS['Fixed k=2'],
        zorder=4,
        label=f'k=2: {fixed.total_cost[2]:.2f} CNY'
    )
    ax.scatter(
        [fixed.best_k],
        [fixed.total_cost[fixed.best_k]],
        s=90,
        color=COLORS['Draw until complete'],
        zorder=4,
        label=f'Optimal $k=${fixed.best_k}: {fixed.total_cost[fixed.best_k]:.2f} CNY'
    )
    ax.set_xlabel('Prespecified blind-draw count $k$')
    ax.set_ylabel('Expected cost (CNY)')
    ax.legend(loc='upper left', ncol=2)
    clean_axis(ax)
    ax.set_title(f'The fixed-stop strategy is minimized at $k={fixed.best_k}$', loc='left')
    zoom = inset_axes(ax, width='43%', height='48%', loc='center right', borderpad=1.15)
    local = np.arange(1, 6)
    zoom.plot(local, fixed.total_cost[local], marker='o', color=COLORS['Fixed k=3'], lw=1.5)
    zoom.scatter([2], [fixed.total_cost[2]], s=45, color=COLORS['Fixed k=2'], zorder=4)
    zoom.scatter([3], [fixed.total_cost[3]], s=50, color=COLORS['Draw until complete'], zorder=4)
    zoom.annotate(
        f'Difference: {fixed.total_cost[2] - fixed.total_cost[3]:.3f} CNY',
        xy=(2.5, (fixed.total_cost[2] + fixed.total_cost[3]) / 2),
        xytext=(3.35, 358.55),
        fontsize=7.5,
        ha='center',
        arrowprops={
            'arrowstyle': '->',
            'color': '#56606B',
            'lw': 0.8
        }
    )
    zoom.set_xticks(local)
    zoom.set_title('Zoomed optimum region', fontsize=8.5)
    zoom.tick_params(labelsize=7)
    zoom.grid(axis='y', alpha=0.2, linewidth=0.6)
    zoom.spines[['top', 'right']].set_visible(False)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.15, top=0.96)
    out.append(
        save_figure(
            fig,
            run.output_root,
            'main',
            'F03',
            'fixed_cost',
            'Fixed-stop strategy',
            'Locate the optimal fixed draw count',
            run.png_dpi
        )
    )
    ks = fixed.k[:-1]
    marginal_benefit = cfg.blind_price - fixed.marginal
    (fig, ax) = plt.subplots(figsize=(7.7, 4.5))
    ax.plot(
        ks,
        marginal_benefit,
        marker='o',
        color=COLORS['Fixed k=3'],
        label='Expected saving from one more draw'
    )
    ax.axhline(
        cfg.blind_price,
        color=COLORS['Draw until complete'],
        ls='--',
        lw=1.6,
        label=f'Price per draw c={cfg.blind_price:.1f} CNY'
    )
    ax.fill_between(
        ks,
        cfg.blind_price,
        marginal_benefit,
        where=marginal_benefit >= cfg.blind_price,
        color=COLORS['green'],
        alpha=0.16,
        label='Continuing lowers expected cost'
    )
    ax.fill_between(
        ks,
        marginal_benefit,
        cfg.blind_price,
        where=marginal_benefit < cfg.blind_price,
        color=COLORS['orange'],
        alpha=0.12,
        label='Continuing raises expected cost'
    )
    for k in [2, 3]:
        ax.scatter(
            k,
            marginal_benefit[k],
            s=76,
            color=COLORS['Dynamic stopping'] if k == 2 else COLORS['Draw until complete'],
            zorder=4
        )
        relation = '> c, Continue' if marginal_benefit[k] > cfg.blind_price else '< c, Stop'
        ax.annotate(
            f'k={k}: {marginal_benefit[k]:.2f} {relation}',
            (k, marginal_benefit[k]),
            xytext=(10, 10 if k == 2 else -17),
            textcoords='offset points',
            fontsize=8.3
        )
    ax.set_xlabel('Current fixed draw count $k$')
    ax.set_ylabel('Expected saving from one more draw (CNY)')
    ax.set_xlim(0, cfg.fixed_k_max - 1)
    ax.legend(ncol=2, loc='upper right')
    clean_axis(ax)
    ax.set_title('The marginal saving crosses the draw price at the optimum', loc='left')
    fig.tight_layout()
    out.append(
        save_figure(
            fig,
            run.output_root,
            'main',
            'F04',
            'fixed_marginal_mechanism',
            'Fixed-stop strategy',
            'Explain the fixed optimum using marginal savings',
            run.png_dpi
        )
    )
    margin = np.full((cfg.n, cfg.n), np.nan)
    for rec in dynamic.pair_records:
        (i, j) = sorted((rec['i'], rec['j']))
        margin[i, j] = rec['G'] - cfg.blind_price
    limit = float(np.nanmax(np.abs(margin)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    decision_cmap = LinearSegmentedColormap.from_list(
        'stop_continue',
        [
            COLORS['orange'],
            '#F7F7F7',
            COLORS['Dynamic stopping']
        ]
    )
    (fig, ax) = plt.subplots(figsize=(7.7, 6.2))
    im = ax.imshow(np.ma.masked_invalid(margin), cmap=decision_cmap, norm=norm)
    stability = {r['pair']: r for r in robustness.pair_records}
    for rec in dynamic.pair_records:
        (i, j) = sorted((rec['i'], rec['j']))
        value = rec['G'] - cfg.blind_price
        action = 'Continue' if value > 0 else 'Stop'
        ax.text(
            j,
            i,
            f'{action}\n{value:+.2f}',
            ha='center',
            va='center',
            fontsize=7.2,
            color='white' if abs(value) > 0.62 * limit else '#222222',
            weight='bold' if abs(value) < 0.25 else 'normal'
        )
        record = stability[rec['pair']]
        edge = '#E59B36' if record['classification'] == 'Boundary-sensitive' else '#3B4650'
        style = '--' if record['classification'] == 'Boundary-sensitive' else '-'
        ax.add_patch(
            Rectangle(
                (j - 0.48, i - 0.48),
                0.96,
                0.96,
                fill=False,
                edgecolor=edge,
                linewidth=1.55 if record['classification'] == 'Boundary-sensitive' else 0.45,
                linestyle=style
            )
        )
    ax.set_xticks(range(cfg.n), cfg.short_roles, rotation=35, ha='right')
    ax.set_yticks(range(cfg.n), cfg.short_roles)
    ax.set_xlim(-0.5, cfg.n - 0.5)
    ax.set_ylim(cfg.n - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, cfg.n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, cfg.n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.text(
        0.02,
        0.02,
        f'Continue {pair_continue} states | Stop {28-pair_continue} states\nCell value: G(U)-c (CNY)',
        transform=ax.transAxes,
        fontsize=8.5,
        va='bottom',
        ha='left',
        bbox={
            'boxstyle': 'round,pad=.35',
            'facecolor': 'white',
            'edgecolor': '#CBD1D8',
            'alpha': 0.94
        }
    )
    ax.legend(
        handles=[
            Patch(
                facecolor='none',
                edgecolor='#E59B36',
                linestyle='--',
                label='Bootstrap Boundary-sensitive'
            ),
            Patch(
                facecolor='none',
                edgecolor='#3B4650',
                label='Bootstrap Stable'
            )
        ],
        loc='lower right',
        fontsize=7.5
    )
    fig.colorbar(im, ax=ax, label='Decision margin $G(U)-c$ (CNY)')
    fig.tight_layout()
    out.append(
        save_figure(
            fig,
            run.output_root,
            'main',
            'F05',
            'dynamic_pair_heatmap',
            'Price-aware dynamic stopping',
            'Classify all 28 two-design states',
            run.png_dpi
        )
    )
    if is_report_baseline(cfg):
        (fig, (ax, calc)) = plt.subplots(
            1,
            2,
            figsize=(9.6, 5.15),
            gridspec_kw={'width_ratios': [1.55, 1.0], 'wspace': 0.04}
        )
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 10)
        ax.axis('off')
        calc.set_xlim(0, 1)
        calc.set_ylim(0, 1)
        calc.axis('off')
        ax.text(
            0.15,
            9.72,
            'State tree: decisions branch after the second distinct design',
            fontsize=12,
            weight='bold',
            color=COLORS['dark'],
            va='top'
        )
        nodes = {
            'start': (3.0, 8.55, 'Start', '0  designs', COLORS['Fixed k=3']),
            'one': (3.0, 6.65, 'First distinct design', 'Always continue', COLORS['Fixed k=3']),
            'two': (
                3.0,
                4.75,
                'Second distinct design',
                'Compare $G(U)$ and $c$',
                COLORS['Dynamic stopping']
            ),
            'stop2': (1.35, 2.2, 'Stop and complete', '18 / 28  states', COLORS['green']),
            'cont': (4.65, 2.85, 'Continue drawing', '10 / 28  states', COLORS['Dynamic stopping']),
            'stop3': (4.65, 0.8, 'Obtain a third design', 'Stop in all states', COLORS['green'])
        }
        for (key, (x, y0, title, subtitle, edge)) in nodes.items():
            width = 2.0 if key in {'two', 'stop2', 'cont'} else 1.75
            height = 1.02
            fill = '#F1ECF7' if key in {'two', 'cont'} else '#EAF5ED' if key in {'stop2', 'stop3'} else '#EDF4F9'
            box = FancyBboxPatch(
                (x - width / 2, y0 - height / 2),
                width,
                height,
                boxstyle='round,pad=.06,rounding_size=.10',
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.7
            )
            ax.add_patch(box)
            ax.text(
                x,
                y0 + 0.15,
                title,
                ha='center',
                va='center',
                fontsize=9.0,
                weight='bold',
                color=COLORS['dark']
            )
            ax.text(x, y0 - 0.2, subtitle, ha='center', va='center', fontsize=7.8, color='#56606B')
        ax.annotate(
            '',
            xy=(3, 7.18),
            xytext=(3, 8.02),
            arrowprops={
                'arrowstyle': '-|>',
                'lw': 1.5,
                'color': '#88939E'
            }
        )
        ax.text(3.15, 7.6, 'First draw', fontsize=7.5, color='#56606B', va='center')
        ax.annotate(
            '',
            xy=(3, 5.28),
            xytext=(3, 6.12),
            arrowprops={
                'arrowstyle': '-|>',
                'lw': 1.5,
                'color': '#88939E'
            }
        )
        ax.text(3.15, 5.7, 'Wait for the second design', fontsize=7.5, color='#56606B', va='center')
        ax.annotate(
            '',
            xy=(1.7, 2.73),
            xytext=(2.52, 4.22),
            arrowprops={'arrowstyle': '-|>', 'lw': 1.55, 'color': COLORS['green']}
        )
        ax.text(1.62, 3.55, '$G(U)\\leq c$', fontsize=8.2, color='#39774C', ha='center')
        ax.annotate(
            '',
            xy=(4.3, 3.38),
            xytext=(3.48, 4.22),
            arrowprops={'arrowstyle': '-|>', 'lw': 1.55, 'color': COLORS['Dynamic stopping']}
        )
        ax.text(4.34, 4.02, '$G(U)>c$', fontsize=8.2, color=COLORS['Dynamic stopping'], ha='center')
        ax.annotate(
            '',
            xy=(4.65, 1.33),
            xytext=(4.65, 2.32),
            arrowprops={
                'arrowstyle': '-|>',
                'lw': 1.5,
                'color': '#88939E'
            }
        )
        ax.text(4.82, 1.8, 'Wait for the third design', fontsize=7.5, color='#56606B', va='center')
        card = FancyBboxPatch(
            (0.06, 0.08),
            0.88,
            0.84,
            boxstyle='round,pad=.025,rounding_size=.03',
            facecolor='#F7F8FA',
            edgecolor='#CBD2D9',
            linewidth=1.2
        )
        calc.add_patch(card)
        calc.text(
            0.12,
            0.86,
            'Analytical decomposition of the mean draw count',
            fontsize=11,
            weight='bold',
            color=COLORS['dark'],
            va='top'
        )
        items = [
            ('①', 'Obtain the first design', '$1$', COLORS['Fixed k=3']),
            ('②', 'Wait from one to two designs', '$8/7$', COLORS['Fixed k=3']),
            (
                '③',
                'Wait for a third design only in Continue states',
                '$(10/28)(8/6)$',
                COLORS['Dynamic stopping']
            )
        ]
        for (i, (num, label, value, color)) in enumerate(items):
            y0 = 0.69 - i * 0.17
            calc.text(0.13, y0, num, fontsize=10, weight='bold', color=color, va='center')
            calc.text(0.21, y0 + 0.025, label, fontsize=8.0, color='#56606B', va='center')
            calc.text(0.83, y0 - 0.025, value, fontsize=9.5, color=color, ha='right', va='center')
        calc.plot([0.13, 0.86], [0.27, 0.27], color='#CBD2D9', lw=1)
        calc.text(
            0.13,
            0.2,
            '$E(\\tau)=1+\\dfrac{8}{7}+\\dfrac{10}{28}\\dfrac{8}{6}$',
            fontsize=10.2,
            color=COLORS['dark']
        )
        calc.text(
            0.13,
            0.125,
            f'$={dynamic.exact_draws:.3f}$  draws',
            fontsize=11,
            color=COLORS['Dynamic stopping'],
            weight='bold'
        )
        calc.text(
            0.5,
            0.025,
            f'Exact expected cost  {dynamic.exact_cost:.2f} CNY',
            fontsize=8.6,
            ha='center',
            color='#56606B',
            va='bottom'
        )
        fig.subplots_adjust(left=0.02, right=0.99, bottom=0.04, top=0.97, wspace=0.03)
        out.append(
            save_figure(
                fig,
                run.output_root,
                'main',
                'F06',
                'dynamic_state_evolution',
                'Price-aware dynamic stopping',
                'Explain dynamic state transitions and stopping',
                run.png_dpi
            )
        )
    else:
        out.append(make_state_layers(cfg, dynamic, run))
    scenario_names = [
        'Complete BOX (without replacement)',
        'Dynamic stopping',
        'Fixed k=3',
        'Fixed k=2',
        'Secondary market only',
        'Draw until complete'
    ]
    scenario_values = [
        cfg.n * cfg.blind_price,
        dynamic.exact_cost,
        fixed.total_cost[3],
        fixed.total_cost[2],
        cfg.direct_cost,
        collect.expected_cost
    ]
    scenario_colors = [
        COLORS['green'],
        COLORS['Dynamic stopping'],
        COLORS['Fixed k=3'],
        COLORS['Fixed k=2'],
        COLORS['Secondary market only'],
        COLORS['Draw until complete']
    ]
    order = np.argsort(scenario_values)[::-1]
    (fig, ax) = plt.subplots(figsize=(8.2, 5.0))
    yy = np.arange(len(order))
    for (pos, idx) in enumerate(order):
        ax.plot([300, scenario_values[idx]], [pos, pos], color='#D6DBE1', lw=2.5)
        ax.scatter(scenario_values[idx], pos, s=86, color=scenario_colors[idx], zorder=3)
        ax.text(scenario_values[idx] + 4, pos, f'{scenario_values[idx]:.2f}', va='center', fontsize=8.4)
    labels = [scenario_names[i] for i in order]
    ax.set_yticks(yy, labels)
    ax.invert_yaxis()
    ax.set_xlim(300, max(scenario_values) + 45)
    ax.axvspan(300, 330, color=COLORS['green'], alpha=0.08)
    ax.set_xlabel('Expected net cost (CNY)')
    clean_axis(ax, 'x')
    ax.set_title('A complete box and independent blind draws use different sampling mechanisms', loc='left')
    ax.text(
        0.99,
        0.04,
        'BOX is a scenario bound; the main model uses independent draws with replacement',
        transform=ax.transAxes,
        ha='right',
        fontsize=8.2,
        color='#56606B'
    )
    fig.tight_layout()
    out.append(
        save_figure(
            fig,
            run.output_root,
            'main',
            'F14',
            'scenario_comparison',
            'Model boundaries',
            'Compare loose-pack strategies and the complete-box scenario',
            run.png_dpi
        )
    )
    dd = sims.draws['Dynamic stopping']
    (values, counts) = np.unique(dd, return_counts=True)
    (fig, ax) = plt.subplots(figsize=(7.2, 4.3))
    ax.bar(values, counts / counts.sum(), color=COLORS['Dynamic stopping'], alpha=0.8)
    ax.axvline(
        dynamic.exact_draws,
        color=COLORS['dark'],
        ls='--',
        label=f'Analytical mean {dynamic.exact_draws:.3f}'
    )
    ax.set_xlabel('Draws under dynamic stopping')
    ax.set_ylabel('Probability')
    ax.legend()
    clean_axis(ax)
    fig.tight_layout()
    out.append(
        save_figure(
            fig,
            run.output_root,
            'appendix',
            'A09',
            'dynamic_draws',
            'Appendix',
            'Show the dynamic stopping-time distribution',
            run.png_dpi
        )
    )
    (fig, ax) = plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(0, color='#A8AFB8', lw=1)
    ax.stem(fixed.k[:-1], fixed.marginal, linefmt=COLORS['Fixed k=3'], markerfmt='o', basefmt=' ')
    ax.set_xlabel('Increase from $k$ to $k+1$')
    ax.set_ylabel('Marginal expected cost (CNY)')
    clean_axis(ax)
    fig.tight_layout()
    out.append(
        save_figure(
            fig,
            run.output_root,
            'appendix',
            'A02',
            'fixed_marginal',
            'Appendix',
            'Check adjacent differences at the fixed optimum',
            run.png_dpi
        )
    )
    (fig, ax) = plt.subplots(figsize=(7.4, 4.3))
    ax.stackplot(
        fixed.k,
        fixed.draw_cost,
        fixed.missing_cost,
        -fixed.resale_income,
        labels=['Blind-draw spending', 'Completion spending', 'Resale offset'],
        colors=[COLORS['Fixed k=3'], COLORS['Draw until complete'], COLORS['Dynamic stopping']],
        alpha=0.75
    )
    ax.plot(fixed.k, fixed.total_cost, color=COLORS['dark'], lw=2, label='Net cost')
    ax.set_xlabel('k')
    ax.set_ylabel('Amount (CNY)')
    ax.legend(ncol=2)
    clean_axis(ax)
    fig.tight_layout()
    out.append(
        save_figure(
            fig,
            run.output_root,
            'appendix',
            'A03',
            'fixed_components',
            'Appendix',
            'Decompose fixed-strategy costs',
            run.png_dpi
        )
    )
    (fig, ax) = plt.subplots(figsize=(7.2, 4.4))
    hb = ax.hexbin(dd, sims.costs['Dynamic stopping'], gridsize=25, mincnt=1, cmap='Purples')
    fig.colorbar(hb, ax=ax, label='Simulation count')
    ax.set_xlabel('Draws under dynamic stopping')
    ax.set_ylabel('Final net cost (CNY)')
    clean_axis(ax, None)
    fig.tight_layout()
    out.append(
        save_figure(
            fig,
            run.output_root,
            'appendix',
            'A04',
            'dynamic_draw_cost',
            'Appendix',
            'Inspect dynamic draw count and cost',
            run.png_dpi
        )
    )
    return out
