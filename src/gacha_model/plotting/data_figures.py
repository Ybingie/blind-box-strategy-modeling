import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..uncertainty import load_price_samples
from .style import COLORS, clean_axis, save_both


def make_data_figures(cfg, collect, sims, robustness, run):
    out = []

    samples = load_price_samples(cfg)
    rng = np.random.default_rng(cfg.base_seed + 101)
    y = np.arange(cfg.n)
    medians = np.asarray(cfg.second_hand_prices)
    resale = np.asarray(cfg.expected_resale_values)
    ci = np.asarray(robustness.median_ci)
    xmax = max(max(x) for x in samples) + 22
    fig, ax = plt.subplots(figsize=(8.7, 5.0))
    for i, values in enumerate(samples):
        jitter = rng.uniform(-.16, .16, size=len(values))
        ax.scatter(values, i + jitter, s=16, color="#9DB1C5", alpha=.48, linewidths=0, zorder=1)
        ax.plot(ci[i], [i, i], color=COLORS["固定 k=3"], lw=4, solid_capstyle="round", zorder=2)
        ax.text(xmax - 1, i, f"n={len(values)}", va="center", ha="right", fontsize=7.6, color="#56606B")
    ax.scatter(medians, y, marker="D", s=54, color=COLORS["固定 k=3"], edgecolor="white",
               linewidth=.7, label="样本中位数与 95% Bootstrap 区间", zorder=4)
    ax.scatter(resale, y, marker="|", s=150, linewidth=2.2, color=COLORS["动态停止"],
               label="重复款期望回收值", zorder=4)
    ax.set_xlim(0, xmax); ax.set_yticks(y, cfg.short_roles); ax.invert_yaxis()
    ax.set_xlabel("市场价格或回收价值（元）"); ax.legend(ncol=2, loc="lower right")
    clean_axis(ax, "x"); ax.set_title("125 条市场样本显示：不同款式价格差异明显", loc="left")
    fig.tight_layout()
    out.append(save_both(fig, run.output_root, "main", "F01", "price_samples", "数据收集与分析",
                         "展示原始价格样本、中位数及其不确定性", run.png_dpi))

    draws = sims.draws["一直抽到齐"]
    p95 = float(np.quantile(draws, .95)); cutoff = 80
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bins = np.arange(draws.min() - .5, cutoff + 1.5)
    kept = draws <= cutoff
    ax.hist(draws[kept], bins=bins, weights=np.full(np.sum(kept), 1 / len(draws)),
            color=COLORS["一直抽到齐"], alpha=.76, edgecolor="white", linewidth=.35)
    ax.axvline(collect.expected_draws, color=COLORS["dark"], ls="--", lw=1.5,
               label=f"理论均值 {collect.expected_draws:.2f} 次")
    ax.axvline(p95, color=COLORS["orange"], ls=":", lw=2, label=f"P95 = {p95:.0f} 次")
    ax.text(.98, .94, f"> {cutoff} 次的尾部概率：{np.mean(draws > cutoff):.2%}", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5,
            bbox={"boxstyle": "round,pad=.3", "fc": "white", "ec": "#D8DDE3"})
    ax.set_xlabel("首次集齐所需抽数"); ax.set_ylabel("模拟概率"); ax.legend(loc="upper center", ncol=2)
    ax.set_title("集齐停止具有明显右偏长尾", loc="left"); clean_axis(ax)
    inset = inset_axes(ax, width="39%", height="42%", loc="center right", borderpad=1.25)
    xs = np.sort(draws); survival = 1 - np.arange(1, len(xs) + 1) / len(xs)
    inset.step(xs, np.maximum(survival, 1 / len(xs)), where="post", color=COLORS["一直抽到齐"], lw=1.25)
    inset.set_yscale("log"); inset.set_xlim(10, np.quantile(draws, .999)); inset.set_ylim(1e-3, 1)
    inset.set_title("尾部生存概率 $P(T>t)$", fontsize=8); inset.tick_params(labelsize=7)
    inset.grid(alpha=.15); inset.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out.append(save_both(fig, run.output_root, "main", "F02", "collect_draws_tail", "模型求解与结果分析",
                         "展示集齐停止的主体分布与长尾概率", run.png_dpi))

    costs = sims.costs["一直抽到齐"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    hb = ax.hexbin(draws, costs, gridsize=30, mincnt=1, cmap="Reds")
    fig.colorbar(hb, ax=ax, label="模拟频数"); ax.set_xlabel("抽数"); ax.set_ylabel("最终净成本（元）")
    clean_axis(ax, None); fig.tight_layout()
    out.append(save_both(fig, run.output_root, "appendix", "A01", "collect_draw_cost", "附录", "诊断集齐抽数与成本关系", run.png_dpi))
    return out
