import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from .style import COLORS, clean_axis, save_both


def _draw_one_way_group(series, _title, code, slug, run):
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.9), constrained_layout=True)
    handles = None
    for panel, (ax, item) in enumerate(zip(axes, series)):
        fixed_line, = ax.plot(item.values, item.fixed_min_cost, color=COLORS["固定 k=2"],
                              marker="o", markevery=max(1, item.values.size // 6), label="优化固定策略")
        dynamic_line, = ax.plot(item.values, item.dynamic_cost, color=COLORS["动态停止"],
                                marker="s", markevery=max(1, item.values.size // 6), label="动态策略")
        ax.axvline(item.baseline, color="#D95F59", lw=1.1, ls=":")
        ax.scatter([item.baseline], [np.interp(item.baseline, item.values, item.dynamic_cost)],
                   marker="*", s=75, color="#D95F59", edgecolor="white", zorder=5)
        ax.set_xlabel(item.label)
        if panel == 0:
            ax.set_ylabel("最小期望净成本（元）")
        clean_axis(ax, "y")

        twin = ax.twinx()
        k_line, = twin.step(item.values, item.fixed_best_k, where="mid", color="#697782",
                            lw=1.25, ls="--", alpha=.82, label="固定最优 $k^*$")
        ticks = np.unique(item.fixed_best_k)
        twin.set_yticks(ticks)
        twin.set_ylim(max(-.4, ticks.min() - .8), ticks.max() + .8)
        twin.spines["top"].set_visible(False)
        twin.spines["right"].set_color("#A8AFB8")
        twin.tick_params(axis="y", colors="#697782", labelsize=8)
        if panel == 2:
            twin.set_ylabel("固定最优 $k^*$", color="#697782")
        ax.text(.02, .96, f"({chr(97 + panel)})", transform=ax.transAxes,
                ha="left", va="top", weight="bold")
        ax.set_title(item.label.split("（")[0], loc="left", fontsize=10)
        handles = ([fixed_line, dynamic_line, k_line], ["优化固定策略", "动态策略", "固定最优 $k^*$"])
    fig.legend(*handles, loc="upper center", bbox_to_anchor=(.5, -.015), ncol=3)
    return save_both(fig, run.output_root, "main", code, slug, "敏感性分析",
                     "比较单参数变化下固定与动态策略的期望成本及固定最优次数", run.png_dpi)


def make_sensitivity_figures(cfg, sensitivity, run):
    out = []
    by_key = {item.key: item for item in sensitivity.one_way}
    out.append(_draw_one_way_group(
        [by_key["c"], by_key["eta"], by_key["lambda"]],
        "核心决策参数的单因素敏感性", "F10", "core_parameter_sensitivity", run))
    out.append(_draw_one_way_group(
        [by_key["gamma"], by_key["d"], by_key["h"]],
        "交易摩擦参数的单因素敏感性", "F11", "friction_parameter_sensitivity", run))

    extent = [sensitivity.c_grid.min(), sensitivity.c_grid.max(), sensitivity.eta_grid.min(), sensitivity.eta_grid.max()]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    kvals = np.arange(np.min(sensitivity.fixed_best_k), np.max(sensitivity.fixed_best_k) + 1)
    kcmap = ListedColormap(plt.cm.YlGnBu(np.linspace(.28, .92, len(kvals))))
    knorm = BoundaryNorm(np.r_[kvals-.5, kvals[-1]+.5], kcmap.N)
    im1 = axes[0].imshow(sensitivity.fixed_best_k, origin="lower", aspect="auto", extent=extent,
                         cmap=kcmap, norm=knorm, interpolation="nearest")
    im2 = axes[1].imshow(sensitivity.dynamic_pair_continue, origin="lower", aspect="auto", extent=extent,
                         cmap="BuPu", interpolation="nearest")
    axes[0].contour(sensitivity.c_grid, sensitivity.eta_grid, sensitivity.fixed_best_k,
                    levels=np.arange(kvals.min()+.5, kvals.max()+.5, 1), colors="white", linewidths=.7, alpha=.85)
    levels = [5, 10, 15, 20, 25]
    cs = axes[1].contour(sensitivity.c_grid, sensitivity.eta_grid, sensitivity.dynamic_pair_continue,
                         levels=levels, colors="white", linewidths=.75, alpha=.8)
    axes[1].clabel(cs, fontsize=6.5, fmt="%d")
    for ax in axes:
        ax.scatter([cfg.blind_price], [cfg.resale_success_prob], marker="*", s=100, color="#E9827D", edgecolor="white")
        ax.annotate("基准参数", (cfg.blind_price, cfg.resale_success_prob), xytext=(7, 7),
                    textcoords="offset points", fontsize=7.5, color="#38404A")
        ax.set_xlabel("单次盲抽价格 c（元）"); ax.set_ylabel("转售成功率 η")
    axes[0].text(-.14, 1.04, "(a)", transform=axes[0].transAxes, weight="bold", fontsize=12)
    axes[1].text(-.14, 1.04, "(b)", transform=axes[1].transAxes, weight="bold", fontsize=12)
    axes[0].set_title("固定策略：最优 k 的分区", loc="left")
    axes[1].set_title("动态策略：继续状态数的变化", loc="left")
    fig.colorbar(im1, ax=axes[0], label="最优 k", ticks=kvals)
    fig.colorbar(im2, ax=axes[1], label="继续状态数（共 28 种）")
    out.append(save_both(fig, run.output_root, "main", "F12", "fixed_dynamic_sensitivity", "敏感性分析", "比较固定与动态策略对 c 和 eta 的敏感性", run.png_dpi))

    order = np.argsort(np.abs(sensitivity.tornado_change))
    fig, ax = plt.subplots(figsize=(7.5, 4.5)); vals = sensitivity.tornado_change[order]
    labels = np.asarray(sensitivity.tornado_labels)[order]
    ax.barh(labels, vals, color=[COLORS["一直抽到齐"] if v > 0 else COLORS["动态停止"] for v in vals])
    ax.axvline(0, color="#7B8794", lw=1); ax.set_xlabel("最低固定期望成本变化（元）"); clean_axis(ax, "x"); fig.tight_layout()
    out.append(save_both(fig, run.output_root, "appendix", "A08", "tornado", "附录", "比较主要参数的局部敏感性", run.png_dpi))
    return out


def make_uncertainty_figure(cfg, robustness, run):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), constrained_layout=True,
                             gridspec_kw={"width_ratios": [0.8, 1.65]})
    axes[0].bar(robustness.fixed_k_values.astype(str), robustness.fixed_k_probabilities,
                color=COLORS["动态停止"])
    axes[0].set_xlabel("Bootstrap 样本中的最优固定次数 k")
    axes[0].set_ylabel("出现比例")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("固定最优点的价格样本稳健性", loc="left", fontsize=10, weight="bold")
    for x, p in zip(robustness.fixed_k_values.astype(str), robustness.fixed_k_probabilities):
        axes[0].text(x, p+.025, f"{p:.1%}", ha="center", fontsize=8)
    clean_axis(axes[0], "y")

    records = sorted((r for r in robustness.pair_records if r["classification"] == "边界敏感"),
                     key=lambda r: r["continue_probability"])
    y = np.arange(len(records))
    probs = np.asarray([r["continue_probability"] for r in records])
    colors = [COLORS["动态停止"] if r["baseline_action"] == "继续" else COLORS["一直抽到齐"] for r in records]
    axes[1].axvspan(0.10, 0.90, color="#F7E7C6", alpha=0.55)
    axes[1].scatter(probs, y, c=colors, s=34, zorder=3)
    axes[1].axvline(0.5, color="#7B8794", lw=1, ls="--")
    axes[1].set_yticks(y, [r["pair"] for r in records], fontsize=7)
    axes[1].set_xlim(-0.03, 1.03)
    axes[1].set_xlabel("价格样本重抽下选择“继续”的比例")
    axes[1].set_ylabel("已获得款编号组合")
    axes[1].axvline(.10, color="#A7AEB6", lw=.8, ls=":"); axes[1].axvline(.90, color="#A7AEB6", lw=.8, ls=":")
    axes[1].set_title(f"仅显示 {len(records)} 个边界敏感状态", loc="left", fontsize=10, weight="bold")
    axes[1].text(.02, .02, "颜色表示基准决策；其余状态在重抽样中保持稳定",
                 transform=axes[1].transAxes, fontsize=7.6, color="#56606B")
    clean_axis(axes[1], "x")
    return [save_both(fig, run.output_root, "main", "F13", "price_bootstrap_robustness",
                      "敏感性分析", "用非参数 Bootstrap 检查固定最优点与动态边界", run.png_dpi)]
