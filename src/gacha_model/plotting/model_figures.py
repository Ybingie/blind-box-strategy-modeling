import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from .style import COLORS, clean_axis, save_both
from .i18n import get_language


def make_model_figures(cfg, fixed, collect, dynamic, sims, robustness, run):
    out = []

    # 模型递进路线图：三层主模型承担论证主线，Bellman 作为验证而非并列模型。
    fig, ax = plt.subplots(figsize=(11.6 if get_language() == "en" else 10.8, 3.45))
    ax.set_xlim(0, 12); ax.set_ylim(0, 4.15); ax.axis("off")
    ax.text(.35, 3.88, "从统一规则到状态决策：每一步只增加必要信息", fontsize=13,
            weight="bold", color=COLORS["dark"], va="top")
    panels = [
        (.45, "01  边界基准", "先回答：盲抽是否值得？", "完全二手  vs.  集齐停止", "不使用过程信息", COLORS["完全二手"], "#F2F4F6"),
        (4.25, "02  固定停止", "再回答：预先抽几次？", "$k^*=3$，但 $k=2$ 几乎等价", "只记录抽取次数 $k$", COLORS["固定 k=3"], "#E9F2F8"),
        (8.05, "03  动态停止", "最后回答：此刻是否继续？", "$G(U)>c$ 时继续", "记录缺失集合 $U$", COLORS["动态停止"], "#F0EAF6"),
    ]
    for idx, (x, title, question, result, info, edge, fill) in enumerate(panels):
        box = FancyBboxPatch((x, .72), 3.15, 2.55, boxstyle="round,pad=.08,rounding_size=.10",
                             facecolor=fill, edgecolor=edge, linewidth=1.7)
        ax.add_patch(box)
        ax.text(x+.22, 2.98, title, fontsize=10.6, weight="bold", color=edge, va="top")
        ax.text(x+.22, 2.53, question, fontsize=9.0, color=COLORS["dark"], va="top")
        ax.text(x+1.575, 1.82, result, fontsize=10.0, color=COLORS["dark"], ha="center", va="center")
        ax.plot([x+.35, x+2.80], [1.30, 1.30], color=edge, lw=1.15, alpha=.75)
        ax.text(x+1.575, 1.02, info, fontsize=8.1, color="#56606B", ha="center", va="center")
        if idx == 1:
            xx = np.linspace(x+.65, x+2.50, 30)
            yy = 1.47 + .16*(xx-(x+1.68))**2
            ax.plot(xx, yy, color=edge, lw=1.6)
            ax.scatter([x+1.68], [1.47], s=25, color=COLORS["一直抽到齐"], zorder=3)
        elif idx == 2:
            ax.plot([x+1.58, x+1.58], [1.43, 1.70], color=edge, lw=1.4)
            ax.plot([x+1.58, x+1.15], [1.43, 1.31], color=edge, lw=1.4)
            ax.plot([x+1.58, x+2.01], [1.43, 1.31], color=edge, lw=1.4)
    for x0, x1 in [(3.60, 4.18), (7.40, 7.98)]:
        ax.annotate("", xy=(x1, 1.98), xytext=(x0, 1.98),
                    arrowprops={"arrowstyle": "-|>", "lw": 1.6, "color": "#88939E"})
    ax.text(7.70, 3.48, "固定 k 丢失款式信息", ha="center", fontsize=8.0, color=COLORS["orange"],
            bbox={"boxstyle": "round,pad=.25", "fc": "#FFF4EA", "ec": COLORS["orange"]})
    badge_x = 8.18 if get_language() == "en" else 8.65
    badge_w = 3.42 if get_language() == "en" else 2.55
    badge = FancyBboxPatch((badge_x, .16), badge_w, .40, boxstyle="round,pad=.06,rounding_size=.12",
                           facecolor="#EAF5ED", edgecolor=COLORS["green"], linewidth=1.2)
    ax.add_patch(badge)
    ax.text(badge_x + badge_w / 2, .36, "Bellman：256 个状态动作完全一致", ha="center", va="center",
            fontsize=7.5 if get_language() == "en" else 8.0, color="#39774C", weight="bold")
    ax.text(.55, .32, "信息量", fontsize=8.3, color="#56606B", va="center")
    ax.annotate("", xy=(7.95, .32), xytext=(1.72 if get_language() == "en" else 1.25, .32),
                arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#A5ADB5"})
    ax.text(2.05, .14, "无", fontsize=7.6, color="#56606B"); ax.text(5.25, .14, "$k$", fontsize=8.0, color="#56606B")
    ax.text(7.62, .14, "$U$", fontsize=8.0, color="#56606B")
    fig.tight_layout()
    out.append(save_both(fig, run.output_root, "main", "F00", "model_progression", "数学模型建立",
                         "概括由边界基准到 Bellman 验证的递进路线", run.png_dpi))

    fig, ax = plt.subplots(figsize=(7.8, 4.7))
    ax.fill_between(fixed.k, robustness.fixed_cost_ci_low, robustness.fixed_cost_ci_high,
                    color=COLORS["固定 k=3"], alpha=.16, label="价格 Bootstrap 95% 区间")
    ax.plot(fixed.k, robustness.fixed_cost_median, color=COLORS["固定 k=3"], lw=1.2, ls="--",
            label="Bootstrap 中位曲线")
    ax.plot(fixed.k, fixed.total_cost, marker="o", color=COLORS["dark"], lw=2, label="基准价格曲线")
    ax.scatter([2], [fixed.total_cost[2]], s=78, color=COLORS["固定 k=2"], zorder=4,
               label=f"k=2：{fixed.total_cost[2]:.2f} 元")
    ax.scatter([fixed.best_k], [fixed.total_cost[fixed.best_k]], s=90, color=COLORS["一直抽到齐"], zorder=4,
               label=f"最优 k={fixed.best_k}：{fixed.total_cost[fixed.best_k]:.2f} 元")
    ax.set_xlabel("预先固定盲抽次数 k"); ax.set_ylabel("理论期望成本（元）"); ax.legend(loc="upper left", ncol=2); clean_axis(ax)
    ax.set_title("固定策略在 k=3 处达到最低期望成本", loc="left")
    zoom = inset_axes(ax, width="43%", height="48%", loc="center right", borderpad=1.15)
    local = np.arange(1, 6)
    zoom.plot(local, fixed.total_cost[local], marker="o", color=COLORS["固定 k=3"], lw=1.5)
    zoom.scatter([2], [fixed.total_cost[2]], s=45, color=COLORS["固定 k=2"], zorder=4)
    zoom.scatter([3], [fixed.total_cost[3]], s=50, color=COLORS["一直抽到齐"], zorder=4)
    zoom.annotate(f"仅差 {fixed.total_cost[2] - fixed.total_cost[3]:.3f} 元",
                  xy=(2.5, (fixed.total_cost[2] + fixed.total_cost[3]) / 2), xytext=(3.35, 358.55),
                  fontsize=7.5, ha="center", arrowprops={"arrowstyle": "->", "color": "#56606B", "lw": .8})
    zoom.set_xticks(local); zoom.set_title("最优区域局部放大", fontsize=8.5)
    zoom.tick_params(labelsize=7); zoom.grid(axis="y", alpha=.2, linewidth=.6)
    zoom.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=.11, right=.98, bottom=.15, top=.96)
    out.append(save_both(fig, run.output_root, "main", "F03", "fixed_cost", "固定停止策略", "定位最优固定抽数", run.png_dpi))

    # 边际收益机制：继续一抽的期望节省与单抽价格在 k=2 和 k=3 之间交叉。
    ks = fixed.k[:-1]
    marginal_benefit = cfg.blind_price - fixed.marginal
    fig, ax = plt.subplots(figsize=(7.7, 4.5))
    ax.plot(ks, marginal_benefit, marker="o", color=COLORS["固定 k=3"], label="再抽一次的期望节省")
    ax.axhline(cfg.blind_price, color=COLORS["一直抽到齐"], ls="--", lw=1.6,
               label=f"单抽价格 c={cfg.blind_price:.1f} 元")
    ax.fill_between(ks, cfg.blind_price, marginal_benefit, where=marginal_benefit >= cfg.blind_price,
                    color=COLORS["green"], alpha=.16, label="继续可降低期望成本")
    ax.fill_between(ks, marginal_benefit, cfg.blind_price, where=marginal_benefit < cfg.blind_price,
                    color=COLORS["orange"], alpha=.12, label="继续会提高期望成本")
    for k in [2, 3]:
        ax.scatter(k, marginal_benefit[k], s=76, color=COLORS["动态停止"] if k == 2 else COLORS["一直抽到齐"], zorder=4)
        relation = "> c，继续" if marginal_benefit[k] > cfg.blind_price else "< c，停止"
        ax.annotate(f"k={k}: {marginal_benefit[k]:.2f} {relation}", (k, marginal_benefit[k]),
                    xytext=(10, 10 if k == 2 else -17), textcoords="offset points", fontsize=8.3)
    ax.set_xlabel("已固定抽取次数 k"); ax.set_ylabel("再抽一次的期望节省（元）")
    ax.set_xlim(0, cfg.fixed_k_max - 1); ax.legend(ncol=2, loc="upper right"); clean_axis(ax)
    ax.set_title("边际收益跨过单抽价格，解释了 k=3 的最优性", loc="left")
    fig.tight_layout()
    out.append(save_both(fig, run.output_root, "main", "F04", "fixed_marginal_mechanism", "固定停止策略",
                         "用边际收益与单抽价格的交叉解释固定最优点", run.png_dpi))

    margin = np.full((cfg.n, cfg.n), np.nan)
    for rec in dynamic.pair_records:
        i, j = sorted((rec["i"], rec["j"]))
        margin[i, j] = rec["G"] - cfg.blind_price
    limit = float(np.nanmax(np.abs(margin)))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    decision_cmap = LinearSegmentedColormap.from_list(
        "stop_continue", [COLORS["orange"], "#F7F7F7", COLORS["动态停止"]]
    )
    fig, ax = plt.subplots(figsize=(7.7, 6.2)); im = ax.imshow(np.ma.masked_invalid(margin), cmap=decision_cmap, norm=norm)
    stability = {r["pair"]: r for r in robustness.pair_records}
    for rec in dynamic.pair_records:
        i, j = sorted((rec["i"], rec["j"]))
        value = rec["G"] - cfg.blind_price
        action = "继续" if value > 0 else "停止"
        ax.text(j, i, f"{action}\n{value:+.2f}", ha="center", va="center", fontsize=7.2,
                color="white" if abs(value) > .62 * limit else "#222222", weight="bold" if abs(value) < .25 else "normal")
        record = stability[rec["pair"]]
        edge = "#E59B36" if record["classification"] == "边界敏感" else "#3B4650"
        style = "--" if record["classification"] == "边界敏感" else "-"
        ax.add_patch(Rectangle((j-.48, i-.48), .96, .96, fill=False, edgecolor=edge,
                               linewidth=1.55 if record["classification"] == "边界敏感" else .45,
                               linestyle=style))
    ax.set_xticks(range(cfg.n), cfg.short_roles, rotation=35, ha="right"); ax.set_yticks(range(cfg.n), cfg.short_roles)
    ax.set_xlim(-.5, cfg.n - .5); ax.set_ylim(cfg.n - .5, -.5)
    ax.set_xticks(np.arange(-.5, cfg.n, 1), minor=True); ax.set_yticks(np.arange(-.5, cfg.n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2); ax.tick_params(which="minor", bottom=False, left=False)
    ax.text(.02, .02, f"继续 10 种  |  停止 18 种\n格内数值为 G(U)-c（元）",
            transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left",
            bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "edgecolor": "#CBD1D8", "alpha": .94})
    ax.legend(handles=[Patch(facecolor="none", edgecolor="#E59B36", linestyle="--", label="Bootstrap 边界敏感"),
                       Patch(facecolor="none", edgecolor="#3B4650", label="Bootstrap 稳定")],
              loc="lower right" if get_language() == "en" else "upper right", fontsize=7.5)
    fig.colorbar(im, ax=ax, label="决策裕量 G(U)-c（元）"); fig.tight_layout()
    out.append(save_both(fig, run.output_root, "main", "F05", "dynamic_pair_heatmap", "价格感知动态停止", "判别 28 种两款状态", run.png_dpi))

    # 动态策略状态树：左侧给出动作分流，右侧单独解释平均抽数的三项来源。
    fig, (ax, calc) = plt.subplots(1, 2, figsize=(9.6, 5.15),
                                   gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": .04})
    ax.set_xlim(0, 6); ax.set_ylim(0, 10); ax.axis("off")
    calc.set_xlim(0, 1); calc.set_ylim(0, 1); calc.axis("off")
    ax.text(.15, 9.72, "状态树：仅在获得第 2 款时分流", fontsize=12, weight="bold",
            color=COLORS["dark"], va="top")
    nodes = {
        "start": (3.0, 8.55, "开始", "0 款", COLORS["固定 k=3"]),
        "one": (3.0, 6.65, "第 1 个不同款", "必继续", COLORS["固定 k=3"]),
        "two": (3.0, 4.75, "第 2 个不同款", "比较 $G(U)$ 与 $c$", COLORS["动态停止"]),
        "stop2": (1.35, 2.20, "停止并补齐", "18 / 28 个状态", COLORS["green"]),
        "cont": (4.65, 2.85, "继续抽取", "10 / 28 个状态", COLORS["动态停止"]),
        "stop3": (4.65, .80, "获得第 3 款", "全部停止", COLORS["green"]),
    }
    for key, (x, y0, title, subtitle, edge) in nodes.items():
        width = 2.00 if key in {"two", "stop2", "cont"} else 1.75
        height = 1.02
        fill = "#F1ECF7" if key in {"two", "cont"} else "#EAF5ED" if key in {"stop2", "stop3"} else "#EDF4F9"
        box = FancyBboxPatch((x-width/2, y0-height/2), width, height,
                             boxstyle="round,pad=.06,rounding_size=.10",
                             facecolor=fill, edgecolor=edge, linewidth=1.7)
        ax.add_patch(box)
        ax.text(x, y0+.15, title, ha="center", va="center", fontsize=9.0, weight="bold", color=COLORS["dark"])
        ax.text(x, y0-.20, subtitle, ha="center", va="center", fontsize=7.8, color="#56606B")
    ax.annotate("", xy=(3, 7.18), xytext=(3, 8.02), arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#88939E"})
    ax.text(3.15, 7.60, "第 1 抽", fontsize=7.5, color="#56606B", va="center")
    ax.annotate("", xy=(3, 5.28), xytext=(3, 6.12), arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#88939E"})
    ax.text(3.15, 5.70, "等到第 2 款", fontsize=7.5, color="#56606B", va="center")
    ax.annotate("", xy=(1.70, 2.73), xytext=(2.52, 4.22), arrowprops={"arrowstyle": "-|>", "lw": 1.55, "color": COLORS["green"]})
    ax.text(1.62, 3.55, "$G(U)\\leq c$", fontsize=8.2, color="#39774C", ha="center")
    ax.annotate("", xy=(4.30, 3.38), xytext=(3.48, 4.22), arrowprops={"arrowstyle": "-|>", "lw": 1.55, "color": COLORS["动态停止"]})
    ax.text(4.34, 4.02, "$G(U)>c$", fontsize=8.2, color=COLORS["动态停止"], ha="center")
    ax.annotate("", xy=(4.65, 1.33), xytext=(4.65, 2.32), arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#88939E"})
    ax.text(4.82, 1.80, "等到第 3 款", fontsize=7.5, color="#56606B", va="center")

    card = FancyBboxPatch((.06, .08), .88, .84, boxstyle="round,pad=.025,rounding_size=.03",
                          facecolor="#F7F8FA", edgecolor="#CBD2D9", linewidth=1.2)
    calc.add_patch(card)
    calc.text(.12, .86, "平均抽数的解析分解", fontsize=11, weight="bold", color=COLORS["dark"], va="top")
    items = [
        ("①", "获得第 1 款", "$1$", COLORS["固定 k=3"]),
        ("②", "从 1 款等到 2 款", "$8/7$", COLORS["固定 k=3"]),
        ("③", "仅继续状态等待第 3 款", "$(10/28)(8/6)$", COLORS["动态停止"]),
    ]
    for i, (num, label, value, color) in enumerate(items):
        y0 = .69 - i*.17
        calc.text(.13, y0, num, fontsize=10, weight="bold", color=color, va="center")
        calc.text(.21, y0+.025, label, fontsize=8.0, color="#56606B", va="center")
        calc.text(.83, y0-.025, value, fontsize=9.5, color=color, ha="right", va="center")
    calc.plot([.13, .86], [.27, .27], color="#CBD2D9", lw=1)
    calc.text(.13, .20, "$E(\\tau)=1+\\dfrac{8}{7}+\\dfrac{10}{28}\\dfrac{8}{6}$", fontsize=10.2, color=COLORS["dark"])
    calc.text(.13, .125, f"$={dynamic.exact_draws:.3f}$ 次", fontsize=11, color=COLORS["动态停止"], weight="bold")
    calc.text(.50, .025, f"精确期望成本  {dynamic.exact_cost:.2f} 元", fontsize=8.6, ha="center",
              color="#56606B", va="bottom")
    fig.subplots_adjust(left=.02, right=.99, bottom=.04, top=.97, wspace=.03)
    out.append(save_both(fig, run.output_root, "main", "F06", "dynamic_state_evolution", "价格感知动态停止",
                         "解释动态策略的状态分流与停止结构", run.png_dpi))

    # 场景边界：整盒无放回与散抽有放回不能直接混为同一随机机制。
    scenario_names = ["整盒 BOX（无放回）", "动态停止", "固定 k=3", "固定 k=2", "完全二手", "一直抽到齐"]
    scenario_values = [cfg.n * cfg.blind_price, dynamic.exact_cost, fixed.total_cost[3],
                       fixed.total_cost[2], cfg.direct_cost, collect.expected_cost]
    scenario_colors = [COLORS["green"], COLORS["动态停止"], COLORS["固定 k=3"],
                       COLORS["固定 k=2"], COLORS["完全二手"], COLORS["一直抽到齐"]]
    order = np.argsort(scenario_values)[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 5.0)); yy = np.arange(len(order))
    for pos, idx in enumerate(order):
        ax.plot([300, scenario_values[idx]], [pos, pos], color="#D6DBE1", lw=2.5)
        ax.scatter(scenario_values[idx], pos, s=86, color=scenario_colors[idx], zorder=3)
        ax.text(scenario_values[idx]+4, pos, f"{scenario_values[idx]:.2f}", va="center", fontsize=8.4)
    labels = [scenario_names[i] for i in order]
    ax.set_yticks(yy, labels); ax.invert_yaxis(); ax.set_xlim(300, max(scenario_values)+45)
    ax.axvspan(300, 330, color=COLORS["green"], alpha=.08)
    ax.set_xlabel("理论期望净成本（元）"); clean_axis(ax, "x")
    ax.set_title("无放回整盒与有放回散抽对应不同的随机场景", loc="left")
    ax.text(.99, .04, "BOX 行仅作情景边界；主模型采用散抽有放回机制", transform=ax.transAxes,
            ha="right", fontsize=8.2, color="#56606B")
    fig.tight_layout()
    out.append(save_both(fig, run.output_root, "main", "F14", "scenario_comparison", "模型适用范围",
                         "比较散抽策略与整盒无放回情景边界", run.png_dpi))

    dd = sims.draws["动态停止"]; values, counts = np.unique(dd, return_counts=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.3)); ax.bar(values, counts / counts.sum(), color=COLORS["动态停止"], alpha=.8)
    ax.axvline(dynamic.exact_draws, color=COLORS["dark"], ls="--", label=f"解析均值 {dynamic.exact_draws:.3f}")
    ax.set_xlabel("动态停止抽数"); ax.set_ylabel("概率"); ax.legend(); clean_axis(ax); fig.tight_layout()
    out.append(save_both(fig, run.output_root, "appendix", "A09", "dynamic_draws", "附录", "展示动态停止时刻分布", run.png_dpi))

    fig, ax = plt.subplots(figsize=(7.2, 4.2)); ax.axhline(0, color="#A8AFB8", lw=1)
    ax.stem(fixed.k[:-1], fixed.marginal, linefmt=COLORS["固定 k=3"], markerfmt="o", basefmt=" ")
    ax.set_xlabel("从 k 增加到 k+1"); ax.set_ylabel("边际期望成本（元）"); clean_axis(ax); fig.tight_layout()
    out.append(save_both(fig, run.output_root, "appendix", "A02", "fixed_marginal", "附录", "验证固定最优点的差分符号", run.png_dpi))

    fig, ax = plt.subplots(figsize=(7.4, 4.3)); ax.stackplot(fixed.k, fixed.draw_cost, fixed.missing_cost, -fixed.resale_income,
        labels=["盲抽支出", "补齐支出", "转售抵扣"], colors=[COLORS["固定 k=3"], COLORS["一直抽到齐"], COLORS["动态停止"]], alpha=.75)
    ax.plot(fixed.k, fixed.total_cost, color=COLORS["dark"], lw=2, label="净成本"); ax.set_xlabel("k"); ax.set_ylabel("金额（元）")
    ax.legend(ncol=2); clean_axis(ax); fig.tight_layout()
    out.append(save_both(fig, run.output_root, "appendix", "A03", "fixed_components", "附录", "拆解固定策略成本构成", run.png_dpi))

    fig, ax = plt.subplots(figsize=(7.2, 4.4)); hb = ax.hexbin(dd, sims.costs["动态停止"], gridsize=25, mincnt=1, cmap="Purples")
    fig.colorbar(hb, ax=ax, label="模拟频数"); ax.set_xlabel("动态停止抽数"); ax.set_ylabel("最终净成本（元）"); clean_axis(ax, None); fig.tight_layout()
    out.append(save_both(fig, run.output_root, "appendix", "A04", "dynamic_draw_cost", "附录", "诊断动态抽数与成本关系", run.png_dpi))
    return out
