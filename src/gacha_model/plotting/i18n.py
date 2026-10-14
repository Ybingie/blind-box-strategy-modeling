"""Figure-language support for the publication plots.

The numerical plotting code remains language-neutral.  Chinese strings are
translated immediately before a figure is saved, so both language versions
share exactly the same data, geometry, and visual style.
"""
from __future__ import annotations

import re
import math

_LANGUAGE = "zh"
_HAN = re.compile(r"[\u3400-\u9fff]")


EXACT_EN = {
    "完全二手": "Secondary market only",
    "固定 k=2": "Fixed $k=2$",
    "固定 k=3": "Fixed $k=3$",
    "动态停止": "Dynamic stopping",
    "一直抽到齐": "Draw until complete",
    "月正脸": "Light-front",
    "月侧脸": "Light-profile",
    "L浅灰": "L-light gray",
    "L深灰": "L-dark gray",
    "海砂甜笑": "Misa-smile",
    "海砂项圈": "Misa-choker",
    "尼亚": "Near",
    "梅洛": "Mello",
    "优化固定策略": "Optimized fixed-stop",
    "动态策略": "Dynamic strategy",
    "固定最优 $k^*$": "Optimal fixed $k^*$",
    "均值": "Mean",
    "概率": "Probability",
    "累计概率": "Cumulative probability",
    "理论值": "Theoretical value",
    "基准参数": "Baseline",
    "最优 k": "Optimal $k$",
    "模拟频数": "Simulation count",
    "净成本": "Net cost",
    "盲抽支出": "Blind-draw spending",
    "补齐支出": "Completion spending",
    "转售抵扣": "Resale offset",
    "附录": "Appendix",
}

# Ordered longest-first; these replacements also cover strings containing
# numerical values generated with f-strings.
PHRASE_EN = {
    "完全二手": "Secondary market only",
    "固定 k=2": "Fixed $k=2$",
    "固定 k=3": "Fixed $k=3$",
    "动态停止": "Dynamic stopping",
    "一直抽到齐": "Draw until complete",
    "从统一规则到状态决策：每一步只增加必要信息": "From a common baseline to state-dependent decisions",
    "125 条市场样本显示：不同款式价格差异明显": "Market prices vary substantially across the eight designs",
    "集齐停止具有明显右偏长尾": "Collect-all stopping has a pronounced right tail",
    "固定策略在 k=3 处达到最低期望成本": "The fixed-stop strategy is minimized at $k=3$",
    "边际收益跨过单抽价格，解释了 k=3 的最优性": "The marginal saving crosses the draw price at the optimum",
    "状态树：仅在获得第 2 款时分流": "State tree: decisions branch after the second distinct design",
    "无放回整盒与有放回散抽对应不同的随机场景": "A complete box and independent blind draws use different sampling mechanisms",
    "样本中位数与 95% Bootstrap 区间": "Sample median and 95% bootstrap interval",
    "价格 Bootstrap 95% 区间": "95% price-bootstrap interval",
    "Bootstrap 中位曲线": "Bootstrap median curve",
    "重复款期望回收值": "Expected resale value of a duplicate",
    "市场价格或回收价值（元）": "Market price or resale value (CNY)",
    "理论期望净成本（元）": "Expected net cost (CNY)",
    "理论期望成本（元）": "Expected cost (CNY)",
    "最低固定期望成本变化（元）": "Change in minimum fixed-strategy cost (CNY)",
    "最小期望净成本（元）": "Minimum expected net cost (CNY)",
    "最终净成本（元）": "Final net cost (CNY)",
    "平均净成本（元）": "Mean net cost (CNY)",
    "累计均值（元）": "Running mean (CNY)",
    "边际期望成本（元）": "Marginal expected cost (CNY)",
    "再抽一次的期望节省（元）": "Expected saving from one more draw (CNY)",
    "金额（元）": "Amount (CNY)",
    "预算阈值（元）": "Budget threshold (CNY)",
    "模拟均值 − 理论期望（元）": "Simulation mean $-$ theoretical expectation (CNY)",
    "决策裕量 G(U)-c（元）": "Decision margin $G(U)-c$ (CNY)",
    "均值与 P95（元，主体策略）": "Mean and P95 (CNY; main strategies)",
    "均值与 P95（元，集齐策略）": "Mean and P95 (CNY; collect-all)",
    "首次集齐所需抽数": "Draws required for the first complete set",
    "动态停止抽数": "Draws under dynamic stopping",
    "抽数": "Number of draws",
    "模拟概率": "Simulated probability",
    "尾部生存概率": "Tail survival probability",
    "理论均值": "Theoretical mean",
    "解析均值": "Analytical mean",
    "次的尾部概率": " draws, tail probability",
    "次": " draws",
    "价格样本重抽下选择“继续”的比例": "Proportion choosing Continue across price resamples",
    "已获得款编号组合": "Pair of acquired item indices",
    "Bootstrap 样本中的最优固定次数 k": "Optimal fixed draw count $k$ in bootstrap samples",
    "出现比例": "Proportion",
    "固定最优点的价格样本稳健性": "Price-sample robustness of the fixed optimum",
    "颜色表示基准决策；其余状态在重抽样中保持稳定": "Colors show baseline actions; all other states remain stable",
    "仅显示": "Showing only",
    "个边界敏感状态": " boundary-sensitive states",
    "边界敏感": "Boundary-sensitive",
    "稳定": "Stable",
    "继续状态数（共 28 种）": "Continue states (out of 28)",
    "固定策略：最优 k 的分区": "Fixed strategy: regions of optimal $k$",
    "动态策略：继续状态数的变化": "Dynamic strategy: number of Continue states",
    "单次盲抽价格 c（元）": "Blind-draw price $c$ (CNY)",
    "单次盲抽价格 c": "Blind-draw price $c$",
    "转售成功率 η": "Resale success probability $\\eta$",
    "二手价格缩放系数 λ": "Secondary-market price multiplier $\\lambda$",
    "二手价格缩放系数 λ": "Secondary-market price multiplier $\\lambda$",
    "平台费率 γ": "Platform fee rate $\\gamma$",
    "买方运费项 d（元）": "Buyer shipping cost $d$ (CNY)",
    "买方运费项 d": "Buyer shipping cost $d$",
    "卖方交易成本 h（元）": "Seller transaction cost $h$ (CNY)",
    "卖方交易成本 h": "Seller transaction cost $h$",
    "二手价格": "Secondary-market price",
    "盲抽价格": "Blind-draw price",
    "转售成功率": "Resale success probability",
    "主体随机策略": "Main randomized strategies",
    "集齐策略（独立纵轴）": "Collect-all strategy (separate scale)",
    "主体策略局部放大": "Zoomed view of main strategies",
    "超过": "Probability above",
    "元概率（%）": " CNY (%)",
    "超预算概率（%）": "Probability over budget (%)",
    "预算": "Budget",
    "误差棒：模拟均值的 95% Monte Carlo 置信区间": "Error bars: 95% Monte Carlo CIs for simulation means",
    "模拟次数（对数坐标）": "Simulation count (log scale)",
    "预先固定盲抽次数 k": "Prespecified blind-draw count $k$",
    "基准价格曲线": "Baseline-price curve",
    "最优区域局部放大": "Zoomed optimum region",
    "仅差": "Difference:",
    "最优 k=": "Optimal $k=$",
    "再抽一次的期望节省": "Expected saving from one more draw",
    "单抽价格": "Price per draw",
    "继续可降低期望成本": "Continuing lowers expected cost",
    "继续会提高期望成本": "Continuing raises expected cost",
    "已固定抽取次数 k": "Current fixed draw count $k$",
    "继续": "Continue",
    "停止": "Stop",
    "格内数值为": "Cell value:",
    "种": " states",
    "固定 k 丢失款式信息": "A fixed $k$ ignores item identity",
    "Bellman：256 个状态动作完全一致": "Bellman check: identical actions in all 256 states",
    "信息量": "State information",
    "无": "None",
    "边界基准": "Boundary baselines",
    "先回答：盲抽是否值得？": "Is blind drawing worthwhile?",
    "完全二手  vs.  集齐停止": "Secondary market vs. collect all",
    "不使用过程信息": "No process information",
    "固定停止": "Fixed stopping",
    "再回答：预先抽几次？": "How many draws should be planned?",
    "但": "but",
    "几乎等价": "is nearly equivalent",
    "只记录抽取次数": "Track only the draw count",
    "最后回答：此刻是否继续？": "Should the next draw be taken now?",
    "时继续": ": Continue",
    "记录缺失集合": "Track the missing-item set",
    "开始": "Start",
    "款": " designs",
    "第 1 个不同款": "First distinct design",
    "必继续": "Always continue",
    "第 2 个不同款": "Second distinct design",
    "比较": "Compare",
    "与": "and",
    "停止并补齐": "Stop and complete",
    "个状态": " states",
    "继续抽取": "Continue drawing",
    "获得第 3 款": "Obtain a third design",
    "全部停止": "Stop in all states",
    "第 1 抽": "First draw",
    "等到第 2 款": "Wait for the second design",
    "等到第 3 款": "Wait for the third design",
    "平均抽数的解析分解": "Analytical decomposition of the mean draw count",
    "获得第 1 款": "Obtain the first design",
    "从 1 款等到 2 款": "Wait from one to two designs",
    "仅继续状态等待第 3 款": "Wait for a third design only in Continue states",
    "精确期望成本": "Exact expected cost",
    "整盒 BOX（无放回）": "Complete BOX (without replacement)",
    "BOX 行仅作情景边界；主模型采用散抽有放回机制": "BOX is a scenario bound; the main model uses independent draws with replacement",
    "从 k 增加到 k+1": "Increase from $k$ to $k+1$",
    "动态策略抽数": "Draws under the dynamic strategy",
}


def set_language(language: str) -> None:
    global _LANGUAGE
    if language not in {"zh", "en"}:
        raise ValueError(f"Unsupported figure language: {language}")
    _LANGUAGE = language


def get_language() -> str:
    return _LANGUAGE


def translate_text(text: str) -> str:
    if _LANGUAGE == "zh" or not text:
        return text
    if text in EXACT_EN:
        return EXACT_EN[text]
    result = text
    for source in sorted(PHRASE_EN, key=len, reverse=True):
        result = result.replace(source, PHRASE_EN[source])
    result = result.replace("（", " (").replace("）", ")").replace("：", ": ").replace("，", ", ")
    result = result.replace("元", " CNY")
    return result


def localize_figure(fig) -> None:
    if _LANGUAGE == "zh":
        return
    from matplotlib.text import Text
    fig.canvas.draw()
    for ax in fig.axes:
        xlabels = [translate_text(label.get_text()) for label in ax.get_xticklabels()]
        ylabels = [translate_text(label.get_text()) for label in ax.get_yticklabels()]
        xticks = ax.get_xticks()
        yticks = ax.get_yticks()
        x_has_han = any(_HAN.search(label.get_text()) for label in ax.get_xticklabels())
        y_has_han = any(_HAN.search(label.get_text()) for label in ax.get_yticklabels())
        if x_has_han and all(math.isfinite(float(value)) for value in xticks):
            ax.set_xticks(xticks, xlabels)
        if y_has_han and all(math.isfinite(float(value)) for value in yticks):
            ax.set_yticks(yticks, ylabels)
    for artist in fig.findobj(match=Text):
        artist.set_text(translate_text(artist.get_text()))
        artist.set_fontfamily("DejaVu Sans")
    fig.canvas.draw()
    remaining = sorted({artist.get_text() for artist in fig.findobj(match=Text) if _HAN.search(artist.get_text())})
    if remaining:
        raise ValueError("Untranslated Chinese text in English figure: " + " | ".join(remaining))
