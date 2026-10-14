from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib import font_manager
from .i18n import get_language, localize_figure, translate_text

COLORS = {
    "完全二手": "#697782", "固定 k=2": "#2A9D8F", "固定 k=3": "#3D7DB8",
    "动态停止": "#7A5AA6", "一直抽到齐": "#D95F59",
    "orange": "#E99052", "green": "#5EAD78", "dark": "#30363D",
}


def setup_style(language: str = "zh") -> Optional[str]:
    for name in ["Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "Source Han Sans CN"]:
        try:
            font_manager.findfont(font_manager.FontProperties(family=name), fallback_to_default=False)
            plt.rcParams["font.sans-serif"] = [name]
            chosen = name
            break
        except Exception:
            chosen = None
    plt.rcParams.update({
        "axes.unicode_minus": False, "font.size": 10, "axes.labelsize": 10.5,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.edgecolor": "#89939E",
        "axes.linewidth": .9, "lines.linewidth": 1.8, "lines.markersize": 5.5,
        "legend.fontsize": 8.5, "legend.frameon": False,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.direction": "in", "ytick.direction": "in",
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.transparent": False, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    return chosen


def clean_axis(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#A8AFB8")
    if grid:
        ax.grid(axis=grid, alpha=0.18, linewidth=0.8)
        ax.set_axisbelow(True)


def save_both(fig, root: Path, group: str, code: str, slug: str, chapter: str, purpose: str, dpi: int):
    localize_figure(fig)
    language = get_language()
    folder = root / f"figures_{language}" / group
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"{code}_{slug}"
    png = folder / f"{stem}.png"
    svg = folder / f"{stem}.svg"
    pdf = folder / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"figure_id": code, "language": language, "group": group,
            "chapter": translate_text(chapter), "purpose": translate_text(purpose),
            "png": str(png), "svg": str(svg), "pdf": str(pdf)}
