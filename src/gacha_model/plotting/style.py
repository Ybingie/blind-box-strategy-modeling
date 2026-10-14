from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
COLORS = {
    'Secondary market only': '#697782',
    'Fixed k=2': '#2A9D8F',
    'Fixed k=3': '#3D7DB8',
    'Dynamic stopping': '#7A5AA6',
    'Draw until complete': '#D95F59',
    'orange': '#E99052',
    'green': '#5EAD78',
    'dark': '#30363D'
}

def setup_style() -> Optional[str]:
    chosen = 'DejaVu Sans'
    plt.rcParams['font.sans-serif'] = [chosen]
    plt.rcParams.update(
        {
            'axes.unicode_minus': False,
            'font.size': 10,
            'axes.labelsize': 10.5,
            'axes.titlesize': 11,
            'axes.titleweight': 'semibold',
            'axes.edgecolor': '#89939E',
            'axes.linewidth': 0.9,
            'lines.linewidth': 1.8,
            'lines.markersize': 5.5,
            'legend.fontsize': 8.5,
            'legend.frameon': False,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'xtick.direction': 'in',
            'ytick.direction': 'in',
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.transparent': False,
            'pdf.fonttype': 42,
            'svg.fonttype': 'none'
        }
    )
    return chosen

def clean_axis(ax, grid='y'):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#A8AFB8')
    if grid:
        ax.grid(axis=grid, alpha=0.18, linewidth=0.8)
        ax.set_axisbelow(True)

def save_figure(fig, root: Path, group: str, code: str, slug: str, chapter: str, purpose: str, dpi: int):
    # Use the same English font for labels, annotations and legends.
    from matplotlib.text import Text
    for artist in fig.findobj(match=Text):
        artist.set_fontfamily('DejaVu Sans')
    folder = root / 'figures_en' / group
    folder.mkdir(parents=True, exist_ok=True)
    stem = f'{code}_{slug}'
    png = folder / f'{stem}.png'
    svg = folder / f'{stem}.svg'
    pdf = folder / f'{stem}.pdf'
    fig.savefig(png, dpi=dpi, bbox_inches='tight', facecolor='white')
    fig.savefig(svg, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return {
        'figure_id': code,
        'language': 'en',
        'group': group,
        'chapter': chapter,
        'purpose': purpose,
        'png': png.relative_to(root).as_posix(),
        'svg': svg.relative_to(root).as_posix(),
        'pdf': pdf.relative_to(root).as_posix()
    }
