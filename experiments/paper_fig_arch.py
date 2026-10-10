"""Architecture diagram of EnvFiLM (results/paper/fig_arch.pdf)."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'paper')


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.06', fc=fc, ec='k', lw=0.7))
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=7)


def arrow(ax, a, b, **kw):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=7, lw=0.7, color='k', **kw))


def main():
    os.makedirs(OUT, exist_ok=True)
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.6, 3.4)
    ax.axis('off')
    pc, ec, hc = '#dbe8f6', '#f6e3cf', '#e2f0d9'
    box(ax, 0.05, 2.35, 1.6, 0.8, 'particle channel\n$x_{t-N+1..t}$', pc)
    box(ax, 1.95, 2.35, 1.1, 0.8, 'log(1 + x)', pc)
    for k in range(3):
        box(ax, 3.35 + k * 1.25, 2.35, 1.05, 0.8, 'dilated conv\n$d=%s$' % ['1', '2,4,..', '64'][k], pc)
    box(ax, 7.25, 2.35, 1.15, 0.8, 'last + mean\nstate', pc)
    box(ax, 8.6, 1.4, 1.3, 0.9, 'MLP head\n$\\hat y_t$', hc)
    box(ax, 0.05, 0.35, 1.6, 0.8, 'on-board state\nRH, T, $RH_{opc}$, $T_{opc}$', ec)
    box(ax, 1.95, 0.35, 1.1, 0.8, 'clip $\\pm 3\\sigma$\nsummary $z$', ec)
    box(ax, 3.35, 0.35, 1.6, 0.8, 'environment\nencoder (MLP)', ec)
    box(ax, 5.25, 0.35, 1.9, 0.8, 'bounded FiLM\n$1+\\frac{1}{2}\\tanh(\\cdot)$, $\\frac{1}{2}\\tanh(\\cdot)$', ec)
    arrow(ax, (1.65, 2.75), (1.95, 2.75))
    arrow(ax, (3.05, 2.75), (3.35, 2.75))
    arrow(ax, (4.40, 2.75), (4.60, 2.75))
    arrow(ax, (5.65, 2.75), (5.85, 2.75))
    arrow(ax, (6.90, 2.75), (7.25, 2.75))
    arrow(ax, (8.40, 2.75), (8.95, 2.30))
    arrow(ax, (1.65, 0.75), (1.95, 0.75))
    arrow(ax, (3.05, 0.75), (3.35, 0.75))
    arrow(ax, (4.95, 0.75), (5.25, 0.75))
    for k in range(3):
        arrow(ax, (6.2, 1.15), (3.87 + k * 1.25, 2.35), connectionstyle='arc3,rad=0.0', ls='--')
    arrow(ax, (2.5, 0.35), (9.25, 1.40), connectionstyle='arc3,rad=0.25')
    ax.text(6.45, 1.55, 'scale and shift every\nconvolution layer', fontsize=6.5, style='italic')
    ax.text(4.3, -0.5, 'state summary also to the head', fontsize=6.5, style='italic')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_arch.pdf'))
    fig.savefig(os.path.join(OUT, 'fig_arch.png'), dpi=200)


if __name__ == '__main__':
    main()
