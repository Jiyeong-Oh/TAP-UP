"""Fig. 3."""
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
ORDER = ('t2', 'adc', 'pet')
NAMES = {'t2': 'T2W', 'adc': 'ADC', 'pet': 'PSMA-PET'}
COLORS = {'t2': '#CE45D6', 'adc': '#6BD66B', 'pet': '#F6A040'}
(INK, MUTED, RULE) = ('#222222', '#6B6B6B', '#C9C9C9')



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='tapup')
    args = p.parse_args()
    out = f'{REPO}/outputs/{args.tag}'
    d = pd.read_csv(f'{out}/modality_shapley.csv')
    phi = d[[f'phi_{m}' for m in ORDER]].abs().to_numpy()
    share = phi / phi.sum(1, keepdims=True)
    dom = share.argmax(1)
    groups = [i for i in np.argsort(-np.bincount(dom, minlength=3))]
    (order, centers, gap, x, xs) = ([], [], 2.0, 0.0, [])
    for g in groups:
        idx = np.where(dom == g)[0]
        n = len(idx)
        idx = idx[np.argsort(-share[idx, g])]
        centers.append((x + (n - 1) / 2, g, n))
        order.extend(idx)
        xs.extend(x + np.arange(n))
        x += n + gap
    (xs, order) = (np.array(xs), np.array(order))
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral'], 'mathtext.fontset': 'stix', 'font.size': 6.5})
    (fig, ax) = plt.subplots(figsize=(3.4, 1.55))
    bottom = np.zeros(len(order))
    for g in groups:
        sel = np.where(dom[order] == g)[0]
        for j in [k for k in range(3) if k != g] + [g]:
            ax.bar(xs[sel], share[order[sel], j], bottom=bottom[sel], width=0.8, color=COLORS[ORDER[j]], linewidth=0, zorder=2)
            bottom[sel] += share[order[sel], j]
    bcr = d['label'].to_numpy()[order] == 1
    ax.scatter(xs[bcr], np.full(bcr.sum(), -0.07), marker='^', s=5, color=INK, linewidth=0, clip_on=False)
    for (c, g, n) in centers:
        (lo, hi) = (c - (n - 1) / 2 - 0.4, c + (n - 1) / 2 + 0.4)
        ax.plot([lo, hi], [1.04, 1.04], color=COLORS[ORDER[g]], lw=1.2, clip_on=False, solid_capstyle='butt')
        last = g == centers[-1][1]
        ax.text(hi if last else (lo + hi) / 2, 1.07, f'{NAMES[ORDER[g]]} ({n})', ha='right' if last else 'center', va='bottom', color=INK, fontsize=6.5)
    ax.set_ylim(0, 1)
    ax.set_xlim(xs.min() - 0.7, xs.max() + 0.7)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '50', '100'], color=MUTED)
    ax.set_ylabel('Share of $|\\phi|$ (%)', color=INK, labelpad=2)
    ax.set_xticks([])
    for s in ('top', 'right', 'bottom'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(RULE)
    ax.tick_params(axis='y', length=2, color=RULE, pad=1.5)
    handles = [Patch(color=COLORS[m], label=NAMES[m]) for m in ORDER]
    handles.append(plt.Line2D([], [], marker='^', linestyle='', color=INK, markersize=3, label='BCR'))
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=6.5, handlelength=0.9, handleheight=0.7, columnspacing=1.4, handletextpad=0.4, labelcolor=INK)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.87, bottom=0.2)
    fig.savefig(f'{out}/shapley_bars.png', dpi=300)
    fig.savefig(f'{out}/shapley_bars.pdf')
    print('wrote', out, 'groups', [(NAMES[ORDER[g]], n) for (_, g, n) in centers], 'median dominant share', round(float(np.median(share.max(1))), 2), 'range', round(float(share.max(1).min()), 2), round(float(share.max(1).max()), 2))
if __name__ == '__main__':
    main()
