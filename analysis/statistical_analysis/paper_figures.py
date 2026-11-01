"""The three paper figures in one shared style.

Run huddle_feature_analysis.py first (it writes results/*.csv), then run this script.
Writes into overleaf/images/ (or shows inline if PAPER_FIGURES_SHOW is set).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import huddle_feature_analysis as h

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(ROOT, 'overleaf', 'images')
SHOW = bool(os.environ.get('PAPER_FIGURES_SHOW'))  # demo.ipynb sets this to display the figures instead of saving them

# ---- shared style --------------------------------------------------------------------
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#c3c2b7'
BLUE = '#2a78d6'
WON_C, LOST_C = '#2e9e5b', '#8e5bc4'
FIGSIZE, DPI = (9.0, 4.8), 200
LABEL_FS, TICK_FS, NOTE_FS = 12, 10, 8.5
MARGINS = dict(left=0.115, right=0.975, bottom=0.16, top=0.975)
plt.rcParams['font.family'] = 'DejaVu Sans'


def new_fig(ncols=1, sharey=False):
    fig, axes = plt.subplots(1, ncols, figsize=FIGSIZE, dpi=DPI, sharey=sharey)
    fig.patch.set_facecolor('white')
    fig.subplots_adjust(**MARGINS, **({'wspace': 0.06} if ncols > 1 else {}))
    return fig, (axes if ncols > 1 else [axes])


def style(ax, xlabel=None, ylabel=None, ytick_length=3.5, xtick_length=3.5):
    ax.set_facecolor('white')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(axis='x', colors=MUTED, labelsize=TICK_FS, length=xtick_length, color=GRID)
    ax.tick_params(axis='y', colors=MUTED, labelsize=TICK_FS, length=ytick_length, color=GRID)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=LABEL_FS, color=INK, labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=LABEL_FS, color=INK, labelpad=8)


def save(fig, name):
    if SHOW:
        plt.show()
        return
    fig.savefig(os.path.join(OUT, name), dpi=DPI, facecolor='white')
    plt.close(fig)
    print('saved', name)


# ---- 1. team-level scatter ---------------------------------------------------------------
sc = pd.read_csv(os.path.join(HERE, 'results', 'team_consistency.csv'))
r, p = stats.pearsonr(sc['auc'], sc['point_win_pct'])
print(f'scatter: n={len(sc)} r={r:+.3f} p={p:.3f}')
fig, (ax,) = new_fig()
ax.scatter(sc['auc'], sc['point_win_pct'], s=46, color=BLUE, alpha=0.85, edgecolors='white', linewidths=0.6, zorder=3)
slope, icpt = np.polyfit(sc['auc'], sc['point_win_pct'], 1)
xs = np.linspace(sc['auc'].min(), sc['auc'].max(), 100)
ax.plot(xs, slope * xs + icpt, color=BLUE, linewidth=1.5, alpha=0.4, zorder=2)
style(ax, 'Recovery Routine Consistency, per team', 'Point win %')
save(fig, 'auc_vs_winpct.png')

# ---- 2. won vs. lost game ----------------------------------------------------------------
gm = pd.read_csv(os.path.join(HERE, 'results', 'game_consistency.csv')).dropna(subset=['auc'])


def mean_ci(x):
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    t = stats.t.ppf(0.975, len(x) - 1)
    return m, m - t * se, m + t * se


rows = [('Won Game', WON_C, gm[gm['won'] == 1]['auc']), ('Lost Game', LOST_C, gm[gm['won'] == 0]['auc'])]
fig, (ax,) = new_fig()
for y, (name, c, g) in enumerate(rows):
    m, lo, hi = mean_ci(g)
    ax.hlines(y, lo, hi, color=c, linewidth=2.6, zorder=3)
    ax.plot(m, y, 'o', color=c, markersize=11, markeredgecolor='white', markeredgewidth=1.2, zorder=4)
ax.set_yticks([0, 1])
ax.set_yticklabels([r_[0] for r_ in rows], fontsize=LABEL_FS, color=INK)
ax.set_ylim(1.6, -0.6)
ax.set_xlim(0.5, 1.0)
ax.set_xticks(np.arange(0.5, 1.0001, 0.1))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.2f}'))
style(ax, 'Recovery Routine Consistency', None, ytick_length=0)
save(fig, 'recovery_routine_consistency_OLS.png')

# ---- 3. run-state profile ------------------------------------------------------------------
ORDER = h.RUN_STATE_ORDER
COL = {'W3+': '#1d5fb0', 'W2': '#2a78d6', 'W1': '#7fb0ea', 'L1': '#f2a47f', 'L2': '#eb6834', 'L3+': '#b8421a'}
HEAD = {'closeness': 'Closeness', 'duration': 'Duration', 'latency': 'Latency'}
df = h.prep(h.clean_long_df(pd.read_csv(h.LONG_DATASET_CSV_PATH), verbose=False))
df['rs'] = h.run_state_labels(df['momentum'])
df = df[df['rs'].notna()].copy()
df['rs'] = pd.Categorical(df['rs'], categories=['W1'] + [s for s in ORDER if s != 'W1'])


def stars(pv):
    return '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''


fig, axes = new_fig(3, sharey=True)
for ax, prop in zip(axes, h.PROPS):
    col = h.PROP_COL[prop] + '_z'
    d = df.dropna(subset=[col])
    f = smf.ols(f'{col} ~ C(rs) + C(game_id)', data=d).fit(cov_type='cluster', cov_kwds={'groups': d['huddle_label']})
    ci = f.conf_int()
    xs_, ys_ = [], []
    for i, s in enumerate(ORDER):
        if s == 'W1':
            b, lo, hi, pv = 0.0, 0.0, 0.0, np.nan
        else:
            k = f'C(rs)[T.{s}]'
            b, lo, hi, pv = f.params[k], ci.loc[k, 0], ci.loc[k, 1], f.pvalues[k]
        xs_.append(i), ys_.append(b)
        ax.vlines(i, lo, hi, color=COL[s], linewidth=2.6, zorder=3)
        ref = s == 'W1'
        ax.plot(i, b, 'o', color=COL[s], markersize=9 if ref else 11, markerfacecolor='white' if ref else COL[s],
                markeredgecolor=COL[s] if ref else 'white', markeredgewidth=2.0 if ref else 1.2, zorder=4)
        ax.text(i, hi + 0.05, 'ref' if ref else f'{b:+.2f}\n{stars(pv)}'.rstrip('\n'), ha='center', va='bottom',
                fontsize=NOTE_FS, color=INK, linespacing=0.9)
    ax.plot(xs_, ys_, color=MUTED, linewidth=1.0, alpha=0.6, zorder=2)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, fontsize=TICK_FS, color=INK)
    ax.set_xlim(-0.55, len(ORDER) - 0.45)
    style(ax, HEAD[prop], None, xtick_length=0)
axes[0].set_ylabel('Difference from W1 (within-game SD)', fontsize=LABEL_FS, color=INK, labelpad=8)
axes[0].set_ylim(-1.15, 1.15)
for ax in axes[1:]:
    ax.tick_params(axis='y', length=3.5)
save(fig, 'run_state_profile.png')
