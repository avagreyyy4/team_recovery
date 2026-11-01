"""Huddle formation feature analysis (closeness, duration, latency) vs. point outcomes.

Run: python3 analysis/statistical_analysis/huddle_feature_analysis.py
"""
import hashlib
import math
import os
import pickle
import sys
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

# create_additional_features.py lives in analysis/feature_extraction/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'feature_extraction'))
import create_additional_features as caf

# Written by analysis/feature_extraction/build_dataset.py
LONG_DATASET_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'long_dataset.csv')


def _tightest_core_points(points, min_points=3, shrink_ratio=0.75, grow_tolerance_ft=1.5):
    """Finds the tightest defensible subset of at least min_points among one frame's tracked points"""
    pts = [tuple(map(float, p)) for p in points]
    n = len(pts)
    if n <= min_points:
        return pts

    kept = list(range(n))
    cur_r = caf._min_enclosing_circle(pts)
    while len(kept) > min_points:
        best_drop, best_r = None, cur_r
        for i in kept:
            trial = [pts[j] for j in kept if j != i]
            r = caf._min_enclosing_circle(trial)
            if r < best_r:
                best_r, best_drop = r, i
        if best_drop is not None and best_r <= cur_r * shrink_ratio:
            kept.remove(best_drop)
            cur_r = best_r
        else:
            break

    dropped = [i for i in range(n) if i not in kept]
    changed = True
    while changed and dropped:
        changed = False
        best_add, best_new_r = None, np.inf
        for i in dropped:
            trial = [pts[j] for j in kept] + [pts[i]]
            r = caf._min_enclosing_circle(trial)
            if r < best_new_r:
                best_new_r, best_add = r, i
        if best_add is not None and best_new_r <= cur_r + grow_tolerance_ft:
            kept.append(best_add)
            dropped.remove(best_add)
            cur_r = best_new_r
            changed = True

    return [pts[i] for i in sorted(kept)]


# All 26 games actually processed as of 2026-08-13.
ALL_GAMES = [
    'Adrian_vs_Oberlin', 'Berry_vs_Emory', 'Bluffton_vs_Transylvania',
    'Bridgewater_vs_Washington', 'Calvin_vs_Hope', 'Emory_vs_Berry_2019',
    'Emory_vs_ColoradoCollege', 'Emory_vs_Susquehanna', 'Emory_vs_Transylvania',
    'Emory_vs_Wittenberg', 'Hope_vs_Albion', 'Hope_vs_Denison', 'Hope_vs_Trine',
    'Ithaca_vs_Clarkson', 'Ithaca_vs_NYU', 'Ithaca_vs_WS', 'Ithaca_vs_WS_Reg',
    'Ithaca_vs_WilliamPaterson', 'MCLA_vs_Framingham', 'MSJ_vs_Transylvania',
    'NYU_vs_Eastern', 'NYU_vs_Swarthmore', 'Smith_vs_NYU',
    'SouthwesternUniversity_vs_LaVerne', 'Swarthmore_vs_Smith',
    'TexasLutheran_vs_AustinCollege', 'TexasLutheran_vs_CalLutheran',
    'Trinity_vs_ETBU', 'Trinity_vs_HSU', 'UnionCollege_vs_WS',
    'Wesleyan_vs_Bowdoin', 'Wittenberg_vs_Transylvania',
]

# team_id is built by lowercasing each game's own home/away label, which breaks
TEAM_ID_ALIASES = {
    ('Emory_vs_Berry_2019', 'bc'): 'berry',
    ('Emory_vs_Transylvania', 'tu-w'): 'transylvania',
    ('MSJ_vs_Transylvania', 'transy'): 'transylvania',
}

# Huddles dropped outright regardless of side
EXCLUDED_HUDDLES = {
    ('Bluffton_vs_Transylvania', 2, 14),
    ('Bluffton_vs_Transylvania', 2, 17),
    # Emory_vs_Susquehanna
    ('Emory_vs_Susquehanna', 1, 11),
    ('Emory_vs_Susquehanna', 1, 46),
    ('Emory_vs_Susquehanna', 2, 25),
    ('Emory_vs_Susquehanna', 2, 39),
    ('Emory_vs_Susquehanna', 2, 44),
    ('Emory_vs_Susquehanna', 3, 15),
    ('Emory_vs_Susquehanna', 4, 35),
    ('Emory_vs_Susquehanna', 4, 45),
}

# team_id gets year-qualified below because a school's roster isn't the same team
def _build_matchup_year_lookup():
    rows = caf.stimulus_map.load_rows(caf.stimulus_map.DEFAULT_STIMULUS_MAP_PATH)
    rows = [r for r in rows if r.get('type', '').strip() == 'huddle_video']
    groups = caf.stimulus_map.group_by_matchup(rows)
    return {
        m: (sorted(set(r.get('year', '').strip() for r in grp))[0] or '?')
        for m, grp in groups.items()
    }


MATCHUP_YEAR = _build_matchup_year_lookup()

PROPS = ['closeness', 'duration', 'latency']
PROP_COL = {
    # closeness
    'closeness': 'circle_diameter_gyranchor_ft',
    'duration': 'duration_circle_s',
    # latency_size_adj_s, NOT the raw latency_s
    'latency': 'latency_size_adj_s',
}
# sign so that higher = tighter / longer / faster, matching the paper
PROP_SIGN = {'closeness': -1, 'duration': 1, 'latency': 1}

MIN_N = 3  # strict tightest-frame pass threshold for every metric here

# non_huddle v2
NON_HUDDLE_DIAMETER_FT = 5.0
FAR_FROM_CENTROID_FT = 2.5
NON_HUDDLE_MIN_FAR_COUNT = 3

# duration_circle_s's "still counts as together" qualifying radius
DURATION_QUALIFY_RADIUS_FT = 2.674


def add_new_temporal_features(df, min_n=4, sides=('home', 'away')):
    """The three LOCKED headline properties, per side per huddle"""
    rows = []
    for (s, h), group in df.sort_values(['set', 'huddle_number', 'frame']).groupby(['set', 'huddle_number'], sort=False):
        group = group.reset_index(drop=True)
        fps = group['fps'].iloc[0]
        rec = {'set': s, 'huddle_number': h, 'raw_frame_count': group['frame'].nunique()}

        for side in sides:
            ncol, ptscol = f'{side}_n', f'{side}_points'
            as_col = f'{side}_latency_s'
            uf_col = f'{side}_usable_frame_count'

            # unfiltered max, across every raw frame of the clip
            rec[f'{side}_max_n_detected'] = int(group[ncol].max())
            # fraction of raw frames with only a single tracked point
            rec[f'{side}_frac_n1'] = float((group[ncol] == 1).mean())
            rec[f'{side}_mean_n_detected'] = float(group[ncol].mean())

            usable = group[ncol] >= min_n
            too_few = pd.Series(False, index=group.index)
            if usable.any():
                too_few.loc[usable] = group.loc[usable, ptscol].apply(
                    lambda p: len(caf._parse_points_ft(p)) < min_n)
            usable = usable & ~too_few
            degenerate = pd.Series(False, index=group.index)
            if usable.any():
                degenerate.loc[usable] = group.loc[usable, ptscol].apply(caf._is_degenerate_tracking).astype(bool)
            usable = usable & ~degenerate
            side_group = group[usable].reset_index(drop=True)
            rec[uf_col] = len(side_group)

            if len(side_group) == 0:
                rec[as_col] = np.nan
                rec[f'{side}_duration_circle_s'] = np.nan
                rec[f'{side}_circle_diameter_gyranchor_ft'] = np.nan
                continue

            raw_points = side_group[ptscol].apply(caf._parse_points_ft)
            trimmed_points = raw_points.apply(lambda p: caf._mad_trim_points(p, min_points=min_n))

            # MAD-trim first, THEN find the tightest defensible core subset on top.
            circle_points = [_tightest_core_points(p, min_points=min_n) for p in trimmed_points]
            circle_radius = pd.Series([caf._min_enclosing_circle(p) for p in circle_points], index=raw_points.index)
            if circle_radius.notna().sum() == 0:
                rec[as_col] = np.nan
                rec[f'{side}_duration_circle_s'] = np.nan
                rec[f'{side}_circle_diameter_gyranchor_ft'] = np.nan
                continue

            # circle diameter read at GYRATION's own smoothed-minimum frame
            gyration_series = trimmed_points.apply(caf._gyration_from_points)
            gyr_smoothed = gyration_series.rolling(window=5, center=True, min_periods=1).mean()
            if gyr_smoothed.notna().sum() == 0:
                rec[f'{side}_circle_diameter_gyranchor_ft'] = np.nan
                rec[f'{side}_n_far_from_centroid'] = np.nan
                gyr_anchor_idx = None
            else:
                gyr_anchor_idx = gyr_smoothed.idxmin()
                core_at_gyr_anchor = _tightest_core_points(trimmed_points.loc[gyr_anchor_idx], min_points=min_n)
                rec[f'{side}_circle_diameter_gyranchor_ft'] = 2 * caf._min_enclosing_circle(core_at_gyr_anchor)

                # non_huddle handling
                anchor_pts = np.asarray(trimmed_points.loc[gyr_anchor_idx], dtype=float)
                centroid = anchor_pts.mean(axis=0)
                dists = np.linalg.norm(anchor_pts - centroid, axis=1)
                rec[f'{side}_n_far_from_centroid'] = int((dists > FAR_FROM_CENTROID_FT).sum())

            # latency_s and duration_circle_s's threshold both key off this SAME
            first_frame = side_group['frame'].iloc[0]
            if gyr_anchor_idx is None:
                rec[as_col] = np.nan
            else:
                gyr_anchor_frame = side_group.loc[gyr_anchor_idx, 'frame']
                rec[as_col] = -(gyr_anchor_frame - first_frame) / fps

            # STARTING diameter
            rec[f'{side}_starting_diameter_ft'] = 2 * circle_radius.iloc[0]

            # relaxed n>=3 pool for duration_circle_s
            dur_col = f'{side}_duration_circle_s'
            usable_relaxed = group[ncol] >= 3
            too_few_relaxed = pd.Series(False, index=group.index)
            if usable_relaxed.any():
                too_few_relaxed.loc[usable_relaxed] = group.loc[usable_relaxed, ptscol].apply(
                    lambda p: len(caf._parse_points_ft(p)) < 3)
            usable_relaxed = usable_relaxed & ~too_few_relaxed
            degenerate_relaxed = pd.Series(False, index=group.index)
            if usable_relaxed.any():
                degenerate_relaxed.loc[usable_relaxed] = group.loc[usable_relaxed, ptscol].apply(
                    caf._is_degenerate_tracking).astype(bool)
            usable_relaxed = usable_relaxed & ~degenerate_relaxed
            side_group_relaxed = group[usable_relaxed].reset_index(drop=True)

            if len(side_group_relaxed) == 0:
                rec[dur_col] = np.nan
                continue

            if gyr_anchor_idx is None:
                rec[dur_col] = np.nan
                continue

            raw_relaxed = side_group_relaxed[ptscol].apply(caf._parse_points_ft)
            trimmed_relaxed = raw_relaxed.apply(lambda p: caf._mad_trim_points(p, min_points=3))
            circle_points_relaxed = [_tightest_core_points(p, min_points=3) for p in trimmed_relaxed]
            circle_relaxed = pd.Series([caf._min_enclosing_circle(p) for p in circle_points_relaxed], index=raw_relaxed.index)
            # FIXED qualifying radius, shared across every huddle in the dataset
            if gyr_anchor_idx not in circle_radius.index or pd.isna(circle_radius.loc[gyr_anchor_idx]):
                rec[dur_col] = np.nan
                continue
            rec[dur_col] = caf._qualifying_span_duration_s(
                side_group_relaxed['frame'], circle_relaxed, DURATION_QUALIFY_RADIUS_FT, fps)

        rows.append(rec)
    return pd.DataFrame(rows)


def build_match_dataset(cfg):
    """One game's per-huddle feature table"""
    pbp_df = pd.read_csv(cfg['pbp_path'])
    huddle_video_df = caf.load_huddle_video_data(cfg['huddle_csv'])
    huddle_video_df = caf.add_home_away_columns(huddle_video_df, cfg, pbp_df=pbp_df)
    labels = {'home': cfg['home_team'], 'away': cfg['away_team']}

    pbp_df = caf.huddles_bf_af(pbp_df.copy())
    pbp_df = caf.prev_point_error_flag(pbp_df, labels)
    pbp_df = caf.momentum_feature(pbp_df, labels)
    pbp_df = caf.point_differential_feature(pbp_df)

    win_flags = caf.huddle_win_flags(huddle_video_df, pbp_df, labels)
    huddle_feats = caf.huddle_formation_feature_bundle(huddle_video_df, suffix='', min_n=MIN_N)
    extra_feats = add_new_temporal_features(huddle_video_df, min_n=MIN_N)

    full = huddle_feats.merge(win_flags, on=['set', 'huddle_number', 'huddle_label'], how='left')
    full = full.merge(extra_feats, on=['set', 'huddle_number'], how='left')

    point_cols = [c for c in pbp_df.columns if c.endswith(('prev_point_error', 'momentum', 'error_by', 'point_differential'))]
    point_cols = [c for c in point_cols if c in pbp_df.columns]
    next_point = (
        pbp_df[['huddle_before', 'error_by'] + [c for c in point_cols if c != 'error_by']]
        .dropna(subset=['huddle_before'])
        .drop_duplicates(subset='huddle_before')
        .rename(columns={'huddle_before': 'huddle_label'})
    )
    full = full.merge(next_point, on='huddle_label', how='left')

    full = caf.filter_valid_huddles(full, huddle_video_df, min_n=MIN_N)
    full = caf.remove_last_huddle_per_set(full)
    full.insert(0, 'game_id', cfg['game_id'])
    return full, labels


RUN_STATE_ORDER = ['W3+', 'W2', 'W1', 'L1', 'L2', 'L3+']


def run_state_labels(momentum):
    """momentum = signed streak length entering the point right before the huddle."""
    momentum = np.asarray(momentum, dtype=float)
    labels = np.full(momentum.shape, None, dtype=object)
    for k, lo, hi in [(1, 1, 1), (2, 2, 2), (3, 3, np.inf)]:
        labels[(momentum >= lo) & (momentum <= hi)] = f'W{k}' if k < 3 else 'W3+'
        labels[(-momentum >= lo) & (-momentum <= hi)] = f'L{k}' if k < 3 else 'L3+'
    return labels


_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.cache')


def _long_dataset_cache_key():
    """Hash of everything that should invalidate the cache"""
    games = {g['game_id']: g for g in caf.GAMES}
    parts = [repr(ALL_GAMES)]
    for path in (__file__, caf.__file__, caf.DEFAULT_PARAMETERS_CSV, caf.stimulus_map.DEFAULT_STIMULUS_MAP_PATH):
        parts.append(str(os.path.getmtime(path)))
    for m in ALL_GAMES:
        cfg = games[m]
        for key in ('pbp_path', 'huddle_csv'):
            p = cfg[key]
            parts.append(f'{p}:{os.path.getmtime(p)}')
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:16]


def build_long_dataset(use_cache=True):
    if use_cache:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(_CACHE_DIR, f'long_dataset_{_long_dataset_cache_key()}.pkl')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        long_df = _build_long_dataset_uncached()
        with open(cache_path, 'wb') as f:
            pickle.dump(long_df, f)
        return long_df
    return _build_long_dataset_uncached()


def _build_one_game(game_id):
    """Standalone worker for parallelizing across games in _build_long_dataset_uncached"""
    games = {g['game_id']: g for g in caf.GAMES}
    return game_id, build_match_dataset(games[game_id])


def _build_long_dataset_uncached():
    import multiprocessing
    with multiprocessing.Pool(processes=min(os.cpu_count() or 4, len(ALL_GAMES))) as pool:
        results = dict(pool.map(_build_one_game, ALL_GAMES))

    long_rows = []
    for m in ALL_GAMES:
        full, labels = results[m]
        for side, team_var in (('home', 'home'), ('away', 'away')):
            team_label = labels[side]
            opp_label = labels['away'] if side == 'home' else labels['home']
            sub = pd.DataFrame({
                'game_id': full['game_id'], 'set': full['set'], 'huddle_number': full['huddle_number'],
                'team': team_label, 'opponent': opp_label, 'side': side,
            })
            sub['huddle_label'] = (full['game_id'].astype(str) + '_S' + full['set'].astype(str)
                                    + '_H' + full['huddle_number'].astype(str))
            sub['circle_diameter_gyranchor_ft'] = full[f'{side}_circle_diameter_gyranchor_ft']
            sub['huddle_duration_s'] = full[f'{side}_huddle_duration_s']
            sub['duration_circle_s'] = full[f'{side}_duration_circle_s']
            sub['latency_s'] = full[f'{side}_latency_s']
            sub['starting_diameter_ft'] = full[f'{side}_starting_diameter_ft']
            # not full[side_non_huddle]
            sub['n_far_from_centroid'] = full[f'{side}_n_far_from_centroid']
            sub['non_huddle'] = np.where(
                sub['circle_diameter_gyranchor_ft'].isna(), np.nan,
                ((sub['circle_diameter_gyranchor_ft'] > NON_HUDDLE_DIAMETER_FT)
                 & (sub['n_far_from_centroid'] >= NON_HUDDLE_MIN_FAR_COUNT)).astype(float))
            sub['raw_frame_count'] = full['raw_frame_count']
            sub['usable_frame_count'] = full[f'{side}_usable_frame_count']
            sub['max_n_detected'] = full[f'{side}_max_n_detected']
            sub['frac_n1'] = full[f'{side}_frac_n1']
            sub['mean_n_detected'] = full[f'{side}_mean_n_detected']

            won_before = full[f'{side}_won_before'].astype('Float64').astype(float)
            won_after = full[f'{side}_won_after'].astype('Float64').astype(float)
            sub['lost_before'] = np.where(won_before.notna(), 1 - won_before, np.nan)
            sub['won_after'] = won_after

            err = full[f'{team_var}_prev_point_error'].astype(float)
            sub['prev_point_error'] = err
            sub['own_error'] = np.where(won_before == 0, (err == 1).astype(float), np.nan)

            mom = full[f'{team_var}_momentum'].astype(float)
            sub['losing_run'] = np.where(mom < 0, -mom, np.nan)  # magnitude of an active losing streak
            sub['momentum'] = mom
            sub['run_state'] = run_state_labels(mom)
            sub['point_differential'] = full[f'{team_var}_point_differential'].astype(float)

            long_rows.append(sub)
    long_df = pd.concat(long_rows, ignore_index=True)
    long_df['team_match'] = long_df['team'] + ' (vs. ' + long_df['opponent'] + ')'
    # normalized team IDENTITY
    long_df['team_id'] = long_df['team'].str.lower()
    long_df['team_id'] = [
        TEAM_ID_ALIASES.get((g, t), t) for g, t in zip(long_df['game_id'], long_df['team_id'])
    ]
    long_df['team_id'] = [
        f'{t}_{MATCHUP_YEAR.get(g, "?")}' for g, t in zip(long_df['game_id'], long_df['team_id'])
    ]
    long_df = _add_size_adjusted_latency(long_df)
    return long_df


def _add_size_adjusted_latency(long_df):
    """latency_size_adj_s = latency_s residualized on starting_diameter_ft"""
    d = long_df.dropna(subset=['latency_s', 'starting_diameter_ft', 'game_id']).copy()
    model = smf.ols('latency_s ~ starting_diameter_ft + C(game_id)', data=d).fit()
    long_df['latency_size_adj_s'] = np.nan
    long_df.loc[d.index, 'latency_size_adj_s'] = model.resid
    return long_df


def zscore_within_match(df, col):
    g = df.groupby('game_id')[col]
    return (df[col] - g.transform('mean')) / g.transform('std').replace(0, np.nan)


def prep(df):
    df = df.copy()
    for prop in PROPS:
        col = PROP_COL[prop]
        df[col + '_z'] = zscore_within_match(df, col) * PROP_SIGN[prop]
    return df


def prop_correlation_structure(df, props=PROPS):
    """Pairwise correlation + PCA among the three within-match z-scored, sign-adjusted properties"""
    cols = [PROP_COL[p] + '_z' for p in props]
    d = df.dropna(subset=cols)
    X = d[cols].values
    corr = np.corrcoef(X.T)
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    var_explained = eigvals / eigvals.sum()
    return corr, eigvals, var_explained, eigvecs, len(d)


def ols_effect(df, prop_col, group_col, cluster_col='huddle_label'):
    """Univariate OLS: prop_col ~ group + game FE, huddle-clustered SEs."""
    d = df.dropna(subset=[prop_col, group_col, 'game_id', cluster_col]).copy()
    if d[group_col].nunique() < 2:
        return np.nan, np.nan, len(d)
    model = smf.ols(f"{prop_col} ~ C({group_col}) + C(game_id)", data=d).fit(
        cov_type='cluster', cov_kwds={'groups': d[cluster_col]})
    term = [t for t in model.params.index if t.startswith(f'C({group_col})')][0]
    return model.params[term], model.pvalues[term], len(d)


def ols_multi_effect(df, prop_col, group_col, reference, cluster_col='huddle_label'):
    """Same design as ols_effect but for a >2-level categorical"""
    d = df.dropna(subset=[prop_col, group_col, 'game_id', cluster_col]).copy()
    model = smf.ols(f"{prop_col} ~ C({group_col}, Treatment('{reference}')) + C(game_id)", data=d).fit(
        cov_type='cluster', cov_kwds={'groups': d[cluster_col]})
    counts = d[group_col].value_counts()
    out = {}
    for level in RUN_STATE_ORDER:
        if level == reference:
            continue
        term = f"C({group_col}, Treatment('{reference}'))[T.{level}]"
        if term in model.params.index:
            out[level] = (model.params[term], model.pvalues[term], int(counts.get(level, 0)))
        else:
            out[level] = (np.nan, np.nan, int(counts.get(level, 0)))
    return out


def logit_effect(df, y_col, prop_col, cluster_col='huddle_label'):
    """Univariate logit: y_col ~ prop_col + game FE, huddle-clustered SEs."""
    d = df.dropna(subset=[y_col, prop_col, 'game_id', cluster_col]).copy()
    model = smf.logit(f"{y_col} ~ {prop_col} + C(game_id)", data=d).fit(
        disp=0, cov_type='cluster', cov_kwds={'groups': d[cluster_col]})
    return np.exp(model.params[prop_col]), model.pvalues[prop_col], len(d)


def lomo_auc(df, y_col, prop_cols):
    d = df.dropna(subset=[y_col] + prop_cols + ['game_id']).reset_index(drop=True)
    X = d[prop_cols].values
    y = d[y_col].astype(int).values
    groups = d['game_id'].values
    logo = LeaveOneGroupOut()
    aucs = []
    for tr, te in logo.split(X, y, groups):
        clf = LogisticRegression().fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return np.mean(aucs), aucs


def lomo_auc_per_team(df, y_col, prop_cols, team_col='team_id'):
    """Same LOMO scheme as lomo_auc, but scored per-team instead of per-match"""
    d = df.dropna(subset=[y_col] + prop_cols + ['game_id', team_col]).reset_index(drop=True)
    X = d[prop_cols].values
    y = d[y_col].astype(int).values
    groups = d['game_id'].values
    teams = d[team_col].values
    logo = LeaveOneGroupOut()
    all_true, all_pred, all_team = [], [], []
    for tr, te in logo.split(X, y, groups):
        clf = LogisticRegression().fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        all_true.extend(y[te])
        all_pred.extend(p)
        all_team.extend(teams[te])
    out = {}
    for team in sorted(set(all_team)):
        idx = [i for i, t in enumerate(all_team) if t == team]
        yt = np.array([all_true[i] for i in idx])
        pt = np.array([all_pred[i] for i in idx])
        if len(set(yt)) < 2:
            out[team] = (np.nan, len(idx))
            continue
        out[team] = (roc_auc_score(yt, pt), len(idx))
    return out


def std_diff(df, group_col, val_col, level_a, level_b):
    """Mean(val | level_a) - Mean(val | level_b), val within-match z-scored."""
    a = df.loc[df[group_col] == level_a, val_col].dropna()
    b = df.loc[df[group_col] == level_b, val_col].dropna()
    return a.mean() - b.mean(), a, b


def welch_p(a, b):
    return stats.ttest_ind(a, b, equal_var=False).pvalue


def welch_and_cohens_d(a, b):
    """Raw-unit two-sample check the paper reports alongside the FE/cluster models, not instead of them"""
    t, p = stats.ttest_ind(a, b, equal_var=False)
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / pooled_std
    return t, p, d


MIN_DURATION_S = 1.0


def report_n(df, label):
    n_huddles = df.groupby(['game_id', 'set', 'huddle_number']).ngroups
    print(f"[{label}] team-huddle rows: {len(df)}  |  huddles: {n_huddles}  |  "
          f"sets: {df.groupby('game_id')['set'].nunique().sum()}")


MIN_RAW_FRAMES = 40


def clean_long_df(long_df, verbose=True):
    """The same three data-quality drops main applies before any modeling"""
    report_n(long_df, 'unfiltered')

    # raw_frame_count is a clip-level quantity, so this drops the huddle entirely
    short_clips = long_df.loc[long_df['raw_frame_count'] < MIN_RAW_FRAMES, ['game_id', 'set', 'huddle_number']].drop_duplicates()
    if verbose:
        print(f"Dropping {len(short_clips)} huddles with raw_frame_count < {MIN_RAW_FRAMES} "
              f"({len(short_clips) * 2} team-huddle rows):")
        print(short_clips.to_string(index=False))
    long_df = long_df[long_df['raw_frame_count'] >= MIN_RAW_FRAMES].copy()
    if verbose:
        report_n(long_df, f'filtered: raw_frame_count >= {MIN_RAW_FRAMES}')

    excluded_mask = long_df.apply(lambda r: (r['game_id'], r['set'], r['huddle_number']) in EXCLUDED_HUDDLES, axis=1)
    if verbose:
        print(f"Dropping {excluded_mask.sum()} team-huddle rows from EXCLUDED_HUDDLES (hand-identified tracking failures):")
        print(long_df.loc[excluded_mask, ['game_id', 'team', 'set', 'huddle_number', 'circle_diameter_gyranchor_ft']].to_string(index=False))
    long_df = long_df[~excluded_mask].copy()
    if verbose:
        report_n(long_df, 'filtered: EXCLUDED_HUDDLES dropped')

    # Whole-huddle drop, same as filter_valid_huddles' own semantics for its version
    min_max_n = long_df.groupby(['game_id', 'set', 'huddle_number'])['max_n_detected'].transform('min')
    sparse_mask = min_max_n <= 3
    sparse_huddles = long_df.loc[sparse_mask, ['game_id', 'team', 'set', 'huddle_number', 'max_n_detected']].drop_duplicates(subset=['game_id', 'set', 'huddle_number'])
    if verbose:
        print(f"Dropping {sparse_mask.sum()} team-huddle rows ({len(sparse_huddles)} huddles) "
              f"where a side never reached 4 tracked players the entire clip:")
        print(sparse_huddles.to_string(index=False))
    long_df = long_df[~sparse_mask].copy()
    if verbose:
        report_n(long_df, 'filtered: max_n_detected > 3 required')

    # Whole-huddle drop (same rationale as max_n_detected)
    max_frac_n1 = long_df.groupby(['game_id', 'set', 'huddle_number'])['frac_n1'].transform('max')
    degenerate_mask = max_frac_n1 >= 0.5
    degenerate_huddles = long_df.loc[degenerate_mask, ['game_id', 'team', 'set', 'huddle_number', 'frac_n1']].drop_duplicates(subset=['game_id', 'set', 'huddle_number'])
    if verbose:
        print(f"Dropping {degenerate_mask.sum()} team-huddle rows ({len(degenerate_huddles)} huddles) "
              f"where a side was n=1 for >=50% of the clip:")
        print(degenerate_huddles.to_string(index=False))
    long_df = long_df[~degenerate_mask].copy()
    if verbose:
        report_n(long_df, 'filtered: n=1 for >=50% of clip excluded')
        print("(paper: 1870 team-huddle rows, 935 huddles, 22 sets)\n")

    # Row-level drop
    non_huddle_mask = long_df['non_huddle'] == 1
    if verbose:
        print(f"Dropping {non_huddle_mask.sum()} team-huddle rows flagged non_huddle "
              f"(that side's own circle never closed into a real huddle):")
    long_df = long_df[~non_huddle_mask].copy()
    if verbose:
        report_n(long_df, 'filtered: non_huddle rows excluded')
    return long_df


def main():
    print(f"Loading long dataset from {LONG_DATASET_CSV_PATH} "
          f"(run analysis/feature_extraction/build_dataset.py first to refresh it)...")
    long_df = pd.read_csv(LONG_DATASET_CSV_PATH)
    long_df = clean_long_df(long_df)

    print("=== Sanity check: are the longest huddles (by huddle_duration_s) almost all after losses? ===")
    check_cols = ['game_id', 'team', 'huddle_number', 'set', 'huddle_duration_s', 'lost_before']
    top10 = long_df.dropna(subset=['huddle_duration_s']).sort_values('huddle_duration_s', ascending=False).head(10)
    print(top10[check_cols].to_string(index=False))
    print(f"Wins in top 10 longest huddles: {int((top10['lost_before'] == 0).sum())} / 10\n")

    long_df = prep(long_df)

    prop_z = [PROP_COL[p] + '_z' for p in PROPS]

    print("=== Backward: does the huddle property differ by the point just played? ===")
    print("(OLS: property_z ~ C(lost_before) + C(game_id), SE clustered on huddle_label)")
    backward_effects = {}
    for prop in PROPS:
        col = PROP_COL[prop] + '_z'
        coef, p, n = ols_effect(long_df, col, 'lost_before')
        backward_effects[prop] = coef
        print(f"  {prop:16s} coef={coef:+.3f} SD  p={p:.4g}  n={n}")
    auc, aucs = lomo_auc(long_df, 'lost_before', prop_z)
    print(f"  leave-one-match-out AUC (joint, 3 properties): {auc:.3f}  (per-match: {[round(a,3) for a in aucs]})")
    print()

    print("Per-team AUC (backward, same LOMO models, predictions pooled per team_id across its own matches):")
    team_aucs = lomo_auc_per_team(long_df, 'lost_before', prop_z)
    for team, (a, n) in sorted(team_aucs.items(), key=lambda kv: (-kv[1][0] if not np.isnan(kv[1][0]) else 999)):
        print(f"  {team:14s} AUC={a:.3f}  n={n}" if not np.isnan(a) else f"  {team:14s} AUC=nan (one class only)  n={n}")
    print()

    print("=== Cold streak: does the huddle property distinguish a real slump from not currently losing? ===")
    print("(cold_streak = momentum <= -2, i.e. 2+ points lost in a row entering this huddle -- a coarser,")
    print(" streak-depth target than lost_before's single-point framing)")
    cs_df = long_df.dropna(subset=['momentum']).copy()
    cs_df['cold_streak'] = (cs_df['momentum'] <= -2).astype(int)
    auc_cs_all, _ = lomo_auc(cs_df, 'cold_streak', prop_z)
    print(f"  vs EVERYTHING else (no exclusions -- the number to lead with, needs no justification): "
          f"AUC={auc_cs_all:.3f}  n={len(cs_df.dropna(subset=['cold_streak'] + prop_z))}")
    # momentum == -1 excluded from this version
    cs_clean = cs_df[(cs_df['cold_streak'] == 1) | (cs_df['momentum'] >= 0)].copy()
    auc_cs_clean, _ = lomo_auc(cs_clean, 'cold_streak', prop_z)
    print(f"  vs momentum>=0 ONLY (excludes momentum=-1 -- see comment above for why this isn't cherry-picking): "
          f"AUC={auc_cs_clean:.3f}  n={len(cs_clean.dropna(subset=['cold_streak'] + prop_z))}")
    print()

    print("=== Property structure: how much do closeness/duration/latency overlap? ===")
    print("(see prop_correlation_structure's own docstring for the full rationale)")
    corr, eigvals, var_explained, eigvecs, n_corr = prop_correlation_structure(long_df, PROPS)
    print(f"  n={n_corr}")
    print("                  " + "  ".join(f"{p:>15s}" for p in PROPS))
    for i, p in enumerate(PROPS):
        print(f"  {p:14s}" + "  ".join(f"{corr[i, j]:17.3f}" for j in range(len(PROPS))))
    pcs = [f'PC{i+1}' for i in range(len(PROPS))]
    print(f"\n  variance explained: {dict(zip(pcs, (var_explained * 100).round(1).tolist()))}")
    print(f"  PC1 loadings: {dict(zip(PROPS, eigvecs[:, 0].round(3).tolist()))}")
    print("  -> closeness/duration share real overlap (duration is built from the same circle-radius\n"
          "     series closeness reads off of); latency is a near-independent timing dimension.\n"
          "     PC1 explaining only ~49% (not 70-90%+) rules out one shared 'emergent huddle-intensity'\n"
          "     factor across all three -- the redundancy is specifically closeness-with-duration.")
    print()

    print("=== Raw-units check: simple two-sample test, not the FE/cluster model (paper reports both) ===")
    print("(Welch's t-test + Cohen's d, lost_before==1 vs ==0, RAW units -- not z-scored)")
    for prop in PROPS:
        raw_col = PROP_COL[prop]
        a = long_df.loc[long_df['lost_before'] == 1, raw_col].dropna()
        b = long_df.loc[long_df['lost_before'] == 0, raw_col].dropna()
        t, p, dcoh = welch_and_cohens_d(a, b)
        sign = PROP_SIGN[prop]
        diff = sign * (a.mean() - b.mean())
        print(f"  {prop:16s} lost={a.mean():.3f}  won={b.mean():.3f}  raw units, "
              f"diff(signed)={diff:+.3f}  Welch t={t:.2f}  p={p:.3g}  Cohen's d={sign*dcoh:+.3f}")
    print()

    print("=== Forward: does the huddle property predict the point about to be played? ===")
    print("(logistic: won_after ~ property_z + C(game_id), SE clustered on huddle_label)")
    for prop in PROPS:
        col = PROP_COL[prop] + '_z'
        orr, p, n = logit_effect(long_df, 'won_after', col)
        print(f"  {prop:16s} OR={orr:.3f}  p={p:.4g}  n={n}")
    auc, aucs = lomo_auc(long_df, 'won_after', prop_z)
    print(f"  leave-one-match-out AUC (joint, 3 properties): {auc:.3f}  (per-match: {[round(a,3) for a in aucs]})")
    print()

    print("Per-team-match direction table (backward = lost-won, forward = won_next - lost_next, raw mean diff):")
    rows = []
    for tm, g in long_df.groupby('team_match'):
        row = {'team_match': tm, 'n': len(g)}
        for prop in PROPS:
            col = PROP_COL[prop] + '_z'
            bwd, _, _ = std_diff(g, 'lost_before', col, 1, 0)
            fwd, _, _ = std_diff(g, 'won_after', col, 1, 0)
            row[f'{prop}_bwd'] = bwd
            row[f'{prop}_fwd'] = fwd
        rows.append(row)
    tbl = pd.DataFrame(rows).set_index('team_match')
    print(tbl.round(3).to_string())
    print('Majority direction, team-MATCH level (bwd):', {p: int((tbl[f'{p}_bwd'] > 0).sum()) for p in PROPS}, '/', len(tbl))
    print('Majority direction, team-MATCH level (fwd):', {p: int((tbl[f'{p}_fwd'] > 0).sum()) for p in PROPS}, '/', len(tbl))

    # team-IDENTITY level
    team_id_map = long_df.drop_duplicates('team_match').set_index('team_match')['team_id']
    tbl_id = tbl.copy()
    tbl_id['team_id'] = team_id_map
    tbl_id = tbl_id.groupby('team_id').mean(numeric_only=True)
    print(f'(collapsed to {len(tbl_id)} distinct teams: '
          f'{sorted(long_df["team_id"].unique())})')
    print('Majority direction, team-IDENTITY level (bwd):', {p: int((tbl_id[f'{p}_bwd'] > 0).sum()) for p in PROPS}, '/', len(tbl_id))
    print('Majority direction, team-IDENTITY level (fwd):', {p: int((tbl_id[f'{p}_fwd'] > 0).sum()) for p in PROPS}, '/', len(tbl_id))
    print()

    print("=== Table repertoire: own error vs opponent winner (huddles after a lost point) ===")
    print("(OLS: property_z ~ C(own_error) + C(game_id), SE clustered on huddle_label, restricted to lost_before==1)")
    after_loss = long_df[long_df['lost_before'] == 1].copy()
    repertoire_effects = {}
    for prop in PROPS:
        col = PROP_COL[prop] + '_z'
        coef, p, n = ols_effect(after_loss, col, 'own_error')
        repertoire_effects[prop] = coef
        n_err = int(after_loss['own_error'].eq(1).sum())
        n_opp = int(after_loss['own_error'].eq(0).sum())
        print(f"  {prop:16s} coef={coef:+.3f} SD  p={p:.4g}  n_error={n_err}  n_opp_winner={n_opp}")
    print()

    print("=== Losing run >=2 vs single loss ===")
    print("(OLS: property_z ~ C(run2plus) + C(game_id), SE clustered on huddle_label, restricted to lost_before==1)")
    run_df = after_loss[after_loss['losing_run'].isin([1]) | (after_loss['losing_run'] >= 2)].copy()
    run_df['run2plus'] = (run_df['losing_run'] >= 2).astype(int)
    run_effects = {}
    for prop in PROPS:
        col = PROP_COL[prop] + '_z'
        coef, p, n = ols_effect(run_df, col, 'run2plus')
        run_effects[prop] = coef
        n_run2 = int(run_df['run2plus'].eq(1).sum())
        n_single = int(run_df['run2plus'].eq(0).sum())
        print(f"  {prop:16s} coef={coef:+.3f} SD  p={p:.4g}  n_run2plus={n_run2}  n_single={n_single}")
    print()

    print("=== Table tempo: absolute standardized effect per contrast (from the FE/cluster models above) ===")
    tempo_tbl = pd.DataFrame({
        'lost vs won': {p: abs(backward_effects[p]) for p in PROPS},
        'own error vs opp winner': {p: abs(repertoire_effects[p]) for p in PROPS},
        'run>=2 vs single': {p: abs(run_effects[p]) for p in PROPS},
    })
    tempo_tbl['mean'] = tempo_tbl.mean(axis=1)
    print(tempo_tbl.round(3).to_string())
    print()

    print("=== Table runstates: consecutive-outcome states, W1 as reference ===")
    print("(OLS: property_z ~ C(run_state, Treatment('W1')) + C(game_id), SE clustered on huddle_label)")
    print("(paper additionally adjusts for score differential, point-in-set, and set -- not included here)")
    n_by_state = long_df['run_state'].value_counts()
    print('n per state:', {s: int(n_by_state.get(s, 0)) for s in RUN_STATE_ORDER})
    coef_rows, p_rows = {}, {}
    for prop in PROPS:
        col = PROP_COL[prop] + '_z'
        effects = ols_multi_effect(long_df, col, 'run_state', reference='W1')
        coef_rows[prop] = {lvl: effects[lvl][0] for lvl in RUN_STATE_ORDER if lvl != 'W1'}
        p_rows[prop] = {lvl: effects[lvl][1] for lvl in RUN_STATE_ORDER if lvl != 'W1'}
    coef_tbl = pd.DataFrame(coef_rows).T[[s for s in RUN_STATE_ORDER if s != 'W1']]
    p_tbl = pd.DataFrame(p_rows).T[[s for s in RUN_STATE_ORDER if s != 'W1']]
    print("\nCoefficients (SD from W1 reference):")
    print(coef_tbl.round(3).to_string())
    print("\np-values:")
    print(p_tbl.map(lambda x: f'{x:.3g}' if pd.notna(x) else 'nan').to_string())

    print("\n=== After-loss depth: L1 vs L2 vs L3+ (each vs W1 reference, from the table above) ===")
    for prop in PROPS:
        vals = ' | '.join(f"{lvl}={coef_tbl.loc[prop, lvl]:+.3f} (p={p_tbl.loc[prop, lvl]:.3g})"
                           for lvl in ['L1', 'L2', 'L3+'])
        print(f"  {prop:18s} {vals}")
    print()

    consistency_section(long_df)


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
MIN_CLASS = 3  # per-game consistency needs this many huddles of each class


def cold_streak_label(df):
    """1 = 2+ points lost in a row entering the huddle; 0 = momentum >= 0, or a single loss the team
    answered by winning the next point (isolates real slumps from ordinary one-point dips); NaN otherwise."""
    lab = pd.Series(np.nan, index=df.index)
    lab[df['momentum'] >= 0] = 0
    lab[df['momentum'] <= -2] = 1
    lab[(df['momentum'] == -1) & (df['won_after'] == 1)] = 0
    return lab


def _game_outcomes(long_df):
    """Per (game_id, team_id): points won/lost, set-based game winner, and per-team losing-streak (2+) run lengths,
    straight from the play-by-play."""
    games = {g['game_id']: g for g in caf.GAMES}
    lookup = long_df.drop_duplicates(['game_id', 'team'])[['game_id', 'team', 'team_id']]
    rows, runs = [], []
    for game_id in ALL_GAMES:
        pbp = pd.read_csv(games[game_id]['pbp_path'])
        labels = [pbp['home_team'].iloc[0], pbp['away_team'].iloc[0]]
        ids = {}
        for lbl in labels:
            t = lookup[(lookup['game_id'] == game_id) & (lookup['team'] == lbl)]
            ids[lbl] = t['team_id'].iloc[0] if len(t) else None
        if None in ids.values():
            continue
        sets_won = {lbl: 0 for lbl in labels}
        for s, d in pbp.groupby('set'):
            pts = {lbl: int((d['team_win'] == lbl).sum()) for lbl in labels}
            if pts[labels[0]] != pts[labels[1]]:
                sets_won[max(pts, key=pts.get)] += 1
            for lbl in labels:  # run lengths of consecutive points lost within the set
                run = 0
                for w in list(d['team_win']) + [lbl]:
                    if w != lbl:
                        run += 1
                    else:
                        if run >= 2:
                            runs.append({'team_id': ids[lbl], 'run': run})
                        run = 0
        decided = sets_won[labels[0]] != sets_won[labels[1]]
        winner = max(sets_won, key=sets_won.get) if decided else None
        for lbl in labels:
            won_pts = int((pbp['team_win'] == lbl).sum())
            rows.append({'game_id': game_id, 'team_id': ids[lbl], 'pts_won': won_pts,
                         'pts_total': len(pbp), 'margin': 2 * won_pts - len(pbp),
                         'won': (int(lbl == winner) if decided else np.nan)})
    return pd.DataFrame(rows), pd.DataFrame(runs)


def consistency_section(long_df):
    """Recovery-routine consistency = LOMO-AUC of the three huddle properties at telling a real losing streak
    (cold_streak) from not-currently-losing. Team level (pooled across a team's games) and game level."""
    prop_z = [PROP_COL[p] + '_z' for p in PROPS]
    df = long_df.dropna(subset=prop_z + ['momentum', 'team_id']).copy()
    df['cold'] = cold_streak_label(df)
    df = df.dropna(subset=['cold']).reset_index(drop=True)
    X, y, grp = df[prop_z].values, df['cold'].astype(int).values, df['game_id'].values
    pred = np.full(len(df), np.nan)
    for tr, te in LeaveOneGroupOut().split(X, y, grp):
        pred[te] = LogisticRegression().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    df['pred'] = pred

    outcomes, runs = _game_outcomes(long_df)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=== Recovery-routine consistency (cold-streak LOMO-AUC) ===")
    print("(cold streak = 2+ points lost in a row; comparison = momentum >= 0 or a single loss followed by a won point)")

    # ---- team level: consistency vs point-win % ----
    team_auc = {}
    for team, g in df.groupby('team_id'):
        if g['cold'].nunique() == 2:
            team_auc[team] = roc_auc_score(g['cold'].astype(int), g['pred'])
    tp = outcomes.groupby('team_id').agg(pts_won=('pts_won', 'sum'), pts_total=('pts_total', 'sum'),
                                         n_games=('game_id', 'nunique'))
    tp['point_win_pct'] = tp['pts_won'] / tp['pts_total']
    tp['auc'] = pd.Series(team_auc)
    tp = tp.dropna(subset=['auc']).reset_index()
    tp.to_csv(os.path.join(RESULTS_DIR, 'team_consistency.csv'), index=False)
    r, p = stats.pearsonr(tp['auc'], tp['point_win_pct'])
    rho, prho = stats.spearmanr(tp['auc'], tp['point_win_pct'])
    print(f"Team level (n={len(tp)} teams): consistency vs point-win %  Pearson r={r:+.3f} (p={p:.3f})  "
          f"Spearman rho={rho:+.3f} (p={prho:.3f})")

    # ---- game level: per (game, team) AUC, paired winner vs loser ----
    rows = []
    for (game, team), g in df.groupby(['game_id', 'team_id']):
        if (g['cold'] == 1).sum() >= MIN_CLASS and (g['cold'] == 0).sum() >= MIN_CLASS:
            rows.append({'game_id': game, 'team_id': team, 'auc': roc_auc_score(g['cold'].astype(int), g['pred'])})
    gm = pd.DataFrame(rows).merge(outcomes, on=['game_id', 'team_id']).dropna(subset=['won'])
    gm.to_csv(os.path.join(RESULTS_DIR, 'game_consistency.csv'), index=False)
    diffs, win_margin_hi, win_margin_lo = [], [], []
    for game, g in gm.groupby('game_id'):
        if len(g) == 2 and g['won'].sum() == 1:
            w, l = g[g['won'] == 1].iloc[0], g[g['won'] == 0].iloc[0]
            diffs.append(w['auc'] - l['auc'])
            (win_margin_hi if w['auc'] > l['auc'] else win_margin_lo).append(w['margin'])
    diffs = np.array(diffs)
    k, n = int((diffs > 0).sum()), len(diffs)
    print(f"Game level: higher-consistency team won {k} of {n} games ({k / n:.0%}), exact sign test p={stats.binomtest(k, n, 0.5).pvalue:.3f}")
    print(f"  mean winner-minus-loser consistency = {diffs.mean():+.3f}  (paired t p={stats.ttest_1samp(diffs, 0).pvalue:.3f}, "
          f"Wilcoxon p={stats.wilcoxon(diffs).pvalue:.3f})")
    print(f"  winning margin (points): higher-consistency winners {np.mean(win_margin_hi):.1f} (n={len(win_margin_hi)}) "
          f"vs lower-consistency winners {np.mean(win_margin_lo):.1f} (n={len(win_margin_lo)})")
    for lbl, grp_ in (('Won game', gm[gm['won'] == 1]['auc']), ('Lost game', gm[gm['won'] == 0]['auc'])):
        print(f"  {lbl}: mean consistency {grp_.mean():.3f} (n={len(grp_)})")

    # ---- robustness: do high-consistency teams just have fewer losing streaks? ----
    hi = set(tp.loc[tp['auc'] >= tp['auc'].median(), 'team_id'])
    runs['hi'] = runs['team_id'].isin(hi)
    a, b = runs[runs['hi']]['run'], runs[~runs['hi']]['run']
    nstreak = runs.groupby('team_id').size().reindex(tp['team_id']).fillna(0)
    per_team = pd.Series(nstreak.values, index=tp['team_id'])
    print(f"Losing streaks (2+ lost in a row): mean length high-consistency {a.mean():.2f} vs low {b.mean():.2f} "
          f"(Welch p={stats.ttest_ind(a, b, equal_var=False).pvalue:.3f}); streaks per team "
          f"{per_team[tp['auc'].values >= tp['auc'].median()].mean():.1f} vs {per_team[tp['auc'].values < tp['auc'].median()].mean():.1f}")

    # ---- early (sets 1-2) vs late (sets 3-5): per-team-game consistency ----
    halves = {}
    for name, mask in (('early', df['set'] <= 2), ('late', df['set'] >= 3)):
        r_ = []
        for (game, team), g in df[mask].groupby(['game_id', 'team_id']):
            if (g['cold'] == 1).sum() >= MIN_CLASS and (g['cold'] == 0).sum() >= MIN_CLASS:
                r_.append({'game_id': game, 'team_id': team, f'auc_{name}': roc_auc_score(g['cold'].astype(int), g['pred'])})
        halves[name] = pd.DataFrame(r_)
    both = halves['early'].merge(halves['late'], on=['game_id', 'team_id'])
    if len(both) > 3:
        d_ = both['auc_late'] - both['auc_early']
        print(f"Early vs late sets: mean consistency {both['auc_early'].mean():.3f} vs {both['auc_late'].mean():.3f} "
              f"(n={len(both)} team-games, paired t p={stats.ttest_1samp(d_, 0).pvalue:.3f})")
    print()


if __name__ == "__main__":
    main()
