"""Per-game huddle features: joins SAM3 tracking with play-by-play and computes
closeness, duration, latency, momentum and point outcomes.

GAMES is built from stimulus_map.csv, parameters.csv, and each matchup's
play_by_play.csv and all_sets.csv; matchups missing an input are skipped.
"""

import argparse
import ast
import math
import os
import sys

import pandas as pd
import numpy as np

# stimulus_map.py lives in this same directory (analysis/feature_extraction/).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'feature_extraction'))
import stimulus_map

REPO_ROOT = stimulus_map.REPO_ROOT
DEFAULT_PARAMETERS_CSV = REPO_ROOT / "analysis" / "modeling" / "parameters.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "analysis" / "modeling" / "merged-data"
# Where sam3_automation.py's per-matchup all_sets.csv lands (data/<matchup>/all_sets.csv
# in this repo) - kept separate from COMPUTER_VISION_DRIVE_ROOT, which is where the
# much larger raw video clips live (Quest lab storage, Drive, etc.) and has no
# bearing on where this small tracking-output CSV should be read from.
DEFAULT_DRIVE_ROOT = str(REPO_ROOT / "data")

# sam3_automation.py's homography calibrates directly off the real court in meters (COURT_W, COURT_H =
# 18.0, 9.0 there), so every points_str column it writes is in meters.
METERS_TO_FEET = 3.280839895013123
COURT_X_FT = 18.0 * METERS_TO_FEET
COURT_Y_FT = 9.0 * METERS_TO_FEET


def _bool(value) -> bool:
    return str(value).strip().upper() == "TRUE"


# parameters.csv's own home_team column is unreliable (data-entry errors --
# e.g. it names the road team as home for both TexasLutheran games and for
# Ithaca_v_NYU, confirmed by cross-checking against left_team_start, which
# is trustworthy). home_start_side is computed against the REAL home team
# (play_by_play's home_team, the one team_win etc. actually use) instead,
# resolved through this abbreviation table since play_by_play uses each
# site's own box-score abbreviation (e.g. 'eu') while parameters.csv spells
# names out (e.g. 'Emory'). Derived by cross-checking every row where the
# two sources already agreed; add an entry here (not to parameters.csv) if
# a new game's box-score abbreviation isn't recognized.
HOME_TEAM_ABBREV_TO_FULL = {
    'adrian': 'adrian', 'bc': 'bridgewater', 'bluffton': 'bluffton', 'eu': 'emory',
    'hope': 'hope', 'ith': 'ithaca', 'mcla': 'mcla', 'msj': 'msj', 'nyu': 'nyu',
    'smi': 'smith', 'swat': 'swarthmore', 'trintx': 'trinity', 'ulv': 'laverne',
    'tlu': 'texas lutheran', 'uni': 'union',
}


def build_games(stimulus_map_path=stimulus_map.DEFAULT_STIMULUS_MAP_PATH,
                 parameters_csv=DEFAULT_PARAMETERS_CSV,
                 drive_root=DEFAULT_DRIVE_ROOT):
    """One config dict per matchup, joining stimulus_map.csv, parameters.csv and the scraped play-by-play."""
    rows = stimulus_map.load_rows(stimulus_map_path)
    rows = [r for r in rows if r.get("type", "").strip() == "huddle_video"]
    matchups = sorted(stimulus_map.group_by_matchup(rows))

    params = pd.read_csv(parameters_csv)
    params.columns = [c.strip() for c in params.columns]

    games = []
    for matchup in matchups:
        pbp_path = os.path.join(drive_root, matchup, "play_by_play.csv")
        huddle_csv = os.path.join(drive_root, matchup, "all_sets.csv")

        if not os.path.isfile(pbp_path):
            print(f"[SKIP] {matchup}: no play-by-play at {pbp_path} (run build_dataset.py first)")
            continue
        if not os.path.isfile(huddle_csv):
            print(f"[SKIP] {matchup}: no huddle tracking data at {huddle_csv} (run sam3_automation.py first)")
            continue

        pbp_head = pd.read_csv(pbp_path, nrows=1)
        home_team, away_team = pbp_head["home_team"].iloc[0], pbp_head["away_team"].iloc[0]

        matched_game_id = stimulus_map.find_by_team_tokens(matchup, list(params["game_id"]))
        prow = params.loc[params["game_id"] == matched_game_id].iloc[0]

        switches_sides = _bool(prow["teams_switch"])
        left_start_team = str(prow["left_team_start"]).strip().lower()
        real_home_team = str(home_team).strip().lower()
        real_home_full = HOME_TEAM_ABBREV_TO_FULL.get(real_home_team, real_home_team)
        home_start_side = "left" if left_start_team == real_home_full else "right"

        games.append({
            'game_id': matchup,
            'pbp_path': pbp_path,
            'huddle_csv': huddle_csv,
            'home_team': home_team,
            'away_team': away_team,
            'switches_sides': switches_sides,
            'home_start_side': home_start_side,
        })

    return games


GAMES = build_games()

# kept for backwards compatibility with robustness_checks.py's caf.pbp_data
pbp_data = pd.read_csv(GAMES[0]['pbp_path']) if GAMES else None


def load_huddle_video_data(huddle_csv):
    '''Loads the combined all_sets.csv written by sam3_automation.py.'''
    return pd.read_csv(huddle_csv)


def _huddle_label(huddle_num, set_num):
    return f"huddle_{int(huddle_num)}_set_{int(set_num)}"


def huddles_bf_af(df):
    """Add huddle_before / huddle_after labels for the huddles around each point."""
    df = df.sort_values(['set', 'point_num']).reset_index(drop=True)

    def huddle_label(h, s):
        return _huddle_label(h, s) if pd.notna(h) and h > 0 else np.nan

    df['huddle_before'] = [huddle_label(h, s) for h, s in zip(df['huddle'], df['set'])]

    next_huddle = df.groupby('set')['huddle'].shift(-1)
    df['huddle_after'] = [
        huddle_label(nh, s) if pd.notna(nh) and nh > h else np.nan
        for h, nh, s in zip(df['huddle'], next_huddle, df['set'])
    ]

    return df


def label_video_huddles(df):
    """Label each tracking frame with its huddle."""
    df['huddle_label'] = [
        _huddle_label(h, s) for h, s in zip(df['huddle_number'], df['set'])
    ]
    return df


def _parse_points_ft(points_str):
    """Parse a stringified list of [x, y] court points into feet."""
    return [(x * METERS_TO_FEET, y * METERS_TO_FEET) for x, y in ast.literal_eval(points_str)]


def _has_off_court_point(points_str, margin, x_bounds=(0, COURT_X_FT), y_bounds=(0, COURT_Y_FT)):
    """True if any tracked point in the frame lies off the court beyond a margin."""
    x_lo, x_hi = x_bounds[0] - margin, x_bounds[1] + margin
    y_lo, y_hi = y_bounds[0] - margin, y_bounds[1] + margin
    return any(x < x_lo or x > x_hi or y < y_lo or y > y_hi for x, y in _parse_points_ft(points_str))


def _is_degenerate_tracking(points_str, y_bounds=(0, COURT_Y_FT), near=0.5 * METERS_TO_FEET, collinear_std=0.3 * METERS_TO_FEET):
    """True if a frame's points collapsed onto a flat line along a court boundary."""
    pts = _parse_points_ft(points_str)
    if len(pts) < 3:
        return False
    ys = np.array([p[1] for p in pts])
    near_bound = any(abs(ys.mean() - b) < near for b in y_bounds)
    return bool(ys.std() < collinear_std and near_bound)


def _is_elongated_formation(points, long_std_threshold=1.0 * METERS_TO_FEET, ratio_threshold=4.0):
    """True if the points form a thin line rather than a cluster."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return False
    centered = pts - pts.mean(axis=0)
    cov = np.cov(centered.T)
    eigvals = np.linalg.eigvalsh(cov)
    short_std = math.sqrt(max(eigvals[0], 0))
    long_std = math.sqrt(max(eigvals[-1], 0))
    ratio = long_std / short_std if short_std > 1e-6 else math.inf
    return long_std > long_std_threshold and ratio > ratio_threshold


def _qualifying_span_duration_s(frames, values, threshold, fps, max_gap_s=0.1):
    """Seconds spanned by the longest episode of qualifying frames."""
    frames = np.asarray(frames)
    qualifies = np.asarray(values) <= threshold
    episode = _longest_qualifying_episode(frames, qualifies, max_gap_frames=max(1, round(max_gap_s * fps)))
    if episode is None:
        return 0.0
    lo, hi = episode
    return (frames[hi] - frames[lo]) / fps


def _qualifying_episodes(frames, qualifies, max_gap_frames):
    """Group qualifying frames into episodes, merging runs separated by a short gap."""
    frames = np.asarray(frames)
    idx = np.nonzero(np.asarray(qualifies))[0]
    if len(idx) == 0:
        return []
    episodes = []
    lo = hi = idx[0]
    for i in idx[1:]:
        gap_frames = frames[i] - frames[hi] - 1
        if gap_frames <= max_gap_frames:
            hi = i
        else:
            episodes.append((int(lo), int(hi)))
            lo = hi = i
    episodes.append((int(lo), int(hi)))
    return episodes


def _longest_qualifying_episode(frames, qualifies, max_gap_frames):
    """The (lo, hi) indices of the episode spanning the most time."""
    episodes = _qualifying_episodes(frames, qualifies, max_gap_frames)
    if not episodes:
        return None
    frames = np.asarray(frames)
    return max(episodes, key=lambda e: frames[e[1]] - frames[e[0]])


def _mad_trim_points(points, k=2.5, min_points=4):
    """Drop points far from the median center (k * 1.4826 * MAD)."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < min_points:
        return points
    center = np.median(pts, axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    med = np.median(d)
    mad = np.median(np.abs(d - med)) + 1e-6
    keep = d < med + k * 1.4826 * mad
    trimmed = pts[keep]
    return [tuple(p) for p in trimmed] if len(trimmed) >= min_points else points


def _convex_hull_2d(points):
    """Convex hull vertices (Andrew's monotone chain), counter-clockwise."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _hull_area_from_points(points):
    """Convex-hull area via the shoelace formula; NaN under 3 points."""
    hull = _convex_hull_2d([tuple(p) for p in points])
    if len(hull) < 3:
        return np.nan
    area = 0.0
    n = len(hull)
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _gyration_from_points(points):
    p = np.asarray(points, dtype=float)
    if len(p) == 0:
        return np.nan
    c = p.mean(axis=0)
    return float(np.sqrt(((p - c) ** 2).sum(axis=1).mean()))


def _min_enclosing_circle(points):
    """Radius of the smallest circle enclosing the points (Welzl)."""
    import random as _random
    pts = [tuple(map(float, p)) for p in points]
    if len(pts) == 0:
        return np.nan

    def circle_from_2(a, b):
        cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
        return (cx, cy), math.dist(a, b) / 2

    def circle_from_3(a, b, c):
        ax, ay = a; bx, by = b; cx, cy = c
        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-12:
            best = max([(a, b), (a, c), (b, c)], key=lambda p: math.dist(*p))
            return circle_from_2(*best)
        ux = ((ax**2 + ay**2) * (by - cy) + (bx**2 + by**2) * (cy - ay) + (cx**2 + cy**2) * (ay - by)) / d
        uy = ((ax**2 + ay**2) * (cx - bx) + (bx**2 + by**2) * (ax - cx) + (cx**2 + cy**2) * (bx - ax)) / d
        center = (ux, uy)
        return center, math.dist(center, a)

    def in_circle(p, center, r, eps=1e-7):
        return math.dist(p, center) <= r + eps

    shuffled = pts[:]
    _random.Random(0).shuffle(shuffled)

    center, r = shuffled[0], 0.0
    for i, p in enumerate(shuffled):
        if in_circle(p, center, r):
            continue
        center, r = p, 0.0
        for j, q in enumerate(shuffled[:i]):
            if in_circle(q, center, r):
                continue
            center, r = circle_from_2(p, q)
            for s in shuffled[:j]:
                if in_circle(s, center, r):
                    continue
                center, r = circle_from_3(p, q, s)
    return r


def _frame_outlier_point(points, k_outlier=3.0, min_distance_ft=1.0 * METERS_TO_FEET):
    """The point farthest from the rest, if it qualifies as a peripheral player."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return None, None
    excess = []
    for i in range(len(pts)):
        rest = np.delete(pts, i, axis=0)
        c = rest.mean(axis=0)
        excess.append(np.linalg.norm(pts[i] - c))
    i_max = int(np.argmax(excess))
    rest = np.delete(pts, i_max, axis=0)
    c_rest = rest.mean(axis=0)
    rest_d = np.linalg.norm(rest - c_rest, axis=1)
    med_rest = np.median(rest_d) if len(rest_d) else 0.0
    if excess[i_max] >= max(k_outlier * max(med_rest, 1e-6), min_distance_ft):
        return tuple(pts[i_max]), float(excess[i_max])
    return None, None


def _near_court_boundary(point, x_bounds=(0, COURT_X_FT), y_bounds=(0, COURT_Y_FT), margin=0.5 * METERS_TO_FEET):
    """True if a point is within `margin` feet of a court boundary line."""
    x, y = point
    return (x < x_bounds[0] + margin or x > x_bounds[1] - margin or
            y < y_bounds[0] + margin or y > y_bounds[1] - margin)


def _peripheral_player_episode(frame_points, k_outlier=3.0, min_distance_ft=1.0 * METERS_TO_FEET,
                                boundary_margin=0.5 * METERS_TO_FEET,
                                continuity_ft=0.25 * METERS_TO_FEET, max_spread_ft=3.0 * METERS_TO_FEET):
    """Longest run of consecutive frames with a peripheral player."""
    best_len, best_dists = 0, []
    run_len, run_dists, prev_point, run_start = 0, [], None, None
    for points in frame_points:
        point, dist = _frame_outlier_point(points, k_outlier, min_distance_ft)
        if point is not None and _near_court_boundary(point, margin=boundary_margin):
            point = None
        broken = (
            point is None
            or (prev_point is not None and math.dist(point, prev_point) > continuity_ft)
            or (run_start is not None and math.dist(point, run_start) > max_spread_ft)
        )
        if broken:
            run_len, run_dists, prev_point, run_start = 0, [], None, None
            if point is None:
                continue
        run_len += 1
        run_dists.append(dist)
        prev_point = point
        if run_start is None:
            run_start = point
        if run_len > best_len:
            best_len, best_dists = run_len, list(run_dists)
    return best_len, (float(np.mean(best_dists)) if best_dists else np.nan)


def _home_side_for_set(set_num, switches_sides, home_start_side):
    '''Which physical side ("left"/"right") the home team occupies for this set.'''
    if not switches_sides:
        return home_start_side
    other_side = 'right' if home_start_side == 'left' else 'left'
    return home_start_side if set_num % 2 == 1 else other_side


# NCAA rule: sets 1-4 are played to 25 and teams only change ends between
# sets; the 5th (deciding) set is played to 15 and teams ALSO change ends
# mid-set, the moment either team's score first reaches 8. A match only
# ever reaches a 5th set when one is needed to decide it, so this rule is
# tied to set number 5 specifically, never "however many sets this match's
# own parameters.csv row says it went" - a match parameters.csv marks as
# 4 sets never triggers it, since it never played a 5th at all.
DECIDING_SET_NUM = 5
DECIDING_SET_SWITCH_SCORE = 8


def _deciding_set_switch_huddle(pbp_df):
    """The set-5 huddle number after which teams switch ends."""
    s5 = pbp_df[pbp_df['set'] == DECIDING_SET_NUM].sort_values('point_num')
    if s5.empty:
        return None
    reached = s5[(s5['home_score'] >= DECIDING_SET_SWITCH_SCORE) |
                 (s5['visitor_score'] >= DECIDING_SET_SWITCH_SCORE)]
    if reached.empty:
        return None
    return int(reached.iloc[0]['huddle'])


def add_home_away_columns(df, cfg, pbp_df=None):
    """Rename left_/right_ columns to home_/visitor_."""
    df = df.copy()
    unique_sets = df['set'].astype(int).unique()
    side_map = {
        s: _home_side_for_set(s, cfg['switches_sides'], cfg['home_start_side'])
        for s in unique_sets
    }
    home_side = df['set'].astype(int).map(side_map)

    if pbp_df is not None and cfg['switches_sides'] and DECIDING_SET_NUM in unique_sets:
        switch_after = _deciding_set_switch_huddle(pbp_df)
        if switch_after is not None:
            flip = (df['set'].astype(int) == DECIDING_SET_NUM) & (df['huddle_number'].astype(int) > switch_after)
            opposite_side = home_side.map({'left': 'right', 'right': 'left'})
            home_side = home_side.where(~flip, opposite_side)

    is_home_left = (home_side == 'left').to_numpy()

    bases = sorted({
        col[len('left_'):] for col in df.columns if col.startswith('left_')
    } | {
        col[len('right_'):] for col in df.columns if col.startswith('right_')
    })

    for base in bases:
        left_col, right_col = f'left_{base}', f'right_{base}'
        left_vals = df[left_col] if left_col in df.columns else pd.Series(np.nan, index=df.index)
        right_vals = df[right_col] if right_col in df.columns else pd.Series(np.nan, index=df.index)
        df[f'home_{base}'] = np.where(is_home_left, left_vals, right_vals)
        df[f'away_{base}'] = np.where(is_home_left, right_vals, left_vals)

    drop_cols = [c for c in df.columns if c.startswith('left_') or c.startswith('right_')]
    return df.drop(columns=drop_cols)


def huddle_formation_features(df, min_n=4, margin=None, first_n_frames=5,
                               sides=('home', 'away'), suffix=''):
    """Collapse per-frame tracking to one row per huddle with closeness, duration and latency features."""
    rows = []
    for (s, h), group in df.sort_values(['set', 'huddle_number', 'frame']).groupby(['set', 'huddle_number'], sort=False):
        group = group.reset_index(drop=True)
        fps = group['fps'].iloc[0]
        rec = {'set': s, 'huddle_number': h, 'huddle_label': _huddle_label(h, s)}

        for side in sides:
            ncol = f'{side}_n{suffix}'
            ptscol = f'{side}_points{suffix}'
            blobcol = f'{side}_blob_hull_px{suffix}'

            usable = group[ncol] >= min_n
            too_few_points = pd.Series(False, index=group.index)
            if usable.any():
                too_few_points.loc[usable] = group.loc[usable, ptscol].apply(
                    lambda p: len(ast.literal_eval(p)) < min_n)
            usable = usable & ~too_few_points
            degenerate = pd.Series(False, index=group.index)
            if usable.any():
                degenerate.loc[usable] = group.loc[usable, ptscol].apply(_is_degenerate_tracking).astype(bool)
            usable = usable & ~degenerate
            if margin is not None:
                off_court = pd.Series(False, index=group.index)
                if usable.any():
                    off_court.loc[usable] = group.loc[usable, ptscol].apply(
                        lambda p: _has_off_court_point(p, margin)).astype(bool)
                usable = usable & ~off_court
            side_group = group[usable].reset_index(drop=True)

            feature_names = (
                f'{side}_circle_diameter_ft{suffix}',
                f'{side}_huddle_closeness_ft2{suffix}', f'{side}_huddle_gyration_closeness_ft{suffix}',
                f'{side}_blob_hull_at_min_closeness{suffix}', f'{side}_huddle_duration_s{suffix}',
                f'{side}_convergence_rate_ft2{suffix}', f'{side}_convergence_travel_mean_ft{suffix}',
                f'{side}_convergence_travel_std_ft{suffix}', f'{side}_peripheral_player_flag{suffix}',
                f'{side}_peripheral_player_frames{suffix}', f'{side}_peripheral_player_distance_ft{suffix}',
            )
            if len(side_group) == 0:
                for name in feature_names:
                    rec[name] = np.nan
                continue

            raw_points = side_group[ptscol].apply(_parse_points_ft)
            trimmed_points = raw_points.apply(lambda p: _mad_trim_points(p, min_points=min_n))

            hull = trimmed_points.apply(_hull_area_from_points)
            elongated = trimmed_points.apply(_is_elongated_formation)
            hull = hull.mask(elongated)
            gyration = trimmed_points.apply(_gyration_from_points)

            if hull.notna().sum() == 0:
                for name in feature_names:
                    rec[name] = np.nan
                continue

            # huddle_closeness is reported off a 5-frame centered rolling mean of hull,
            # not the raw per-frame min - a single noisy/occluded frame can otherwise drag
            # the reported minimum down artificially. Confirmed empirically: smoothing
            # raises the metric's own correlation with next5_win_rate (naive r -0.109 ->
            # -0.160, within-R^2 1.18% -> 2.57%) without touching everything else keyed off
            # the RAW min (min_hull_idx below) - gyration_closeness, the tight-episode
            # threshold, convergence_rate, travel, and peripheral-player detection all stay
            # anchored to the single actual tightest frame, unchanged from before.
            smoothed_hull = hull.rolling(window=5, center=True, min_periods=1).mean()
            closeness_smoothed = smoothed_hull.min()

            min_hull = hull.min()
            min_hull_idx = hull.idxmin()
            min_gyration = gyration.loc[min_hull_idx]
            blob_at_min = side_group.loc[min_hull_idx, blobcol] if blobcol in side_group.columns else np.nan

            threshold = (1 + 0.5) * min_hull

            # Duration and the peripheral-player window use a SEPARATE, more permissive n>=3 pass rather
            # than reusing the strict side_group above.
            usable_relaxed = group[ncol] >= 3
            too_few_relaxed = pd.Series(False, index=group.index)
            if usable_relaxed.any():
                too_few_relaxed.loc[usable_relaxed] = group.loc[usable_relaxed, ptscol].apply(
                    lambda p: len(ast.literal_eval(p)) < 3)
            usable_relaxed = usable_relaxed & ~too_few_relaxed
            degenerate_relaxed = pd.Series(False, index=group.index)
            if usable_relaxed.any():
                degenerate_relaxed.loc[usable_relaxed] = group.loc[usable_relaxed, ptscol].apply(
                    _is_degenerate_tracking).astype(bool)
            usable_relaxed = usable_relaxed & ~degenerate_relaxed
            side_group_relaxed = group[usable_relaxed].reset_index(drop=True)
            raw_points_relaxed = side_group_relaxed[ptscol].apply(_parse_points_ft)
            trimmed_relaxed = raw_points_relaxed.apply(lambda p: _mad_trim_points(p, min_points=3))
            hull_relaxed = trimmed_relaxed.apply(_hull_area_from_points)
            elongated_relaxed = trimmed_relaxed.apply(_is_elongated_formation)
            hull_relaxed = hull_relaxed.mask(elongated_relaxed)

            # circle_diameter (THE primary closeness measure - see docstring) gets its
            # own independently-found tightest frame from this same relaxed n>=3 pool,
            # not min_hull_idx above - that's deliberate, see docstring for why.
            if hull_relaxed.notna().sum() == 0:
                circle_diameter = np.nan
            else:
                min_hull_idx_relaxed = hull_relaxed.idxmin()
                circle_diameter = 2 * _min_enclosing_circle(trimmed_relaxed.loc[min_hull_idx_relaxed])

            duration_s = _qualifying_span_duration_s(side_group_relaxed['frame'], hull_relaxed, threshold, fps)

            firstN_mean = hull.iloc[:first_n_frames].mean()
            min_hull_frame = side_group.loc[min_hull_idx, 'frame']
            first_frame = side_group['frame'].iloc[0]
            seconds_to_min = (min_hull_frame - first_frame) / fps
            convergence_rate = (firstN_mean - min_hull) / seconds_to_min if seconds_to_min > 0 else np.nan

            final_center = np.mean(trimmed_points.loc[min_hull_idx], axis=0)
            travel_dists = [math.dist(p, final_center) for p in raw_points.iloc[0]]
            travel_mean = float(np.mean(travel_dists)) if travel_dists else np.nan
            travel_std = float(np.std(travel_dists)) if travel_dists else np.nan

            # Peripheral-player detection stays on the STRICT n>=min_n series, deliberately NOT the
            # relaxed n>=3 one duration uses above.
            max_gap_frames = max(1, round(0.1 * fps))
            qualifies_strict = (hull <= threshold).to_numpy()
            strict_episode = _longest_qualifying_episode(side_group['frame'].to_numpy(), qualifies_strict, max_gap_frames)
            window_points = raw_points.iloc[strict_episode[0]:strict_episode[1] + 1] if strict_episode is not None else raw_points.iloc[:0]
            if len(window_points) >= 3:
                peripheral_frames, peripheral_dist = _peripheral_player_episode(window_points)
                peripheral_flag = float(peripheral_frames >= 6)
            else:
                peripheral_flag, peripheral_frames, peripheral_dist = np.nan, 0, np.nan

            rec[f'{side}_circle_diameter_ft{suffix}'] = circle_diameter
            rec[f'{side}_huddle_closeness_ft2{suffix}'] = closeness_smoothed
            rec[f'{side}_huddle_gyration_closeness_ft{suffix}'] = min_gyration
            rec[f'{side}_blob_hull_at_min_closeness{suffix}'] = blob_at_min
            rec[f'{side}_huddle_duration_s{suffix}'] = duration_s
            rec[f'{side}_convergence_rate_ft2{suffix}'] = convergence_rate
            rec[f'{side}_convergence_travel_mean_ft{suffix}'] = travel_mean
            rec[f'{side}_convergence_travel_std_ft{suffix}'] = travel_std
            rec[f'{side}_peripheral_player_flag{suffix}'] = peripheral_flag
            rec[f'{side}_peripheral_player_frames{suffix}'] = peripheral_frames
            rec[f'{side}_peripheral_player_distance_ft{suffix}'] = peripheral_dist

        rows.append(rec)

    return pd.DataFrame(rows)


def huddle_win_flags(huddle_video_df, pbp_df, labels):
    """Per-team 0/1 flags for whether each huddle was followed by a won point."""
    prev = (
        pbp_df[['huddle_after', 'team_win']]
        .dropna(subset=['huddle_after'])
        .drop_duplicates(subset='huddle_after')
        .rename(columns={'huddle_after': 'huddle_label', 'team_win': 'prev_team_win'})
    )
    nxt = (
        pbp_df[['huddle_before', 'team_win']]
        .dropna(subset=['huddle_before'])
        .drop_duplicates(subset='huddle_before')
        .rename(columns={'huddle_before': 'huddle_label', 'team_win': 'next_team_win'})
    )

    huddles = huddle_video_df[['set', 'huddle_number']].drop_duplicates().copy()
    huddles['huddle_label'] = [_huddle_label(h, s) for h, s in zip(huddles['huddle_number'], huddles['set'])]
    huddles = huddles.merge(prev, on='huddle_label', how='left').merge(nxt, on='huddle_label', how='left')

    last_point_per_set = pbp_df.sort_values('point_num').groupby('set').tail(1).set_index('set')

    def infer_prev_winner(row):
        if pd.notna(row['prev_team_win']):
            return row['prev_team_win']
        last = last_point_per_set.loc[row['set']]
        if row['huddle_number'] == last['huddle'] + 1:
            return last['team_win']
        return np.nan

    huddles['prev_team_win'] = huddles.apply(infer_prev_winner, axis=1)

    for team_var, team_label in labels.items():
        huddles[f'{team_var}_won_before'] = (huddles['prev_team_win'] == team_label).astype('Int64')
        huddles.loc[huddles['prev_team_win'].isna(), f'{team_var}_won_before'] = pd.NA
        huddles[f'{team_var}_won_after'] = (huddles['next_team_win'] == team_label).astype('Int64')
        huddles.loc[huddles['next_team_win'].isna(), f'{team_var}_won_after'] = pd.NA

    flag_cols = [f'{team_var}_won_before' for team_var in labels] + [f'{team_var}_won_after' for team_var in labels]
    return huddles[['set', 'huddle_number', 'huddle_label'] + flag_cols]


def add_huddle_occurred_flag(df, sides=('home', 'away'), suffix='', percentile=75):
    """Flag whether a huddle actually formed (closeness under a threshold)."""
    df = df.copy()
    for side in sides:
        size_col = f'{side}_huddle_closeness_ft2{suffix}'
        flag_col = f'{side}_huddle_occurred{suffix}'
        valid = df[size_col].dropna()
        if len(valid) == 0:
            df[flag_col] = np.nan
            continue
        threshold = np.percentile(valid, percentile)
        df[flag_col] = np.where(df[size_col].isna(), np.nan, (df[size_col] <= threshold).astype(float))
    return df


def add_duration_size_adjustment(df, sides=('home', 'away'), suffix='', min_points=5):
    """Add the size-adjusted duration, the primary duration measure."""
    df = df.copy()
    for side in sides:
        size_col = f'{side}_huddle_closeness_ft2{suffix}'
        dur_col = f'{side}_huddle_duration_s{suffix}'
        adj_col = f'{side}_huddle_duration_s_adj{suffix}'
        mask = df[size_col].notna() & df[dur_col].notna()
        if mask.sum() < min_points:
            df[adj_col] = np.nan
            continue
        slope, intercept = np.polyfit(df.loc[mask, size_col], df.loc[mask, dur_col], 1)
        predicted = intercept + slope * df[size_col]
        df[adj_col] = df[dur_col] - predicted
    return df


NON_HUDDLE_CIRCLE_DIAMETER_FT = 1.75 * METERS_TO_FEET


def add_circle_non_huddle_flag(df, sides=('home', 'away'), suffix='', threshold_ft=NON_HUDDLE_CIRCLE_DIAMETER_FT):
    """Flag huddles whose circle diameter exceeds the non-huddle threshold."""
    df = df.copy()
    for side in sides:
        size_col = f'{side}_circle_diameter_ft{suffix}'
        flag_col = f'{side}_non_huddle{suffix}'
        df[flag_col] = np.where(df[size_col].isna(), np.nan, (df[size_col] > threshold_ft).astype(float))
    return df


def huddle_formation_feature_bundle(huddle_video_df, sides=('home', 'away'), suffix='', min_n=4):
    """Run huddle_formation_features for one column variant."""
    feats = huddle_formation_features(huddle_video_df, sides=sides, suffix=suffix, min_n=min_n)
    feats = add_huddle_occurred_flag(feats, sides=sides, suffix=suffix)
    feats = add_circle_non_huddle_flag(feats, sides=sides, suffix=suffix)
    feats = add_duration_size_adjustment(feats, sides=sides, suffix=suffix)
    return feats


def _other_team_label(team_label, labels):
    return next(t for t in labels.values() if t != team_label)


def _erring_team(row, labels):
    '''The team that committed the error on this row (None if no error).'''
    if pd.isna(row['error_by']):
        return None
    return _other_team_label(row['team_win'], labels)


def _next_points_for_team(df, team_label, category_fn, window_size=5):
    """For each row, the next points won by a team within the same set."""
    n = len(df)
    windows = []
    for i in range(n):
        s = df.loc[i, 'set']
        labels = []
        found = 0
        j = i + 1
        while j < n and df.loc[j, 'set'] == s and found < window_size:
            if df.loc[j, 'team_win'] == team_label:
                row_labels = category_fn(df.loc[j])
                if row_labels:
                    labels.extend(row_labels)
                    found += 1
            j += 1
        windows.append(labels)
    return windows


def _normalized_entropy(labels, window_size=5):
    if not labels:
        return np.nan
    probs = pd.Series(labels).value_counts(normalize=True)
    h = -np.sum(probs * np.log(probs))
    return h / np.log(window_size)


def win_player_entropy(df, labels):
    """Normalized entropy of which players scored a team's next points."""
    df = df.sort_values(['set', 'point_num']).reset_index(drop=True)

    def contributing_players(row):
        if row['won_via'] == 'ace' and pd.notna(row['server']):
            return [row['server']]
        if row['won_via'] == 'attack' and pd.notna(row['attacker']):
            if pd.notna(row['assist']):
                return [f"{row['attacker']} | {row['assist']}"]
            return [row['attacker']]
        if row['won_via'] == 'block' and pd.notna(row['blockers']):
            blockers = [b.strip() for b in row['blockers'].split(';')]
            return [' & '.join(blockers)]
        return []

    for team_var, team_label in labels.items():
        windows = _next_points_for_team(df, team_label, contributing_players)
        df[f'{team_var}_win_player_entropy'] = [_normalized_entropy(w) for w in windows]
    return df


def win_method_entropy(df, labels):
    """Normalized entropy of how a team's next points were won."""
    df = df.sort_values(['set', 'point_num']).reset_index(drop=True)

    def contributing_method(row):
        if row['won_via'] in ('attack', 'block'):
            return [row['won_via']]
        if row['won_via'] == 'error' and row['play_type'] == 'attack_error':
            return [row['won_via']]
        return []

    for team_var, team_label in labels.items():
        windows = _next_points_for_team(df, team_label, contributing_method)
        df[f'{team_var}_win_method_entropy'] = [_normalized_entropy(w) for w in windows]
    return df


def point_win_diversity_score(df, labels):
    """Point-win diversity: mean of the normalized player and win-method entropies, per team."""
    df = win_player_entropy(df, labels)
    df = win_method_entropy(df, labels)
    for team_var in labels:
        df[f'{team_var}_point_win_diversity_score'] = (
            df[f'{team_var}_win_player_entropy'] + df[f'{team_var}_win_method_entropy']
        ) / 2
    return df


def post_error_response_score(df):
    """(w - e) / (w + e) over the points following a player error."""
    df = df.sort_values(['set', 'point_num']).reset_index(drop=True)
    n = len(df)

    def is_positive(row, player):
        if row['won_via'] == 'attack':
            return player in (row['attacker'], row['assist'])
        if row['won_via'] == 'ace':
            return player == row['server']
        if row['won_via'] == 'block' and pd.notna(row['blockers']):
            blockers = [b.strip() for b in row['blockers'].split(';')]
            return player in blockers
        return False

    def is_error(row, player):
        return pd.notna(row['error_by']) and row['error_by'] == player

    scores = []
    for i in range(n):
        player = df.loc[i, 'error_by']
        if pd.isna(player):
            scores.append(np.nan)
            continue

        s = df.loc[i, 'set']
        w = 0
        e = 0
        j = i + 1
        while j < n and df.loc[j, 'set'] == s and j <= i + 5:
            row = df.loc[j]
            if is_positive(row, player):
                w += 1
            elif is_error(row, player):
                e += 1
            j += 1

        scores.append((w - e) / (w + e) if (w + e) > 0 else np.nan)

    df['post_error_response_score'] = scores
    return df


def team_post_error_response_score(df, labels):
    """Team-level post-error response score."""
    df = df.sort_values(['set', 'point_num']).reset_index(drop=True)
    n = len(df)

    for team_var, team_label in labels.items():
        scores = []
        for i in range(n):
            if _erring_team(df.loc[i], labels) != team_label:
                scores.append(np.nan)
                continue

            s = df.loc[i, 'set']
            w = 0
            e = 0
            j = i + 1
            while j < n and df.loc[j, 'set'] == s and j <= i + 5:
                row = df.loc[j]
                if row['team_win'] == team_label and row['won_via'] in ('attack', 'ace', 'block'):
                    w += 1
                elif _erring_team(row, labels) == team_label:
                    e += 1
                j += 1

            scores.append((w - e) / (w + e) if (w + e) > 0 else np.nan)

        df[f'{team_var}_post_error_response_score'] = scores

    return df


def prev_point_error_flag(df, labels):
    """Per-team flag: 1 if the preceding point in the set was lost by that team's own error."""
    df = df.sort_values(['set', 'point_num']).reset_index(drop=True)
    n = len(df)

    for team_var, team_label in labels.items():
        flags = []
        for i in range(n):
            if i == 0 or df.loc[i, 'set'] != df.loc[i - 1, 'set']:
                flags.append(0)
            else:
                flags.append(int(_erring_team(df.loc[i - 1], labels) == team_label))
        df[f'{team_var}_prev_point_error'] = flags

    return df


def momentum_feature(df, labels):
    """Signed win/loss streak per team entering each point."""
    df = df.sort_values(['set', 'point_num']).reset_index(drop=True)
    n = len(df)

    for team_var, team_label in labels.items():
        momentum = []
        streak = 0
        prev_set = None
        for i in range(n):
            s = df.loc[i, 'set']
            if s != prev_set:
                streak = 0
                prev_set = s
            momentum.append(streak)
            if df.loc[i, 'team_win'] == team_label:
                streak = streak + 1 if streak > 0 else 1
            else:
                streak = streak - 1 if streak < 0 else -1
        df[f'{team_var}_momentum'] = momentum

    return df


def point_differential_feature(df):
    """Score differential at each point, per team."""
    df = df.copy()
    df['home_point_differential'] = df['home_score'] - df['visitor_score']
    df['away_point_differential'] = -df['home_point_differential']
    return df


def _build_player_roster(df, labels):
    """Infer each player's team from the play-by-play."""
    roster = {}
    for _, row in df.iterrows():
        winners = []
        if row['won_via'] == 'attack':
            winners += [row['attacker'], row['assist']]
        elif row['won_via'] == 'ace':
            winners += [row['server']]
        elif row['won_via'] == 'block' and pd.notna(row['blockers']):
            winners += [b.strip() for b in row['blockers'].split(';')]
        for p in winners:
            if pd.notna(p):
                roster[p] = row['team_win']
        if pd.notna(row['error_by']):
            roster.setdefault(row['error_by'], _other_team_label(row['team_win'], labels))
    return roster


def _wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z**2 / (4 * n**2)) ** 0.5) / denom
    return (center - margin, center + margin)


def _kill_baseline_rates(df):
    """Return baseline_rate(setter, hitter): the setter's match-wide rate of setting that hitter on a kill."""
    kills = df[(df['won_via'] == 'attack') & df['assist'].notna() & df['attacker'].notna()]
    baseline_opps = kills.groupby('assist').size()
    baseline_hits = kills.groupby(['assist', 'attacker']).size()

    def baseline_rate(setter, hitter):
        denom = baseline_opps.get(setter, 0)
        if denom == 0:
            return np.nan
        return baseline_hits.get((setter, hitter), 0) / denom

    return baseline_rate


def refeed_risk_ratio_events(pbp_df, labels, ks=(1, 2, 3, 4, 5, 6)):
    """Event-level re-feed risk: one row per error."""
    df = pbp_df.sort_values(['set', 'point_num']).reset_index(drop=True)
    n = len(df)
    roster = _build_player_roster(df, labels)
    baseline_rate = _kill_baseline_rates(df)

    rows = []
    for i in range(n):
        row = df.loc[i]
        if row['won_via'] != 'error':
            continue
        X, error_type = row['error_by'], row['play_type']
        x_team = roster.get(X)
        if x_team is None:
            continue

        s = row['set']
        opp_rows = []
        j = i + 1
        while j < n and df.loc[j, 'set'] == s and len(opp_rows) < max(ks):
            r = df.loc[j]
            if r['won_via'] == 'attack' and pd.notna(r['assist']) and roster.get(r['attacker']) == x_team:
                opp_rows.append(r)
            j += 1

        rec = {
            'set': s,
            'point_num': row['point_num'],
            'player': X,
            'team': x_team,
            'error_type': error_type,
            'huddle_after': row['huddle_after'],
        }
        for k in ks:
            window = opp_rows[:k]
            n_opps = len(window)
            n_hits = sum(1 for r in window if r['attacker'] == X)
            p_post = n_hits / n_opps if n_opps else np.nan

            setter_mix = pd.Series([r['assist'] for r in window]).value_counts() if window else pd.Series(dtype=object)
            weighted_sum, weighted_n = 0.0, 0
            for setter, cnt in setter_mix.items():
                br = baseline_rate(setter, X)
                if pd.notna(br):
                    weighted_sum += cnt * br
                    weighted_n += cnt
            p_baseline = weighted_sum / weighted_n if weighted_n else np.nan
            rrr = p_post / p_baseline if pd.notna(p_baseline) and p_baseline > 0 else np.nan

            rec[f'n_opportunities_k{k}'] = n_opps
            rec[f'n_hits_k{k}'] = n_hits
            rec[f'p_post_error_k{k}'] = p_post
            rec[f'p_baseline_k{k}'] = p_baseline
            rec[f'RRR_k{k}'] = rrr

        rows.append(rec)

    return pd.DataFrame(rows)


def refeed_features_by_huddle(pbp_df, labels, ks=(1, 2, 3, 4, 5, 6)):
    """Huddle-level numeric version of refeed_risk_ratio_events."""
    events = refeed_risk_ratio_events(pbp_df, labels, ks=ks)
    side_of = {label: side for side, label in labels.items()}
    events = events.copy()
    events['side'] = events['team'].map(side_of)

    error_type_short = {
        'attack_error': 'attack',
        'service_error': 'service',
        'ball_handling_error': 'ballhandling',
    }
    for et, short in error_type_short.items():
        events[f'error_was_{short}'] = (events['error_type'] == et).astype(int)

    value_cols = [f'{stat}_k{k}' for k in ks for stat in
                  ('n_opportunities', 'n_hits', 'p_post_error', 'p_baseline', 'RRR')]
    value_cols += [f'error_was_{short}' for short in error_type_short.values()]

    per_side = []
    for side in labels:
        sub = events.loc[events['side'] == side, ['huddle_after'] + value_cols]
        sub = sub.rename(columns={'huddle_after': 'huddle_label'})
        sub = sub.rename(columns={c: f'{side}_refeed_{c}' for c in value_cols})
        per_side.append(sub)

    return per_side[0].merge(per_side[1], on='huddle_label', how='outer')


def refeed_risk_ratio(pbp_df, labels, ks=(1, 2, 3, 4, 5, 6), min_errors=3):
    """Whether a setter routes sets away from a hitter right after that hitter's error."""
    df = pbp_df.sort_values(['set', 'point_num']).reset_index(drop=True)
    n = len(df)
    roster = _build_player_roster(df, labels)
    baseline_rate = _kill_baseline_rates(df)

    windows = {}   # (player, error_type, k) -> list of (setter, was_hit)
    n_errors = {}  # (player, error_type) -> count

    for i in range(n):
        row = df.loc[i]
        if row['won_via'] != 'error':
            continue
        X, error_type = row['error_by'], row['play_type']
        x_team = roster.get(X)
        if x_team is None:
            continue
        n_errors[(X, error_type)] = n_errors.get((X, error_type), 0) + 1

        s = row['set']
        opp_rows = []
        j = i + 1
        while j < n and df.loc[j, 'set'] == s and len(opp_rows) < max(ks):
            r = df.loc[j]
            if r['won_via'] == 'attack' and pd.notna(r['assist']) and roster.get(r['attacker']) == x_team:
                opp_rows.append(r)
            j += 1

        for k in ks:
            key = (X, error_type, k)
            windows.setdefault(key, []).extend(
                (r['assist'], r['attacker'] == X) for r in opp_rows[:k]
            )

    results = []
    for (X, error_type, k), obs in windows.items():
        if n_errors[(X, error_type)] < min_errors:
            continue

        n_opps = len(obs)
        n_hits = sum(1 for _, hit in obs if hit)
        p_post = n_hits / n_opps if n_opps else np.nan

        setter_mix = pd.Series([setter for setter, _ in obs]).value_counts()
        weighted_sum, weighted_n = 0.0, 0
        for setter, cnt in setter_mix.items():
            br = baseline_rate(setter, X)
            if pd.notna(br):
                weighted_sum += cnt * br
                weighted_n += cnt
        p_baseline = weighted_sum / weighted_n if weighted_n else np.nan

        rrr = p_post / p_baseline if pd.notna(p_baseline) and p_baseline > 0 else np.nan
        ci_lo, ci_hi = _wilson_ci(n_hits, n_opps)

        results.append({
            'player': X,
            'team': roster.get(X),
            'error_type': error_type,
            'k': k,
            'n_errors': n_errors[(X, error_type)],
            'n_eligible_opportunities_in_window': n_opps,
            'n_sets_to_X_in_window': n_hits,
            'p_post_error': p_post,
            'p_baseline_reweighted': p_baseline,
            'RRR': rrr,
            'p_post_error_ci_lo': ci_lo,
            'p_post_error_ci_hi': ci_hi,
        })

    return pd.DataFrame(results)


def join_huddle_features(pbp_df, huddle_features_df, labels,
                          save_path=os.path.join(DEFAULT_OUT_DIR, "pbp_with_huddle_features.csv")):
    """Inner-join play-by-play rows with the features of the preceding huddle."""
    huddle_features_df = huddle_features_df.drop(columns=['set', 'huddle_number'])
    merged = pbp_df.merge(
        huddle_features_df,
        left_on='huddle_before',
        right_on='huddle_label',
        how='inner',
    )
    merged = merged.groupby('huddle_label', as_index=False).first()
    for team_var, team_label in labels.items():
        merged[f'{team_var}_point_won'] = (merged['team_win'] == team_label).astype(int)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    merged.to_csv(save_path, index=False)
    return merged


def filter_valid_huddles(merged_df, huddle_video_df, sides=('home', 'away'), min_n=4, save_path=None):
    """Keep only huddles with usable tracking."""
    bad_huddles = set()
    for (s, h), group in huddle_video_df.groupby(['set', 'huddle_number']):
        for side in sides:
            if (group[f'{side}_n'] < min_n).all():
                bad_huddles.add(_huddle_label(h, s))
                break

    filtered = merged_df[~merged_df['huddle_label'].isin(bad_huddles)].reset_index(drop=True)

    if save_path:
        filtered.to_csv(save_path, index=False)

    return filtered


def remove_last_huddle_per_set(df):
    """Drop each set's last huddle."""
    keep = df.groupby('set')['huddle_number'].transform('max') != df['huddle_number']
    return df[keep].reset_index(drop=True)


def build_final_dataset(pbp_df, huddle_video_df, game_id, labels, has_geom, min_n=4, save_path=None):
    """Build the per-huddle feature table for one game."""
    pbp_df = huddles_bf_af(pbp_df.copy())
    pbp_df = point_win_diversity_score(pbp_df, labels)
    pbp_df = post_error_response_score(pbp_df)
    pbp_df = team_post_error_response_score(pbp_df, labels)
    pbp_df = prev_point_error_flag(pbp_df, labels)
    pbp_df = point_differential_feature(pbp_df)
    pbp_df = momentum_feature(pbp_df, labels)

    win_flags = huddle_win_flags(huddle_video_df, pbp_df, labels)

    huddle_feats = huddle_formation_feature_bundle(huddle_video_df, suffix='')
    if has_geom:
        geom_feats = huddle_formation_feature_bundle(huddle_video_df, suffix='_geom')
        huddle_feats = huddle_feats.merge(geom_feats, on=['set', 'huddle_number', 'huddle_label'])

    full = huddle_feats.merge(win_flags, on=['set', 'huddle_number', 'huddle_label'], how='left')

    point_feature_cols = [c for c in pbp_df.columns if c.endswith((
        'win_player_entropy', 'win_method_entropy', 'point_win_diversity_score',
        'post_error_response_score', 'prev_point_error', 'point_differential', 'momentum',
    ))]
    next_point_features = (
        pbp_df[['huddle_before'] + point_feature_cols]
        .dropna(subset=['huddle_before'])
        .drop_duplicates(subset='huddle_before')
        .rename(columns={'huddle_before': 'huddle_label'})
    )
    full = full.merge(next_point_features, on='huddle_label', how='left')

    refeed_feats = refeed_features_by_huddle(pbp_df, labels)
    full = full.merge(refeed_feats, on='huddle_label', how='left')

    full = filter_valid_huddles(full, huddle_video_df, min_n=min_n)
    full = remove_last_huddle_per_set(full)

    full.insert(0, 'game_id', game_id)
    id_cols = ['game_id', 'set', 'huddle_number']
    feature_cols = [c for c in full.columns if c not in id_cols and c != 'huddle_label']
    full = full[id_cols + feature_cols]

    non_numeric = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(full[c])]
    if non_numeric:
        raise ValueError(f"Non-numeric feature columns found: {non_numeric}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        full.to_csv(save_path, index=False)

    return full


def run_game_pipeline(cfg, out_dir=DEFAULT_OUT_DIR):
    """Load one game's play-by-play and tracking data and build its per-huddle feature table."""
    pbp_df = pd.read_csv(cfg['pbp_path'])
    huddle_video_df = load_huddle_video_data(cfg['huddle_csv'])
    huddle_video_df = add_home_away_columns(huddle_video_df, cfg, pbp_df=pbp_df)
    labels = {'home': cfg['home_team'], 'away': cfg['away_team']}
    has_geom = 'home_n_geom' in huddle_video_df.columns

    final_data = build_final_dataset(
        pbp_df, huddle_video_df, game_id=cfg['game_id'], labels=labels, has_geom=has_geom,
        save_path=os.path.join(out_dir, f"{cfg['game_id']}_full_features.csv"),
    )
    rrr = refeed_risk_ratio(pbp_df, labels)
    return final_data, rrr


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stimulus-map", default=stimulus_map.DEFAULT_STIMULUS_MAP_PATH,
                         help="Path to stimulus_map.csv")
    parser.add_argument("--parameters-csv", default=DEFAULT_PARAMETERS_CSV,
                         help="Path to parameters.csv (camera_static / teams_switch / etc.)")
    parser.add_argument("--drive-root", default=DEFAULT_DRIVE_ROOT,
                         help="Root folder containing <matchup>/all_sets.csv from sam3_automation.py's "
                              "--features-out-root and <matchup>/play_by_play.csv from build_dataset.py "
                              "(default: this repo's data/ folder)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                         help="Directory to write the per-game and combined feature CSVs into")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    games = build_games(args.stimulus_map, args.parameters_csv, args.drive_root)
    assert games, "No matchups had both play-by-play and huddle tracking data ready — nothing to do."

    os.makedirs(args.out_dir, exist_ok=True)
    all_features = []
    for cfg in games:
        final_data, rrr = run_game_pipeline(cfg, out_dir=args.out_dir)
        all_features.append(final_data)
        print(f"[{cfg['game_id']}] Saved {len(final_data)} huddles x {len(final_data.columns)} columns.")

        rrr.to_csv(os.path.join(args.out_dir, f"{cfg['game_id']}_refeed_risk_ratio.csv"), index=False)
        print(f"[{cfg['game_id']}] Saved {len(rrr)} RRR rows (descriptive only - see refeed_risk_ratio docstring for caveats).")

    combined = pd.concat(all_features, ignore_index=True)
    combined.to_csv(os.path.join(args.out_dir, "all_games_full_features.csv"), index=False)
    print(f"Saved combined dataset: {len(combined)} huddles x {len(combined.columns)} columns across {len(games)} games.")
