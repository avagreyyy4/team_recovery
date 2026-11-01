"""Build data/long_dataset.csv, the one-row-per-team-huddle table the analysis reads.

Scrapes any missing play_by_play.csv, then joins each game's SAM3 tracking (all_sets.csv)
with its play-by-play and stacks the games.

Run: python3 analysis/feature_extraction/build_dataset.py
"""

import os
import sys
from pathlib import Path

from stimulus_map import load_rows, group_by_matchup, DEFAULT_STIMULUS_MAP_PATH
from pbp_scrape import scrape_matchup, DEFAULT_DRIVE_ROOT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'statistical_analysis'))
import huddle_feature_analysis as hfa

OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data', 'long_dataset.csv')


def ensure_play_by_play(matchups=None, drive_root=DEFAULT_DRIVE_ROOT, stimulus_map_path=DEFAULT_STIMULUS_MAP_PATH):
    """Scrape play_by_play.csv for any matchup that is missing one."""
    drive_root = Path(drive_root)
    rows = load_rows(stimulus_map_path)
    rows = [r for r in rows if r.get("type", "").strip() == "huddle_video"]
    groups = group_by_matchup(rows)
    if matchups is None:
        matchups = sorted(groups)

    missing = [m for m in matchups if not (drive_root / m / "play_by_play.csv").is_file()]
    if not missing:
        print(f"Play-by-play already present for all {len(matchups)} games.")
        return

    print(f"Scraping play-by-play for {len(missing)} game(s) missing from {drive_root}.")
    for matchup in missing:
        if matchup not in groups:
            print(f"[SKIP] {matchup}: not found in {stimulus_map_path}")
            continue
        scrape_matchup(matchup, groups[matchup], drive_root=drive_root)


def export_long_dataset():
    long_df = hfa.build_long_dataset()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    long_df.to_csv(OUTPUT_PATH, index=False)
    print(f'Saved {len(long_df)} rows x {len(long_df.columns)} cols to {OUTPUT_PATH}')


if __name__ == "__main__":
    ensure_play_by_play()
    export_long_dataset()
