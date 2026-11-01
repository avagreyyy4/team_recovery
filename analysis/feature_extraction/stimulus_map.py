"""Helpers for reading stimulus_map.csv, the source of truth for which games exist."""

import csv
import re
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_STIMULUS_MAP_PATH = REPO_ROOT / "stimulus_map.csv"

# Matches the game folder out of a location like:
#   /content/drive/MyDrive/Northwestern/Computer_Vision/Calvin_vs_Hope/huddles/huddle_1_set_1.mp4
LOCATION_MATCHUP_RE = re.compile(r"Computer_Vision/([^/]+)/huddles/")


def load_rows(csv_path: Path = DEFAULT_STIMULUS_MAP_PATH) -> list[dict]:
    """Load stimulus_map.csv with whitespace-stripped column names."""
    with open(csv_path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        rows = list(reader)

    assert rows, f"{csv_path} contains no data rows."
    return rows


def matchup_name(row: dict) -> str:
    """Game folder name parsed from a stimulus_map `location` value."""
    match = LOCATION_MATCHUP_RE.search(row["location"])
    assert match, f"Could not parse a matchup folder out of location: {row['location']!r}"
    return match.group(1)


def group_by_matchup(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        groups[matchup_name(row)].append(row)
    return dict(groups)


def canonical_youtube_url(group_rows: list[dict]) -> str:
    """The meta_data_link (YouTube URL) for a matchup."""
    urls = {row["meta_data_link"] for row in group_rows}
    assert len(urls) == 1, f"Expected one YouTube URL per matchup, found {len(urls)}: {urls}"
    return urls.pop()


def team_tokens(name: str) -> frozenset:
    """'Calvin_v_Hope' / 'Calvin_vs_Hope' -> frozenset({'calvin', 'hope'})."""
    return frozenset(t.lower() for t in re.split(r'_vs?_', name) if t)


def find_by_team_tokens(target_name: str, candidate_names: list[str]) -> str:
    """Match a game name to a matchup folder by team-name tokens."""
    target = team_tokens(target_name)
    matches = [c for c in candidate_names if team_tokens(c) == target]
    assert matches, f"No candidate matches {target_name!r} (teams: {sorted(target)})"
    assert len(matches) == 1, f"Ambiguous match for {target_name!r}: {matches}"
    return matches[0]
