"""Scrapes a Sidearm Sports box-score page into play_by_play.csv per
matchup, and builds the per-game config list (GAMES) the rest of the
pipeline reads.
"""

import csv
import os
import re
import sys
from pathlib import Path

import pandas as pd

from stimulus_map import REPO_ROOT, load_rows, group_by_matchup, find_by_team_tokens, DEFAULT_STIMULUS_MAP_PATH

DEFAULT_PARAMETERS_CSV = REPO_ROOT / "analysis" / "modeling" / "parameters.csv"
DEFAULT_DRIVE_ROOT = str(REPO_ROOT / "data")

CSV_COLUMNS = [
    "set", "point_num", "huddle", "team_win", "won_via", "play_type",
    "serving_team", "server", "attacker", "assist", "blockers", "error_by",
    "ace_receiver", "timeout_before", "subs_before",
    "visitor_score", "home_score", "description",
    "home_team", "away_team",
]


# ── description parsing (one play's text -> structured fields) ──

def clean(text):
    """Strip tags/whitespace from an HTML fragment."""
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_description(desc):
    """Break a play description into structured fields."""
    fields = {
        "server": "", "play_type": "", "won_via": "", "attacker": "",
        "assist": "", "blockers": "", "error_by": "", "ace_receiver": "",
    }

    if re.match(r"Timeout (.+?)\.?$", desc):
        fields["play_type"] = "timeout"
        return fields
    if re.match(r"(.+?) subs: (.+?)\.?$", desc):
        fields["play_type"] = "substitution"
        return fields
    if re.match(r"(.+?) starters: (.+?)\.?$", desc):
        fields["play_type"] = "lineup"
        return fields

    m = re.match(r"\[(.+?)\]\s*(.+?)\.?$", desc)
    if not m:
        fields["play_type"] = "other"
        return fields
    fields["server"], event = m.groups()

    m = re.match(r"Kill by (.+?)(?: \(from (.+?)\))?(?:, block error by (.+?))?$", event)
    if m:
        fields["play_type"] = "kill"
        fields["won_via"] = "attack"
        fields["attacker"] = m.group(1)
        fields["assist"] = m.group(2) or ""
        fields["error_by"] = m.group(3) or ""
        return fields

    m = re.match(r"Attack error by (.+?)(?: \(from (.+?)\))?(?: \(block by (.+?)\))?$", event)
    if m:
        fields["play_type"] = "attack_error"
        fields["won_via"] = "block" if m.group(3) else "error"
        fields["attacker"] = m.group(1)
        fields["assist"] = m.group(2) or ""
        fields["error_by"] = m.group(1)
        fields["blockers"] = m.group(3) or ""
        return fields

    m = re.match(r"Service ace \((.+?)\)$", event)
    if m:
        fields["play_type"] = "service_ace"
        fields["won_via"] = "ace"
        fields["ace_receiver"] = m.group(1)
        return fields

    if event.startswith("Service error"):
        fields["play_type"] = "service_error"
        fields["won_via"] = "error"
        fields["error_by"] = fields["server"]
        return fields

    # wittenbergtigers.com: "Ball Handling Error by X"; blufftonbeavers.com:
    # "Bad set by X" -- same category, hence case-insensitive combined match.
    m = re.match(r"(?:Ball handling error by|Bad set by) (.+?)$", event, re.I)
    if m:
        fields["play_type"] = "ball_handling_error"
        fields["won_via"] = "error"
        fields["error_by"] = m.group(1)
        return fields

    fields["play_type"] = "other"
    return fields


# ── HTML parsing (box-score page -> point-level rows) ──

POINT_SUFFIX_RE = re.compile(r"\.?\s*Point\s+\S+\.?$")


def _rows_from_cells(sections, visitor_team, home_team):
    """Parse box-score table cells into play-by-play rows."""
    rows = []
    for set_num, section_rows in sections:
        prev_v, prev_h = 0, 0
        computed_v, computed_h = 0, 0
        point_num = 0
        timeout_before = 0
        subs_before = 0
        for serving_team, score, raw_description in section_rows:
            description = POINT_SUFFIX_RE.sub("", raw_description).strip()
            fields = parse_description(description)

            # subs/timeouts/lineup rows don't advance point_num -- they'd
            # throw off downstream huddle numbering and score tracking.
            if fields["play_type"] == "timeout":
                timeout_before = 1
                continue
            if fields["play_type"] == "substitution":
                subs_before = 1
                continue
            if fields["play_type"] == "lineup":
                continue

            if score:
                v, h = int(score.group(1)), int(score.group(2))
            else:
                v, h = prev_v, prev_h

            # Only the direction of change matters for team_win, not the
            # absolute score -- robust to some sites' occasional score typos.
            team_win = ""
            if v > prev_v:
                team_win = visitor_team
                computed_v += 1
            elif h > prev_h:
                team_win = home_team
                computed_h += 1
            prev_v, prev_h = v, h

            point_num += 1
            rows.append({
                "set": int(set_num),
                "point_num": point_num,
                "huddle": point_num - 1,
                "team_win": team_win,
                "serving_team": serving_team,
                "timeout_before": timeout_before,
                "subs_before": subs_before,
                "visitor_score": computed_v,
                "home_score": computed_h,
                "description": description,
                "home_team": home_team,
                "away_team": visitor_team,
                **fields,
            })
            timeout_before = 0
            subs_before = 0
    return rows


def _iter_rows_new_template(sections):
    for set_num, section in sections:
        cells = []
        for tr in re.findall(r"<tr>(.*?)</tr>", section, re.S):
            serve = re.search(r'<td class="text-center">(.*?)</td>', tr)
            score = re.search(r'<td class="hide-on-large text-center"[^>]*>\s*(\d+)-(\d+)', tr)
            desc = re.search(r'<td class="hide-on-large" aria-hidden="true">(.*?)</td>', tr, re.S)
            if not desc:
                continue  # header or malformed row
            cells.append((clean(serve.group(1)) if serve else "", score, clean(desc.group(1))))
        yield set_num, cells


def _iter_rows_legacy_template(html):
    """Older box-score pages use a different HTML layout entirely."""
    set_starts = [(m.start(), m.group(1)) for m in re.finditer(r'<th id="set(\d+)">', html)]
    if not set_starts:
        return
    table_end = html.find("</table>", set_starts[-1][0])
    boundaries = [pos for pos, _ in set_starts] + [table_end if table_end != -1 else len(html)]

    for i, (start, set_num) in enumerate(set_starts):
        section = html[start:boundaries[i + 1]]
        cells = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", section, re.S):
            tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
            if len(tds) != 3:
                continue  # set-header row, or malformed
            left, desc, right = (clean(t) for t in tds)
            score = re.search(r"(\d+)-(\d+)", left) or re.search(r"(\d+)-(\d+)", right)
            if not desc:
                continue
            cells.append(("", score, desc))
        yield set_num, cells


# Team names show up in different markup across sites -- try each pattern.
TEAM_NAME_PATTERNS = [
    re.compile(
        r'<th scope="col" class="hide-on-medium-down text-center">([^<]+)<.*?'
        r'<th scope="col" class="hide-on-medium-down text-center">([^<]+)<', re.S),
    re.compile(
        r'<span class="stats-header">\s*(?:<a[^>]*>)?\s*([A-Za-z][^<\n]*?)\s*(?:</a>)?\s*<br.*?'
        r'<span class="stats-header">\s*(?:<a[^>]*>)?\s*([A-Za-z][^<\n]*?)\s*(?:</a>)?\s*<br', re.S),
    re.compile(r'<h2>\s*([A-Za-z][^<]*?)\s+at\s+([A-Za-z][^<]*?)\s*</h2>'),
]


def _extract_team_names(html):
    for pattern in TEAM_NAME_PATTERNS:
        m = pattern.search(html)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None


def parse_play_by_play(html):
    names = _extract_team_names(html)
    if names is None:
        sys.exit("Could not find team names -- page layout may have changed.")
    visitor_team, home_team = names

    sections_new = re.findall(r'<section id="set-(\d+)">(.*?)</table>', html, re.S)
    if sections_new:
        rows = _rows_from_cells(_iter_rows_new_template(sections_new), visitor_team, home_team)
    else:
        legacy_sections = list(_iter_rows_legacy_template(html))
        if not legacy_sections:
            sys.exit("No play-by-play sets found -- check the URL points at a box score page.")
        rows = _rows_from_cells(legacy_sections, visitor_team, home_team)

    if not rows:
        sys.exit("No play-by-play sets found -- check the URL points at a box score page.")
    return rows, visitor_team, home_team


# ── scraping (fetch a matchup's box-score page, write play_by_play.csv) ──

def fetch_html(url):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return resp.read().decode("utf-8", errors="replace")


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def report(rows: list[dict], visitor_team: str, home_team: str, out_path: Path) -> None:
    n_sets = max(r["set"] for r in rows)
    print(f"Wrote {len(rows)} points across {n_sets} sets to {out_path}")
    print(f"Teams: {visitor_team} (visitor) vs {home_team} (home)")
    counts = {}
    for r in rows:
        key = (r["team_win"], r["won_via"])
        counts[key] = counts.get(key, 0) + 1
    for (team, via), n in sorted(counts.items()):
        print(f"  {team} via {via}: {n}")


def scrape_matchup(matchup: str, group_rows: list[dict], drive_root=DEFAULT_DRIVE_ROOT) -> None:
    # Every row for a matchup should agree on one pbp_link -- if not, fix
    # stimulus_map.csv at the source, don't guess which value was real.
    links = {r["pbp_link"] for r in group_rows}
    assert len(links) == 1, f"Multiple distinct pbp_link values for one matchup: {links}"
    url = links.pop()
    out_path = Path(drive_root) / matchup / "play_by_play.csv"

    print(f"\n=== {matchup} ===")
    print(f"Fetching {url}")

    rows, visitor_team, home_team = parse_play_by_play(fetch_html(url))
    write_csv(rows, out_path)
    report(rows, visitor_team, home_team, out_path)


def _bool(value) -> bool:
    return str(value).strip().upper() == "TRUE"


# Many teams were saved with shortened names in play_by_play.csv.
HOME_TEAM_ABBREV_TO_FULL = {
    'adrian': 'adrian', 'bc': 'bridgewater', 'bluffton': 'bluffton', 'eu': 'emory',
    'hope': 'hope', 'ith': 'ithaca', 'mcla': 'mcla', 'msj': 'msj', 'nyu': 'nyu',
    'smi': 'smith', 'swat': 'swarthmore', 'trintx': 'trinity', 'ulv': 'laverne',
    'tlu': 'texas lutheran', 'uni': 'union',
}


def build_games(stimulus_map_path=DEFAULT_STIMULUS_MAP_PATH,
                 parameters_csv=DEFAULT_PARAMETERS_CSV,
                 drive_root=DEFAULT_DRIVE_ROOT):
    """Per-game config list, matching stimulus_map.csv rows up with
    parameters.csv settings (team names vary in spelling between the two)."""
    rows = load_rows(stimulus_map_path)
    rows = [r for r in rows if r.get("type", "").strip() == "huddle_video"]
    matchups = sorted(group_by_matchup(rows))

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

        matched_game_id = find_by_team_tokens(matchup, list(params["game_id"]))
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
