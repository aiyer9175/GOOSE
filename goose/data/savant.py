"""Baseball Savant leaderboard CSV downloads.

Each function caches the raw CSV under data/raw/savant/<board>/<year>.csv and
returns a DataFrame. Boards that don't exist for a year (e.g. bat tracking
before 2023) come back empty.
"""
from __future__ import annotations

import io

import pandas as pd

from .http import RAW, cached_text

BASE = "https://baseballsavant.mlb.com"
OUT = RAW / "savant"

# Custom-leaderboard fields. Rate stats only; counting stats come from the MLB API.
BATTER_FIELDS = [
    "player_age", "pa", "ab", "k_percent", "bb_percent", "woba", "xwoba", "xba", "xslg", "xiso",
    "xobp", "wobacon", "xwobacon", "babip", "isolated_power",
    "exit_velocity_avg", "launch_angle_avg", "sweet_spot_percent", "barrel_batted_rate",
    "hard_hit_percent", "avg_best_speed", "avg_hyper_speed", "solidcontact_percent",
    "flareburner_percent", "poorlyunder_percent", "poorlytopped_percent", "poorlyweak_percent",
    "groundballs_percent", "flyballs_percent", "linedrives_percent", "popups_percent",
    "pull_percent", "straightaway_percent", "opposite_percent",
    "z_swing_percent", "z_swing_miss_percent", "oz_swing_percent", "oz_swing_miss_percent",
    "oz_contact_percent", "iz_contact_percent", "swing_percent", "whiff_percent",
    "meatball_swing_percent", "edge_percent", "f_strike_percent",
    "avg_swing_speed", "fast_swing_rate", "avg_swing_length", "squared_up_contact",
    "squared_up_swing", "blasts_contact", "blasts_swing", "swords",
    "attack_angle", "attack_direction", "ideal_angle_rate", "vertical_swing_path",
    "sprint_speed", "hp_to_1b", "n_bolts",
    "batting_run_value", "swing_take_run_value", "runner_run_value", "fielding_run_value",
    "n_outs_above_average",
]

PITCHER_FIELDS = [
    "player_age", "pa", "p_formatted_ip", "p_game", "p_starting_p", "p_total_pitches", "p_quality_start",
    "k_percent", "bb_percent", "woba", "xwoba", "xba", "xslg", "xera", "babip", "wobacon", "xwobacon",
    "exit_velocity_avg", "launch_angle_avg", "barrel_batted_rate", "hard_hit_percent",
    "sweet_spot_percent", "groundballs_percent", "flyballs_percent", "linedrives_percent",
    "popups_percent", "pull_percent",
    "z_swing_percent", "z_swing_miss_percent", "oz_swing_percent", "oz_swing_miss_percent",
    "oz_contact_percent", "iz_contact_percent", "swing_percent", "whiff_percent",
    "in_zone_percent", "edge_percent", "f_strike_percent", "meatball_percent",
    "arm_angle", "pitch_hand",
    "fastball_avg_speed", "fastball_avg_spin", "fastball_avg_break_x", "fastball_avg_break_z_induced",
    "breaking_avg_speed", "breaking_avg_spin", "breaking_avg_break_x", "breaking_avg_break_z_induced",
    "offspeed_avg_speed", "offspeed_avg_spin", "offspeed_avg_break_x", "offspeed_avg_break_z_induced",
    "pitch_count_fastball", "pitch_count_breaking", "pitch_count_offspeed",
    "pitch_run_value_fastball", "pitch_run_value_breaking", "pitch_run_value_offspeed",
    "n_ff_formatted", "n_si_formatted", "n_fc_formatted", "n_sl_formatted", "n_st_formatted",
    "n_cu_formatted", "n_ch_formatted", "n_fs_formatted", "n_sv_formatted", "n_kn_formatted",
    "ff_avg_speed", "si_avg_speed", "fc_avg_speed",
]


def _csv(path_key: str, url: str, refresh: bool) -> pd.DataFrame:
    text = cached_text(OUT / f"{path_key}.csv", url, refresh=refresh)
    if not text.strip() or text.lstrip().startswith("<"):
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(text))
    return df.rename(columns={"last_name, first_name": "name", "pitcher": "player_id", "id": "player_id"})


def custom(year: int, kind: str, min_pa: int = 1, refresh: bool = False) -> pd.DataFrame:
    """kind: 'batter' | 'pitcher'."""
    fields = BATTER_FIELDS if kind == "batter" else PITCHER_FIELDS
    url = (f"{BASE}/leaderboard/custom?year={year}&type={kind}&filter=&min={min_pa}"
           f"&selections={','.join(fields)}&csv=true")
    return _csv(f"custom_{kind}/{year}", url, refresh)


def expected_stats(year: int, kind: str, refresh: bool = False) -> pd.DataFrame:
    url = f"{BASE}/leaderboard/expected_statistics?type={kind}&year={year}&position=&team=&min=1&csv=true"
    return _csv(f"expected_{kind}/{year}", url, refresh)


def exit_velo(year: int, kind: str, refresh: bool = False) -> pd.DataFrame:
    url = f"{BASE}/leaderboard/statcast?type={kind}&year={year}&position=&team=&min=1&csv=true"
    return _csv(f"exitvelo_{kind}/{year}", url, refresh)


def bat_tracking(year: int, refresh: bool = False) -> pd.DataFrame:
    url = (f"{BASE}/leaderboard/bat-tracking?attackZone=&batSide=&contactType=&count=&dateStart=&dateEnd="
           f"&gameType=Regular&isHardHit=&minSwings=1&minGroupSwings=1&pitchHand=&pitchType="
           f"&seasonStart={year}&seasonEnd={year}&team=&type=batter&csv=true")
    return _csv(f"bat_tracking/{year}", url, refresh)


def swing_path(year: int, refresh: bool = False) -> pd.DataFrame:
    url = (f"{BASE}/leaderboard/bat-tracking/swing-path-attack-angle?dateStart={year}-03-01&dateEnd={year}-11-30"
           f"&gameType=R&minSwings=1&minGroupSwings=1&seasonStart={year}&seasonEnd={year}&type=batter&csv=true")
    return _csv(f"swing_path/{year}", url, refresh)


def swing_timing(year: int, date_start: str | None = None, date_end: str | None = None,
                 refresh: bool = False) -> pd.DataFrame:
    """Swing timing / miss distance (early, late, over, under, tied-up, flailed). 2026+ only.

    The endpoint ignores year params and serves the current season; dateStart/dateEnd filter
    within it. Per-pitch-group splits come from the manual export in data/raw/user/.
    """
    start, end = date_start or f"{year}-03-01", date_end or f"{year}-11-30"
    url = (f"{BASE}/leaderboard/bat-tracking/swing-timing-miss-distance?type=batter&min=1"
           f"&dateStart={start}&dateEnd={end}&csv=true")
    df = _csv(f"swing_timing/{year}_{start}_{end}", url, refresh)
    if not df.empty and "year" in df and (df["year"] != year).any():
        return pd.DataFrame()
    return df


def sprint_speed(year: int, refresh: bool = False) -> pd.DataFrame:
    url = f"{BASE}/leaderboard/sprint_speed?year={year}&position=&team=&min=1&csv=true"
    return _csv(f"sprint_speed/{year}", url, refresh)


def pitch_arsenal_stats(year: int, kind: str = "pitcher", refresh: bool = False) -> pd.DataFrame:
    """Per pitch type run values / whiff / xwOBA. kind='batter' gives hitter results by pitch type."""
    url = f"{BASE}/leaderboard/pitch-arsenal-stats?type={kind}&pitchType=&year={year}&team=&min=1&csv=true"
    return _csv(f"arsenal_stats_{kind}/{year}", url, refresh)


def pitch_arsenals(year: int, metric: str, refresh: bool = False) -> pd.DataFrame:
    """metric: 'avg_speed' | 'avg_spin' | 'n_' (usage)."""
    url = f"{BASE}/leaderboard/pitch-arsenals?year={year}&min=1&type={metric}&hand=&csv=true"
    return _csv(f"arsenals_{metric}/{year}", url, refresh)


def arm_angles(year: int, refresh: bool = False) -> pd.DataFrame:
    url = (f"{BASE}/leaderboard/pitcher-arm-angles?batSide=&dateStart=&dateEnd=&gameType=R&groupBy="
           f"&min=1&minGroupPitches=1&perspective=back&pitchHand=&pitchType=&season={year}&size=small"
           f"&sort=ascending&team=&csv=true")
    return _csv(f"arm_angles/{year}", url, refresh)
