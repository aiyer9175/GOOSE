"""Hitter component rates on the shared empirical-Bayes engine (see goose.models.eb)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from goose.models.eb import EBRates, Spec, lag_frame, park_lookup

# Component definitions: (num, den) expressed over the season columns.
COMPONENTS = {
    "k": ("k", "pa"),
    "bb": ("bb", "pa"),
    "hbp": ("hbp", "pa"),
    "hr": ("hr", "contact"),
    "babip": ("h_nohr", "contact_nohr"),
    "xbh": ("xbh", "h_nohr"),
    "tri": ("triple", "xbh"),
    "sb_att": ("sb_att", "sb_opp"),
    "sb_succ": ("sb", "sb_att"),
}
COVARIATES = {
    "k": ["p_whiff", "p_z_con", "p_chase", "p_bat_speed", "p_br_whiff", "age_c"],
    "bb": ["p_chase", "p_z_swing", "p_br_chase", "p_whiff", "age_c"],
    "hbp": ["age_c"],
    "hr": ["p_barrel", "p_pull_air", "p_ev", "p_hard_hit", "p_bat_speed", "p_gb", "p_pu", "park_hr", "age_c"],
    "babip": ["sprint", "p_ld", "p_gb", "p_pu", "p_hard_hit", "p_ev", "p_pull_air", "bats_l", "park_h", "age_c"],
    "xbh": ["sprint", "p_barrel", "p_ev", "p_pull_air", "p_gb", "age_c"],
    "tri": ["sprint", "age_c"],
    "sb_att": ["sprint", "age_c"],
    "sb_succ": ["sprint", "age_c"],
}
PROCESS = {
    "whiff": ("whiffs", "swings"), "z_con": ("z_con_n", "z_swings"), "chase": ("o_swings", "o_pitches"),
    "z_swing": ("z_swings", "z_pitches"), "barrel": ("barrels", "bbe"), "pull_air": ("pull_air", "bbe"),
    "hard_hit": ("hard_hit", "bbe"), "gb": ("gb", "bbe"), "ld": ("ld", "bbe"), "pu": ("pu", "bbe"),
    "ev": ("ev_sum", "ev_n"), "bat_speed": ("bs_sum", "bs_n"), "br_whiff": ("br_whiffs", "br_swings"),
    "br_chase": ("br_o_swings", "br_o_pitches"),
}
X_EVIDENCE = {"hr": "x_hr", "babip": "x_h_nohr", "xbh": "x_xbh", "tri": "x_3b"}


def pedigree_fade(season, last_listed, debut_year, age):
    """How much prospect pedigree still counts. Clock = years since MLB debut (graduating isn't
    failing), or since last ranked for players who never debuted; first year free. Also fades
    past age 25. A faded former top prospect (e.g. debuted 2023, poor since) keeps ~1/4 from time
    alone; the development term's career-PA decay removes most of the rest."""
    since = np.where(debut_year.notna() & (debut_year <= season), season - debut_year, season - last_listed)
    since = np.nan_to_num(np.asarray(since, dtype=float), nan=0.0)
    return pd.Series(np.exp(-np.maximum(since - 1, 0) / 1.5) * np.exp(-np.maximum(age - 25, 0) / 2.5),
                     index=season.index)


def _extra(frame: pd.DataFrame) -> pd.DataFrame:
    parks = park_lookup(frame)
    # Pedigree is "known" when the player could have appeared on a Board list (first list: 2021)
    # and debuted 2021+; veterans who graduated earlier are unknown, not unranked.
    listed = frame.fv_last.notna()
    known = listed | ((frame.season >= 2021) & (frame.debut_year.fillna(9999) >= 2021))
    # Fade: latest grade (FanGraphs downgrades busts), shrinking with years since last ranked
    # and with age past 25, so a 2021 ranking does nothing for a faded 27-year-old.
    fade = pedigree_fade(frame.season, frame.last_listed, frame.debut_year, frame.age)
    fv = np.where(listed, 45 + (frame.fv_last - 45) * fade, np.where(known, 37.5, np.nan))
    fv = pd.Series(fv, index=frame.index)
    return pd.DataFrame({"sprint": frame.spd_sprint_speed.fillna(27.0) - 27.0,
                         "bats_l": (frame.bats == "L").astype(float),
                         "park_hr": parks.park_hr, "park_h": parks.park_h,
                         # Pedigree (FanGraphs Board): FV relative to 45; unknown before 2021 -> 0.
                         "fv_c": ((fv - 45) / 10).fillna(0.0),
                         "top100": frame.best_rank.notna().astype(float) * fade.where(listed, 0.0),
                         "ped_unknown": (~known).astype(float)}, index=frame.index)


SPEC = Spec(COMPONENTS, COVARIATES, PROCESS, X_EVIDENCE, "stabilization_hitters.csv", _extra)
PED_COV = ["fv_c", "top100", "ped_unknown"]
PED_COMPONENTS = ("k", "bb", "hr", "babip", "xbh")
SPEC_PED = Spec(COMPONENTS,
                {c: v + (PED_COV if c in PED_COMPONENTS else []) for c, v in COVARIATES.items()},
                PROCESS, X_EVIDENCE, "stabilization_hitters.csv", _extra,
                post_covariates=("fv_c", "top100"), career_col="career_total")


class HitterRates(EBRates):
    SPEC = SPEC


class HitterRatesPed(HitterRates):
    """Adds prospect pedigree to the prior and a development term that fades with career PA."""
    SPEC = SPEC_PED

    def fit(self, frame, last_target, max_career_pa=None):
        return super().fit(frame, last_target, max_career_pa, career_col="career_pa_before", volume_col="pa")


def add_derived(s: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Derived count columns for MLB (prefix '') or MiLB (prefix 'milb_aaa_' etc.)."""
    p = prefix
    d = pd.DataFrame(index=s.index)
    h = s[f"{p}h"] if f"{p}h" in s else s[f"{p}single"] + s[f"{p}double"] + s[f"{p}triple"] + s[f"{p}hr"]
    d[f"{p}h"] = h
    d[f"{p}contact"] = s[f"{p}ab"] - s[f"{p}k"] + s[f"{p}sf"]
    d[f"{p}h_nohr"] = h - s[f"{p}hr"]
    d[f"{p}contact_nohr"] = d[f"{p}contact"] - s[f"{p}hr"]
    d[f"{p}xbh"] = s[f"{p}double"] + s[f"{p}triple"]
    singles = h - d[f"{p}xbh"] - s[f"{p}hr"]
    d[f"{p}sb_att"] = s[f"{p}sb"].fillna(0) + s[f"{p}cs"].fillna(0)
    d[f"{p}sb_opp"] = singles + s[f"{p}bb"] + s[f"{p}hbp"]
    for c in ("k", "bb", "hbp", "hr", "pa", "triple", "sb"):
        d[f"{p}{c}"] = s[f"{p}{c}"]
    if not p:
        d["x_hr"], d["x_3b"] = s.x_hr, s.x_3b
        d["x_h_nohr"] = s.x_1b + s.x_2b + s.x_3b
        d["x_xbh"] = s.x_2b + s.x_3b
        d["z_con_n"] = s.z_swings - s.z_whiffs
    return d.clip(lower=0)


def build_frame(seasons: pd.DataFrame) -> pd.DataFrame:
    """One row per player-season t with lagged evidence (t, t-1, t-2), MiLB, covariates and t+1 targets."""
    s = seasons[seasons.primary_position != "P"].copy()
    milb = s.fillna({c: 0 for c in s.columns if c.startswith("milb_")})
    proc_cols = [c for nd in PROCESS.values() for c in nd if c in s and c != "z_con_n"]
    aaasc = [c for c in s.columns if c.startswith("aaasc_")]
    base = pd.concat([s[["mlbam_id", "season"]], add_derived(s), add_derived(milb, "milb_aaa_"),
                      add_derived(milb, "milb_aa_"), s[proc_cols], s[aaasc].fillna(0)], axis=1)
    base = base.loc[:, ~base.columns.duplicated()]
    s["career_total"] = s.career_pa_before.fillna(0) + s.pa.fillna(0)
    for c in ("fv_peak", "fv_last", "best_rank", "last_listed"):
        if c not in s:
            s[c] = np.nan
    s["debut_year"] = pd.to_datetime(s.mlb_debut, errors="coerce").dt.year
    info = s[["mlbam_id", "season", "name", "age", "bats", "spd_sprint_speed", "log_pick", "undrafted",
              "pa", "career_pa_before", "career_total", "pos_group", "team", "fv_peak", "fv_last", "best_rank",
              "last_listed", "debut_year", "absent"]]
    return lag_frame(base, info, COMPONENTS)
