"""Pitcher component rates (per batter faced) on the shared empirical-Bayes engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

from goose.models.eb import EBRates, Spec, lag_frame, park_lookup

COMPONENTS = {
    "k": ("k", "bf"),
    "bb": ("bb", "bf"),
    "hbp": ("hbp", "bf"),
    "hr": ("hr", "contact"),
    "babip": ("h_nohr", "contact_nohr"),
    "xbh": ("xbh", "h_nohr"),
}
PROCESS = {
    "whiff": ("whiffs", "swings"), "csw": ("csw_n", "pitches"), "chase": ("o_swings", "o_pitches"),
    "zone": ("z_pitches", "pitches"), "z_con": ("z_con_n", "z_swings"), "fb_velo": ("fbv_sum", "fbv_n"),
    "barrel": ("barrels", "bbe"), "hard_hit": ("hard_hit", "bbe"), "gb": ("gb", "bbe"), "ev": ("ev_sum", "ev_n"),
    "br_whiff": ("br_whiffs", "br_swings"), "os_whiff": ("os_whiffs", "os_swings"),
}
COVARIATES = {
    "k": ["p_whiff", "p_csw", "p_chase", "p_fb_velo", "p_br_whiff", "p_os_whiff", "p_z_con", "sp_share", "age_c"],
    "bb": ["p_zone", "p_chase", "p_csw", "sp_share", "age_c"],
    "hbp": ["p_zone", "age_c"],
    "hr": ["p_gb", "p_barrel", "p_hard_hit", "p_ev", "p_fb_velo", "p_whiff", "sp_share", "age_c"],
    "babip": ["p_gb", "p_hard_hit", "p_ev", "p_whiff", "age_c"],
    "xbh": ["p_gb", "p_ev", "p_barrel", "age_c"],
}


def _extra(frame: pd.DataFrame) -> pd.DataFrame:
    parks = park_lookup(frame)
    return pd.DataFrame({"sp_share": frame.sp_share.fillna(0), "park_hr": parks.park_hr,
                         "park_h": parks.park_h}, index=frame.index)


SPEC = Spec(COMPONENTS, COVARIATES, PROCESS, {"hr": "x_hr", "babip": "x_h_nohr", "xbh": "x_xbh"},
            "stabilization_pitchers.csv", _extra)


class PitcherRates(EBRates):
    SPEC = SPEC

    def fit(self, frame, last_target, max_career=None):
        return super().fit(frame, last_target, max_career, career_col="career_bf_before", volume_col="bf")


def _derived(s: pd.DataFrame, p: str = "") -> pd.DataFrame:
    d = pd.DataFrame(index=s.index)
    h = s[f"{p}h"] if f"{p}h" in s else s[f"{p}single"] + s[f"{p}double"] + s[f"{p}triple"] + s[f"{p}hr"]
    d[f"{p}contact"] = s[f"{p}bf"] - s[f"{p}k"] - s[f"{p}bb"] - s[f"{p}hbp"]
    d[f"{p}h_nohr"] = h - s[f"{p}hr"]
    d[f"{p}contact_nohr"] = d[f"{p}contact"] - s[f"{p}hr"]
    for c in ("k", "bb", "hbp", "hr", "bf"):
        d[f"{p}{c}"] = s[f"{p}{c}"]
    if not p:
        d["xbh"] = s.double + s.triple
        d["x_hr"] = s.x_hr
        d["x_h_nohr"] = s.x_1b + s.x_2b + s.x_3b
        d["x_xbh"] = s.x_2b + s.x_3b
        d["csw_n"] = s.called_strikes + s.whiffs
        d["z_con_n"] = s.z_swings - s.z_whiffs
    return d.clip(lower=0)


def build_frame(seasons: pd.DataFrame) -> pd.DataFrame:
    s = seasons.copy()
    milb = s.fillna({c: 0 for c in s.columns if c.startswith("milb_")})
    proc_cols = [c for nd in PROCESS.values() for c in nd if c in s and c not in ("csw_n", "z_con_n")]
    base = pd.concat([s[["mlbam_id", "season"]], _derived(s), _derived(milb, "milb_aaa_"),
                      _derived(milb, "milb_aa_"), s[proc_cols]], axis=1)
    base = base.loc[:, ~base.columns.duplicated()]
    info = s[["mlbam_id", "season", "name", "age", "throws", "log_pick", "undrafted", "bf", "ip", "gs",
              "sp_share", "career_bf_before", "team", "absent"]]
    return lag_frame(base, info, COMPONENTS)
