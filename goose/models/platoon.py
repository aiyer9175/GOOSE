"""Platoon role and matchup exposure for left-handed hitters.

Findings that shape this (2015-2026 backtests):
- Individual LHH platoon splits barely persist (year-to-year r ~ 0.04): regress ~4,000 PA vs LHP
  toward the league LHH split (~28 wOBA points), i.e. nearly everyone gets the league split.
- What varies is the team's choice: the share of starts vs LHSP (relative to the team's LHSP
  share). PA share vs LHP = 0.084 + 0.234 x that ratio.
- A model of next year's ratio (history, experience, defense, position, quality) beats "same as
  last year" (MAE 0.170 vs 0.194; rookies 0.206 vs 0.250). Rookie lefties are platooned more.

Projected overall rates are re-weighted for the expected exposure; each LHH gets a role label and a
projection vs RHP and vs LHP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from goose.data.http import ROOT

FEAT = ROOT / "data" / "features"
SPLIT_K = 4000.0          # PA vs LHP of league split added when shrinking an individual split
EXPO_A, EXPO_B = 0.084, 0.234
F = ["platoon_ratio", "started", "career", "age", "sv_fielding_run_value", "pa", "pal_share", "pos_prem", "woba"]


def _frame() -> pd.DataFrame:
    s = pd.read_parquet(FEAT / "hitter_seasons.parquet")
    sp = pd.read_parquet(FEAT / "hitter_splits.parquet")
    w = sp.pivot_table(index=["mlbam_id", "season"], columns="p_throws", values=["pa", "woba_num", "woba_den"],
                       aggfunc="sum").fillna(0)
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    s = s.merge(w.reset_index(), on=["mlbam_id", "season"], how="left")
    s["pal_share"] = s.pa_L / (s.pa_L + s.pa_R).clip(lower=1)
    s["career"] = s.career_pa_before.fillna(0) + s.pa.fillna(0)
    s["pos_prem"] = s.primary_position.isin(["C", "SS", "CF"]).astype(int)
    s["woba"] = s.woba_num / s.woba_den.clip(lower=1)
    s["platoon_ratio"] = s.platoon_ratio.where(s.started >= 10)
    key = s.set_index(["mlbam_id", "season"])
    nxt = key.reindex(pd.MultiIndex.from_arrays([s.mlbam_id, s.season + 1]))
    s["ratio_next"] = nxt.platoon_ratio.to_numpy()
    s["starts_next"] = nxt.started.to_numpy()
    return s


class Platoon:
    def fit(self, last_target: int) -> "Platoon":
        s = _frame()
        d = s[(s.bats == "L") & (s.absent == 0) & (s.starts_next >= 60) & s.season.between(2015, last_target - 1)
              & ~s.season.isin([2019, 2020])].dropna(subset=["ratio_next"])
        self.m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=12,
                                               min_samples_leaf=40, random_state=0).fit(d[F], d.ratio_next.clip(0, 2))
        sp = pd.read_parquet(FEAT / "hitter_splits.parquet")
        lg = sp[sp.season <= last_target - 1].groupby(["stand", "p_throws"])[["woba_num", "woba_den"]].sum()
        self.lg_split = float(lg.loc[("L", "R")].woba_num / lg.loc[("L", "R")].woba_den
                              - lg.loc[("L", "L")].woba_num / lg.loc[("L", "L")].woba_den)
        self.s = s
        return self

    def project(self, ids: pd.Series, season: int) -> pd.DataFrame:
        """Per player: role, predicted start ratio vs LHSP, PA share vs LHP (past and next), split."""
        s = self.s
        cur = s[s.season == season].set_index("mlbam_id").reindex(ids.to_numpy())
        out = pd.DataFrame(index=ids.index)
        lhh = (cur.bats == "L").to_numpy()
        ratio = np.where(lhh, np.clip(self.m.predict(cur[F]), 0, 1.3), np.nan)
        out["platoon_ratio_next"] = ratio
        out["pal_share_next"] = np.where(lhh, EXPO_A + EXPO_B * ratio, np.nan)
        # Past exposure: last two seasons' actual PA share vs LHP (weighted), else the rookie norm.
        prev = s[s.season == season - 1].set_index("mlbam_id").reindex(ids.to_numpy())
        num = cur.pa_L.fillna(0).to_numpy() + 0.6 * prev.pa_L.fillna(0).to_numpy()
        den = (cur.pa_L.fillna(0) + cur.pa_R.fillna(0)).to_numpy() + 0.6 * (prev.pa_L.fillna(0) + prev.pa_R.fillna(0)).to_numpy()
        out["pal_share_past"] = np.where(den >= 50, num / np.maximum(den, 1), out.pal_share_next)
        # Individual split, shrunk hard toward the league LHH split.
        def career(col):
            g = s[s.season <= season].groupby("mlbam_id")[col].sum()
            return g.reindex(ids.to_numpy()).fillna(0).to_numpy()
        paL = career("pa_L")
        own = career("woba_num_R") / np.maximum(career("woba_den_R"), 1) - career("woba_num_L") / np.maximum(career("woba_den_L"), 1)
        out["split"] = np.where(lhh, (paL * np.nan_to_num(own) + SPLIT_K * self.lg_split) / (paL + SPLIT_K), np.nan)
        out["platoon_role"] = np.select([~lhh, ratio < 0.35, ratio < 0.75], ["Plays vs both", "Strict platoon",
                                                                               "Soft platoon"], "Everyday")
        return out
