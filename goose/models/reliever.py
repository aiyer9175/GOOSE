"""Relievers: next-season IP, saves and holds from end-of-season role, quality and team context.

Role is the hard part: saves follow who the team trusts in the 9th, which tracks recent
saves/games finished, stuff (K-BB, velocity, whiffs) and whether someone else on the team
already owns the job. Non-appearance counts as 0, so means include role-loss and injury risk.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor

from goose.data.http import ROOT
from goose.models.quantiles import BinnedQuantiles


def is_reliever(s: pd.DataFrame) -> pd.Series:
    """Pitched in relief meaningfully: 8+ relief games, or mostly a reliever with 10+ games."""
    relief_g = s.games.fillna(0) - s.gs.fillna(0)
    active = (relief_g >= 8) | ((s.sp_share.fillna(0) < 0.5) & (s.games.fillna(0) >= 10))
    # Relievers who missed the whole season injured, if they were relievers before.
    # (Look back two seasons: a reliever hurt early in the prior year may have only a few games in it.)
    prior_relief = np.maximum(s.get("relief_g_l1", 0), s.get("relief_g_l2", 0))
    hurt = (s.get("absent", 0) == 1) & (s.il_days.fillna(0) >= 100) & (prior_relief >= 20)
    return active | hurt


def team_role_context(s: pd.DataFrame, exclude_ids: set | None = None) -> pd.DataFrame:
    """Best other save-getter and hold-getter on the player's team (excluding himself)."""
    d = s[s.absent == 0][["mlbam_id", "season", "team", "sv", "hld"]].fillna({"sv": 0, "hld": 0})
    if exclude_ids:
        d = d[~d.mlbam_id.isin(exclude_ids)]
    out = pd.DataFrame(index=s.index, columns=["team_sv_other", "team_hld_other"], dtype=float)
    top = {}
    for (team, season), g in d.groupby(["team", "season"]):
        top[(team, season)] = (g.nlargest(2, "sv")[["mlbam_id", "sv"]].values.tolist(),
                               g.nlargest(2, "hld")[["mlbam_id", "hld"]].values.tolist())
    for i, (pid, team, season) in enumerate(zip(s.mlbam_id, s.team, s.season)):
        svs, hlds = top.get((team, season), ([], []))
        out.iloc[i, 0] = max([v for p, v in svs if p != pid], default=0.0)
        out.iloc[i, 1] = max([v for p, v in hlds if p != pid], default=0.0)
    return out


def build_rp_frame(pt_frame: pd.DataFrame) -> pd.DataFrame:
    """Extends the starter playing-time frame (which has workload, injury, quality lags)."""
    s = pt_frame.copy()
    for c in ("sv", "hld", "gf", "bs"):
        s[c] = s[c].fillna(0)
    s["relief_g"] = s.games.fillna(0) - s.gs.fillna(0)
    s["relief_ip"] = (s.ip - s.sp_outs.fillna(0) / 3).clip(lower=0)
    s["whiff_rate"] = (s.whiffs + 0.25 * 150) / (s.swings + 150)
    key = s.set_index(["mlbam_id", "season"])
    for lag in (1, 2):
        prev = key.reindex(pd.MultiIndex.from_arrays([s.mlbam_id, s.season - lag]))
        for c in ("sv", "hld", "gf", "relief_g"):
            s[f"{c}_l{lag}"] = prev[c].fillna(0).to_numpy()
    nxt = key.reindex(pd.MultiIndex.from_arrays([s.mlbam_id, s.season + 1]))
    s["sv_next"] = nxt.sv.fillna(0).to_numpy()
    s["team_next"] = nxt.team.to_numpy()
    team_sv = s[s.absent.fillna(0) == 0].groupby(["team", "season"]).sv.sum()
    s["team_sv_next"] = team_sv.reindex(pd.MultiIndex.from_arrays([s.team, s.season + 1])).to_numpy()
    s["career_sv"] = s.sort_values("season").groupby("mlbam_id").sv.cumsum().reindex(s.index)
    s["hld_next"] = nxt.hld.fillna(0).to_numpy()
    s = s.join(team_role_context(s))
    # Save-type finishes (last pitcher in a 1-3 run win), season and last 45 days: who held the 9th.
    cp = pd.read_parquet(ROOT / "data" / "features" / "closer_proxy.parquet")
    s = s.merge(cp, on=["mlbam_id", "season"], how="left")
    s[["finish_close", "late_finish_close"]] = s[["finish_close", "late_finish_close"]].fillna(0)
    prev = cp.assign(season=cp.season + 1).rename(columns={"finish_close": "finish_close_l1",
                                                           "late_finish_close": "late_finish_close_l1"})
    s = s.merge(prev, on=["mlbam_id", "season"], how="left").fillna({"finish_close_l1": 0, "late_finish_close_l1": 0})
    return s


FEATURES = [
    "ip", "relief_ip", "games", "relief_g", "gs", "sp_share", "ip_l1", "ip_l2", "sv", "hld", "gf", "bs",
    "sv_l1", "hld_l1", "gf_l1", "relief_g_l1", "sv_l2", "hld_l2", "kbb", "kbb_shrunk", "xwoba_shrunk",
    "fb_velo", "velo_delta", "whiff_rate", "age", "years_since_debut", "il_days", "il_days_l1",
    "il_arm", "il_at_season_end", "options_left", "team_sv_other", "team_hld_other", "on_roster_at_end",
    "absent", "fv_eff", "rehab_late", "milb_gs", "finish_close", "late_finish_close", "finish_close_l1",
]
UP = {"sv", "sv_l1", "sv_l2", "gf", "gf_l1", "kbb_shrunk", "whiff_rate", "finish_close", "late_finish_close"}
DOWN = {"il_days", "il_days_l1", "team_sv_other"}


class RelieverModel:
    QUANTILES = (0.1, 0.5, 0.9)

    def fit(self, frame: pd.DataFrame, last_target: int) -> "RelieverModel":
        tr = frame[(frame.season + 1 <= last_target) & (frame.season + 1 != 2020) & (frame.season >= 2015)]
        tr = tr[is_reliever(tr)]
        X = tr[FEATURES]
        cst = [1 if f in UP else -1 if f in DOWN else 0 for f in FEATURES]
        kw = dict(max_iter=300, learning_rate=0.04, max_leaf_nodes=20, min_samples_leaf=50,
                  l2_regularization=1.0, random_state=0)
        base = HistGradientBoostingRegressor(**kw)
        self.ip = clone(base).fit(X, tr.ip_next)
        self.sv = HistGradientBoostingRegressor(**kw, monotonic_cst=cst).fit(X, tr.sv_next)
        self.hld = clone(base).fit(X, tr.hld_next)
        self.q_ip = BinnedQuantiles(self.QUANTILES, thresholds=(50,)).fit(base, X, tr.ip_next, tr.season)
        self.q_sv = BinnedQuantiles(self.QUANTILES, thresholds=(10, 20)).fit(
            HistGradientBoostingRegressor(**kw, monotonic_cst=cst), X, tr.sv_next, tr.season)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[FEATURES]
        out = pd.DataFrame(index=frame.index)
        out["IP"] = np.clip(self.ip.predict(X), 0, 100)
        out["SV"] = np.clip(self.sv.predict(X), 0, 50)
        out["HLD"] = np.clip(self.hld.predict(X), 0, 40)
        q = self.q_ip.predict(out.IP.to_numpy())
        for qq in self.QUANTILES:
            out[f"IP_p{int(qq * 100)}"] = q[qq].to_numpy()
        qs = self.q_sv.predict(out.SV.to_numpy())
        for qq in self.QUANTILES:
            out[f"SV_p{int(qq * 100)}"] = qs[qq].to_numpy()
        out["P_10SV"], out["P_20SV"] = qs["P_10"].to_numpy(), qs["P_20"].to_numpy()
        return out
