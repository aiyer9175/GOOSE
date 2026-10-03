"""Hitter playing time: next-season MLB PA from end-of-season context.

Gradient-boosted mean model plus quantile models (P10/P50/P90). Non-appearance counts as 0 PA,
so the mean already includes the risk of losing the job, getting hurt, or not being in MLB.
2020 is excluded as a target and down-weighted as a feature season.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.base import clone

from goose.models.quantiles import BinnedQuantiles, HealthyBlend

POS = ["C", "CI", "MI", "OF", "DH"]
# Lineup slots per position group, for the "opportunity" feature.
SLOTS = {"C": ("C", 1), "CI": ("IF", 4), "MI": ("IF", 4), "OF": ("OF", 3), "DH": ("IF", 4)}


def opportunity(s: pd.DataFrame, exclude_ids: set | None = None) -> pd.DataFrame:
    """Weakest incumbent at the player's position group on his team (excluding himself).

    For each team-season, take the top-N MLB hitters by PA in the group (N = slots) and their
    shrunk wOBA; the hole is the weakest of them (empty slots count as replacement, .290).
    exclude_ids drops departing players (e.g. 2027 free agents) from the incumbents.
    """
    d = s[s.absent == 0][["mlbam_id", "season", "team", "pos_group", "pa", "woba_shrunk"]].copy()
    if exclude_ids:
        d = d[~d.mlbam_id.isin(exclude_ids)]
    d["grp"] = d.pos_group.map(lambda g: SLOTS[g][0])
    out = []
    for (team, season, grp), g in d.groupby(["team", "season", "grp"]):
        n = {"C": 1, "IF": 4, "OF": 3}[grp]
        g = g.sort_values("pa", ascending=False)
        out.append((team, season, grp, g.mlbam_id.tolist()[:n + 1], g.woba_shrunk.tolist()[:n + 1], n))
    lookup = {(t, y, gr): (ids, wb, n) for t, y, gr, ids, wb, n in out}
    hole, inc_pa = [], []
    for pid, team, season, pg in zip(s.mlbam_id, s.team, s.season, s.pos_group):
        grp = SLOTS.get(pg, ("IF", 4))[0]
        ids, wb, n = lookup.get((team, season, grp), ([], [], SLOTS.get(pg, ("IF", 4))[1]))
        others = [w for i, w in zip(ids, wb) if i != pid][:n]
        others += [0.290] * (n - len(others))
        hole.append(min(others))
    return pd.DataFrame({"hole_woba": hole}, index=s.index)


def build_pt_frame(seasons: pd.DataFrame) -> pd.DataFrame:
    s = seasons[seasons.primary_position != "P"].copy()
    s["absent"] = s.absent.fillna(0)
    s["woba"] = s.woba_num / s.woba_den.replace(0, np.nan)
    s["xwoba"] = s.xwoba_num / s.woba_den.replace(0, np.nan)
    # Shrunk talent proxies (k from stabilization: xwOBA ~190 PA, wOBA heavier).
    s["woba_shrunk"] = (s.woba_num + 0.310 * 300) / (s.woba_den + 300)
    s["xwoba_shrunk"] = (s.xwoba_num + 0.310 * 190) / (s.woba_den + 190)
    s["k_rate"] = s.k / s.pa.replace(0, np.nan)
    scale = np.where(s.season == 2020, 162 / 60, 1.0)
    for c in ("pa", "started", "games", "il_days"):
        s[c] = s[c] * scale
    s = s.sort_values(["mlbam_id", "season"])
    key = s.set_index(["mlbam_id", "season"])
    for lag in (1, 2):
        prev = key.reindex(pd.MultiIndex.from_arrays([s.mlbam_id, s.season - lag]))
        s[f"pa_l{lag}"] = prev.pa.to_numpy()
        s[f"start_share_l{lag}"] = prev.start_share.to_numpy()
        s[f"il_days_l{lag}"] = prev.il_days.to_numpy()
        s[f"woba_num_l{lag}"] = prev.woba_num.to_numpy()
        s[f"xwoba_num_l{lag}"] = prev.xwoba_num.to_numpy()
        s[f"woba_den_l{lag}"] = prev.woba_den.to_numpy()
    # Track record: one injured season shouldn't erase years as a regular.
    pas = s[["pa", "pa_l1", "pa_l2"]].fillna(0)
    s["best_pa_3y"] = pas.max(axis=1)
    s["full_seasons_3y"] = (pas >= 500).sum(axis=1)
    s["start_share_3y"] = s[["start_share", "start_share_l1", "start_share_l2"]].mean(axis=1)
    w = (1.0, 0.6, 0.35)
    num = sum(wi * s[c].fillna(0) for wi, c in zip(w, ["xwoba_num", "xwoba_num_l1", "xwoba_num_l2"]))
    den = sum(wi * s[c].fillna(0) for wi, c in zip(w, ["woba_den", "woba_den_l1", "woba_den_l2"]))
    s["xwoba_3y"] = (num + 0.310 * 190) / (den + 190)
    num = sum(wi * s[c].fillna(0) for wi, c in zip(w, ["woba_num", "woba_num_l1", "woba_num_l2"]))
    s["woba_3y"] = (num + 0.310 * 300) / (den + 300)
    s["established"] = (s.full_seasons_3y >= 1).astype(int)
    # Pedigree (faded) and the opening at his position.
    from goose.models.hitter_rates import pedigree_fade
    fade = pedigree_fade(s.season, s.last_listed, pd.to_datetime(s.mlb_debut, errors="coerce").dt.year, s.age)
    # Centered faded FV (0 = 45 FV or unknown) plus a flag for whether Board data exists.
    s["fv_eff"] = np.where(s.fv_last.notna(), (s.fv_last - 45) * fade, 0.0)
    s["fv_known"] = (s.season >= 2021).astype(int)
    s["top100_eff"] = np.where(s.best_rank.notna(), fade, 0.0)
    s["milb_aaa"] = s.get("milb_aaa_pa", 0).fillna(0)
    s["milb_aa"] = s.get("milb_aa_pa", 0).fillna(0)
    s["hole_woba"] = opportunity(s).hole_woba
    nxt = key.reindex(pd.MultiIndex.from_arrays([s.mlbam_id, s.season + 1]))
    s["pa_next"] = nxt.pa.fillna(0).to_numpy()
    il_next = s.set_index(["mlbam_id", "season"]).il_days
    s["il_days_next"] = il_next.reindex(pd.MultiIndex.from_arrays([s.mlbam_id, s.season + 1])).to_numpy()
    for p in POS:
        s[f"pos_{p}"] = (s.pos_group == p).astype(int)
    s["milb_pa"] = s.get("milb_aaa_pa", 0).fillna(0) + s.get("milb_aa_pa", 0).fillna(0)
    return s


FEATURES = [
    "pa", "pa_l1", "pa_l2", "career_pa_before", "games", "start_share", "late_start_share",
    "start_share_l1", "on_roster_at_end", "lineup_slot", "vs_l_share", "platoon_ratio",
    "age", "years_since_debut", "woba", "xwoba", "woba_shrunk", "xwoba_shrunk", "k_rate",
    "spd_sprint_speed", "sv_fielding_run_value", "sv_n_outs_above_average",
    "il_days", "il_days_l1", "il_days_l2", "il_stints", "il_at_season_end", "il_surgery",
    "options_left", "prospect_threat", "milb_pa", "log_pick", "undrafted",
    "best_pa_3y", "full_seasons_3y", "start_share_3y", "xwoba_3y", "woba_3y", "absent",
    "fv_eff", "fv_known", "top100_eff", "milb_aaa", "milb_aa", "hole_woba",
    "rehab_n", "rehab_late", "rehab_month",
] + [f"pos_{p}" for p in POS]
BLEND_Z = ["xwoba_3y", "full_seasons_3y", "best_pa_3y"]


def _blend_z(frame):
    talent = ((frame.xwoba_3y.fillna(0.31) - 0.32) / 0.03).to_numpy(float)
    shortened = ((frame.full_seasons_3y.fillna(0) >= 1) & (frame.pa.fillna(0) < 450)).to_numpy(float)
    return np.column_stack([talent, frame.full_seasons_3y.fillna(0).to_numpy(float),
                            (frame.best_pa_3y.fillna(0) / 600).to_numpy(float),
                            np.maximum(talent, 0) * shortened])


class HitterPlayingTime:
    QUANTILES = (0.1, 0.5, 0.9)

    def fit(self, frame: pd.DataFrame, last_target: int) -> "HitterPlayingTime":
        tr = frame[(frame.season + 1 <= last_target) & (frame.season + 1 != 2020) & (frame.season >= 2015)]
        tr = tr[(tr.pa >= 1) | (tr.absent == 1)]
        X, y = tr[FEATURES], tr.pa_next
        kw = dict(max_iter=400, learning_rate=0.04, max_leaf_nodes=24, min_samples_leaf=40,
                  l2_regularization=1.0, random_state=0)
        base = HistGradientBoostingRegressor(loss="squared_error", **kw)
        # "If healthy" = no IL time and in the majors next season (not: absent from any roster).
        h = ((tr.il_days_next.fillna(0) == 0) & (y > 0)).to_numpy()
        self.mean = clone(base).fit(X, y)
        self.healthy = clone(base).fit(X[h], y[h])
        # Out-of-fold E and H (folds = seasons) to learn the blend and calibrate ranges.
        E, H = np.empty(len(y)), np.empty(len(y))
        for season in tr.season.unique():
            te = (tr.season == season).to_numpy()
            E[te] = clone(base).fit(X[~te], y[~te]).predict(X[te])
            H[te] = clone(base).fit(X[~te & h], y[~te & h]).predict(X[te])
        H = np.maximum(H, E)
        self.blend = HealthyBlend().fit(E, H, y.to_numpy(), _blend_z(tr))
        final = E + self.blend.weight(_blend_z(tr)) * (H - E)
        self.q = BinnedQuantiles(self.QUANTILES, thresholds=(300,)).fit_oof(final, y, tr.established)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[FEATURES]
        out = pd.DataFrame(index=frame.index)
        E = np.clip(self.mean.predict(X), 0, 720)
        H = np.maximum(np.clip(self.healthy.predict(X), 0, 720), E)
        w = self.blend.weight(_blend_z(frame))
        out["PA"] = E + w * (H - E)
        out["PA_expected"], out["PA_healthy"], out["healthy_weight"] = E, H, w
        q = self.q.predict(out.PA.to_numpy(), frame.established.to_numpy())
        for qq in self.QUANTILES:
            out[f"PA_p{int(qq * 100)}"] = q[qq].to_numpy()
        out["P_300PA"] = q["P_300"].to_numpy()
        return out
