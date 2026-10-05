"""Hitter playing time: next-season MLB PA from end-of-season context.

Gradient-boosted mean model plus quantile models (P10/P50/P90). Non-appearance counts as 0 PA,
so the mean already includes the risk of losing the job, getting hurt, or not being in MLB.
2020 is excluded as a target and down-weighted as a feature season.
"""
from __future__ import annotations

from goose.data.http import ROOT as ROOT_40

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.base import clone

from goose.models.quantiles import BinnedQuantiles, HealthyBlend

POS = ["C", "CI", "MI", "OF", "DH"]
# Lineup slots per position group, for the "opportunity" feature.
SLOTS = {"C": ("C", 1), "CI": ("IF", 4), "MI": ("IF", 4), "OF": ("OF", 3), "DH": ("IF", 4)}


def position_depth(s: pd.DataFrame, exclude_ids: set | None = None) -> pd.DataFrame:
    """Rivals at the player's own primary position on his team: how many teammates there had 250+ PA,
    and the best rival's quality (3-yr xwOBA). A shortstop with no real rival should play more."""
    d = s[(s.absent == 0) & (s.pa >= 250)][["mlbam_id", "season", "team", "primary_position", "xwoba_3y"]]
    if exclude_ids:
        d = d[~d.mlbam_id.isin(exclude_ids)]
    grp = d.groupby(["team", "season", "primary_position"])
    lookup = {k: (g.mlbam_id.tolist(), g.xwoba_3y.tolist()) for k, g in grp}
    n_r, best = [], []
    for pid, team, season, pos in zip(s.mlbam_id, s.team, s.season, s.primary_position):
        ids, xw = lookup.get((team, season, pos), ([], []))
        rivals = [w for i, w in zip(ids, xw) if i != pid]
        n_r.append(len(rivals))
        best.append(max(rivals) if rivals else 0.28)
    return pd.DataFrame({"pos_rivals": n_r, "pos_rival_best": best}, index=s.index)


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
    s["bats_l"] = (s.bats == "L").astype(int)
    _r40 = pd.read_parquet(ROOT_40 / "data" / "processed" / "roster40_end.parquet")
    s["on_40man"] = pd.MultiIndex.from_arrays([s.mlbam_id, s.season]).isin(
        pd.MultiIndex.from_arrays([_r40.mlbam_id, _r40.season])).astype(int)
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
    s = s.join(position_depth(s))
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
    "rehab_n", "rehab_late", "rehab_month", "bats_l", "pos_rivals", "pos_rival_best", "on_40man",
] + [f"pos_{p}" for p in POS]
BLEND_Z = ["xwoba_3y", "full_seasons_3y", "best_pa_3y"]


def _blend_z(frame):
    talent = ((frame.xwoba_3y.fillna(0.31) - 0.32) / 0.03).to_numpy(float)
    shortened = ((frame.full_seasons_3y.fillna(0) >= 1) & (frame.pa.fillna(0) < 450)).to_numpy(float)
    return np.column_stack([talent, frame.full_seasons_3y.fillna(0).to_numpy(float),
                            (frame.best_pa_3y.fillna(0) / 600).to_numpy(float),
                            np.maximum(talent, 0) * shortened])


def pa_baseline(s: pd.DataFrame) -> np.ndarray:
    """Most recent healthy full season (last 3 years; <30 IL days and 500+ PA); fallback: the higher of
    last season's PA and the average of the two best of the last three."""
    out = np.full(len(s), np.nan)
    for pa, il in (("pa", "il_days"), ("pa_l1", "il_days_l1"), ("pa_l2", "il_days_l2")):
        ok = np.isnan(out) & (s[il].fillna(0).to_numpy() < 30) & (s[pa].fillna(0).to_numpy() >= 500)
        out[ok] = s[pa].to_numpy()[ok]
    pas = s[["pa", "pa_l1", "pa_l2"]].fillna(0).to_numpy()
    return np.where(np.isnan(out), np.maximum(s.pa.fillna(0).to_numpy(), np.sort(pas, axis=1)[:, -2:].mean(axis=1)), out)


def full_time_hitter(s: pd.DataFrame) -> pd.Series:
    """500+ PA in one of the last 3 seasons, and didn't end the season on a long IL stint."""
    full = (s[["pa", "pa_l1", "pa_l2"]].fillna(0) >= 500).sum(axis=1) >= 1
    hurt = (s.il_at_season_end == 1) & (s.il_end_spell_days.fillna(s.il_days) >= 60)
    return full & ~hurt & (s.absent.fillna(0) == 0)


def _pa_gap_x(base, age, talent):
    return np.column_stack([np.ones(len(base)), np.clip(age - 30, -6, 10), np.clip(base - 600, -150, 150) / 50,
                            np.clip(np.nan_to_num((talent - 0.32) / 0.03), -3, 3)])


def established_regular(s: pd.DataFrame) -> pd.Series:
    """A full-time hitter in at least one of the last two seasons (500+ PA)."""
    return (s[["pa", "pa_l1"]].fillna(0).max(axis=1) >= 500) & (s.absent.fillna(0) == 0)


def _regular_x(s: pd.DataFrame) -> np.ndarray:
    age = (s.age - 30).clip(-6, 10)
    return np.column_stack([
        np.ones(len(s)), s.pa.fillna(0) / 600, s.pa_l1.fillna(0) / 600, s.pa_l2.fillna(0) / 600,
        s.full_seasons_3y.fillna(0), (s.xwoba_3y.fillna(.31) - .32) / .03, age, age.clip(lower=0) ** 2 / 10,
        s.il_days.fillna(0) / 100, s.il_at_season_end.fillna(0), s.start_share.fillna(0),
        s.late_start_share.fillna(0), s.pos_C.fillna(0), s.sv_fielding_run_value.fillna(0) / 10])


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
        # Established regulars: average with a ridge model trained only on them. The boosted model
        # over-reacts to age plus a late-season dip (e.g. 34-36 year-olds after two 500-PA seasons
        # averaged 484 PA the next year).
        er = established_regular(tr).to_numpy()
        Xr, yr = _regular_x(tr[er]), y.to_numpy()[er]
        self.reg_b = np.linalg.solve(Xr.T @ Xr + 3 * np.eye(Xr.shape[1]), Xr.T @ yr)
        seasons_r = tr.season.to_numpy()[er]
        oof_r = np.empty(er.sum())
        for season in np.unique(seasons_r):
            te = seasons_r == season
            b = np.linalg.solve(Xr[~te].T @ Xr[~te] + 3 * np.eye(Xr.shape[1]), Xr[~te].T @ yr[~te])
            oof_r[te] = np.clip(Xr[te] @ b, 0, 720)
        final[er] = 0.5 * final[er] + 0.5 * oof_r
        self.q = BinnedQuantiles(self.QUANTILES, thresholds=(300,)).fit_oof(final, y, tr.established)

        # --- Workload thesis for full-time hitters (same structure as starters). Healthy hump:
        # recent healthy full season + gap (age, workload size, quality); injury history only moves
        # P(hump) (backtest: injury-prone hitters on the hump had the same -26 PA gap as everyone).
        ft = full_time_hitter(tr).to_numpy()
        base = pa_baseline(tr[ft])
        yf = y.to_numpy()[ft]
        hump = yf >= 400            # "kept a full-time job" (for P(full season) and the low hump)
        full = yf >= 500            # a genuinely healthy full-time season sets the center
        A = _pa_gap_x(base[full], tr.age.to_numpy()[ft][full], tr.xwoba_3y.to_numpy()[ft][full])
        self.pa_gap_b = np.linalg.lstsq(A, yf[full] - base[full], rcond=None)[0]
        self.pa_hump_sd = float(np.std(yf[hump] - (base[hump] + _pa_gap_x(
            base[hump], tr.age.to_numpy()[ft][hump], tr.xwoba_3y.to_numpy()[ft][hump]) @ self.pa_gap_b)))
        self.pa_low_q = np.quantile(yf[~hump], np.linspace(0, 1, 101))
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.pa_p_hump = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.04, max_leaf_nodes=12, min_samples_leaf=60, l2_regularization=1.0,
            random_state=0).fit(X[ft], hump.astype(int))
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[FEATURES]
        out = pd.DataFrame(index=frame.index)
        E = np.clip(self.mean.predict(X), 0, 720)
        H = np.maximum(np.clip(self.healthy.predict(X), 0, 720), E)
        w = self.blend.weight(_blend_z(frame))
        out["PA"] = E + w * (H - E)
        er = established_regular(frame).to_numpy()
        if er.any():
            out.loc[er, "PA"] = 0.5 * out.loc[er, "PA"] + 0.5 * np.clip(_regular_x(frame[er]) @ self.reg_b, 0, 720)
        out["PA_expected"], out["PA_healthy"], out["healthy_weight"] = E, H, w
        q = self.q.predict(out.PA.to_numpy(), frame.established.to_numpy())
        for qq in self.QUANTILES:
            out[f"PA_p{int(qq * 100)}"] = q[qq].to_numpy()
        out["P_300PA"] = q["P_300"].to_numpy()
        out["P_full_season"] = np.nan

        ft = full_time_hitter(frame).to_numpy()
        if ft.any():
            base = pa_baseline(frame[ft])
            center = np.clip(base + _pa_gap_x(base, frame.age.to_numpy()[ft], frame.xwoba_3y.to_numpy()[ft]) @ self.pa_gap_b,
                             0, 720)
            p = self.pa_p_hump.predict_proba(X[ft])[:, 1]
            expected = p * center + (1 - p) * float(np.mean(self.pa_low_q))
            # Headline = full-time workload when keeping the job is likely; blends smoothly toward the
            # expected value as that chance falls from 50% to 30% (no cliff at a coin flip).
            ramp = np.clip((p - 0.3) / 0.2, 0, 1)
            out.loc[ft, "PA"] = expected + ramp * (center - expected)
            out.loc[ft, "P_full_season"] = p
            from scipy.stats import norm
            qs = np.linspace(0.005, 0.995, 200)
            rng = []
            for c, pp in zip(center, p):
                hi = np.clip(norm.ppf(qs, c, self.pa_hump_sd), 400, 720)
                lo = np.interp(qs, np.linspace(0, 1, 101), self.pa_low_q)
                vals = np.r_[hi, lo]; w = np.r_[np.full(200, pp / 200), np.full(200, (1 - pp) / 200)]
                o = np.argsort(vals); cw = np.cumsum(w[o])
                rng.append([vals[o][np.searchsorted(cw, x)] for x in (0.1, 0.5, 0.9)])
            out.loc[ft, ["PA_p10", "PA_p50", "PA_p90"]] = np.array(rng)
            out.loc[ft, "P_300PA"] = p + (1 - p) * float(np.mean(self.pa_low_q >= 300))
        return out
