"""Starting-pitcher playing time: next-season MLB IP and GS from end-of-season context."""
from __future__ import annotations

from goose.data.http import ROOT as ROOT_40

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.base import clone

from goose.models.quantiles import BinnedQuantiles, HealthyBlend


def build_pt_frame(seasons: pd.DataFrame) -> pd.DataFrame:
    s = seasons.copy()
    s["kbb"] = (s.k - s.bb) / s.bf.replace(0, np.nan)
    s["kbb_shrunk"] = (s.k - s.bb + 0.13 * 200) / (s.bf + 200)
    s["xwoba_shrunk"] = (s.xwoba_num + 0.315 * 330) / (s.woba_den + 330)
    s["fb_velo"] = s.fbv_sum / s.fbv_n.replace(0, np.nan)
    s["milb_gs"] = s.get("milb_aaa_gs", 0).fillna(0) + s.get("milb_aa_gs", 0).fillna(0)
    scale = np.where(s.season == 2020, 162 / 60, 1.0)
    for c in ("ip", "gs", "total_ip", "il_days"):
        s[c] = s[c] * scale
    key = s.set_index(["mlbam_id", "season"])
    for lag in (1, 2, 3, 4):
        prev = key.reindex(pd.MultiIndex.from_arrays([s.mlbam_id, s.season - lag]))
        for c in ("ip", "gs", "total_ip", "il_days", "il_arm", "il_surgery"):
            s[f"{c}_l{lag}"] = prev[c].to_numpy()
        s[f"fb_velo_l{lag}"] = prev.fb_velo.to_numpy()
    s["velo_delta"] = s.fb_velo - s.fb_velo_l1
    _r40 = pd.read_parquet(ROOT_40 / "data" / "processed" / "roster40_end.parquet")
    s["on_40man"] = pd.MultiIndex.from_arrays([s.mlbam_id, s.season]).isin(
        pd.MultiIndex.from_arrays([_r40.mlbam_id, _r40.season])).astype(int)
    from goose.data.http import ROOT as _ROOT
    tw = pd.read_parquet(_ROOT / "data" / "features" / "team_wpct.parquet").set_index(["team", "season"]).team_wpct
    s["team_wpct"] = tw.reindex(pd.MultiIndex.from_arrays([s.team, s.season])).to_numpy()
    from goose.features.pedigree import pedigree_by_season
    from goose.models.hitter_rates import pedigree_fade
    ped = pedigree_by_season().set_index(["mlbam_id", "season"])
    pk = pd.MultiIndex.from_arrays([s.mlbam_id, s.season])
    fv_last, last_listed, rank = (ped[c].reindex(pk).to_numpy() for c in ("fv_last", "last_listed", "best_rank"))
    fade = pedigree_fade(s.season, pd.Series(last_listed, index=s.index),
                         pd.to_datetime(s.mlb_debut, errors="coerce").dt.year, s.age)
    s["fv_eff"] = np.where(np.isfinite(fv_last), (fv_last - 45) * fade, 0.0)
    s["top100_eff"] = np.where(np.isfinite(rank), fade, 0.0)
    s["fv_known"] = (s.season >= 2021).astype(int)
    # Durability over five seasons: how often he has carried a full starter's workload.
    ip5 = s[["ip", "ip_l1", "ip_l2", "ip_l3", "ip_l4"]].fillna(0)
    s["seasons_130_5y"] = (ip5 >= 130).sum(axis=1)
    s["ip_5y_mean"] = ip5.mean(axis=1)
    s["il_days_5y"] = s[["il_days", "il_days_l1", "il_days_l2", "il_days_l3", "il_days_l4"]].fillna(0).sum(axis=1)
    ips = s[["ip", "ip_l1", "ip_l2"]].fillna(0)
    s["best_ip_3y"] = ips.max(axis=1)
    s["full_seasons_3y"] = (ips >= 120).sum(axis=1)
    s["established"] = (s.full_seasons_3y >= 1).astype(int)
    s["max_total_ip_3y"] = s[["total_ip", "total_ip_l1", "total_ip_l2"]].max(axis=1)
    nxt = key.reindex(pd.MultiIndex.from_arrays([s.mlbam_id, s.season + 1]))
    s["ip_next"] = nxt.ip.fillna(0).to_numpy()
    s["il_days_next"] = nxt.il_days.to_numpy()
    s["absent"] = s.absent.fillna(0)
    s["gs_next"] = nxt.gs.fillna(0).to_numpy()
    return s


FEATURES = [
    "ip", "gs", "games", "sp_share", "total_ip", "ip_l1", "gs_l1", "total_ip_l1", "ip_l2", "total_ip_l2",
    "max_total_ip_3y", "ip_per_gs", "pitches_per_gs", "late_gs", "late_outs", "on_roster_at_end",
    "il_days", "il_days_l1", "il_days_l2", "il_stints", "il_arm", "il_shoulder", "il_surgery",
    "il_at_season_end", "il_arm_l1", "il_surgery_l1", "age", "years_since_debut", "career_gs_before",
    "options_left", "kbb", "kbb_shrunk", "xwoba_shrunk", "fb_velo", "velo_delta", "milb_gs",
    "log_pick", "undrafted", "absent", "seasons_130_5y", "ip_5y_mean", "il_days_5y",
    "fv_eff", "top100_eff", "fv_known", "rehab_n", "rehab_late", "rehab_month", "on_40man",
]


def _blend_z(frame):
    return np.column_stack([((frame.kbb_shrunk.fillna(0.13) - 0.13) / 0.04).to_numpy(float),
                            frame.seasons_130_5y.fillna(0).to_numpy(float),
                            (frame.ip_5y_mean.fillna(0) / 100).to_numpy(float)])


def returning_starter(s: pd.DataFrame) -> pd.Series:
    """Established starters (100+ IP in one of the prior 3 seasons) who missed most of the season
    and were still out at season's end (activated before the end = finished healthy)."""
    prior = s[["ip_l1", "ip_l2", "ip_l3"]].fillna(0).max(axis=1)
    still_out = (s.il_at_season_end == 1) | (s.absent == 1)
    return (s.il_days >= 120) & (prior >= 100) & still_out


def _returning_x(s: pd.DataFrame) -> np.ndarray:
    prior = s[["ip_l1", "ip_l2", "ip_l3"]].fillna(0).max(axis=1)
    # Talent: best recent K-BB% (the current-season sample is tiny for these pitchers).
    talent = s[["kbb_shrunk"]].fillna(0.13).to_numpy().ravel()
    velo = (s.fb_velo.fillna(s.fb_velo_l1).fillna(93.5) - 93.5).clip(-5, 5)
    return np.column_stack([np.ones(len(s)), s.rehab_late.fillna(0), s.rehab_n.clip(upper=3).fillna(0),
                            prior / 100, (s.age - 30).clip(-5, 8), s.il_surgery.fillna(0),
                            s.total_ip.fillna(0) / 50, (talent - 0.13) / 0.04, velo])


def workload_baseline(s: pd.DataFrame) -> np.ndarray:
    """Recent full workload = the most recent healthy full season (last 3 years; <30 IL days and
    120+ total IP incl. MiLB/rehab). A short injury year doesn't erase the norm. Fallback: the
    higher of last season's total and the average of the two best of the last three."""
    tot = s[["total_ip", "total_ip_l1", "total_ip_l2"]].fillna(0).to_numpy()
    fallback = np.maximum(s.total_ip.fillna(0).to_numpy(), np.sort(tot, axis=1)[:, -2:].mean(axis=1))
    out = np.full(len(s), np.nan)
    for ipc, ilc in (("total_ip", "il_days"), ("total_ip_l1", "il_days_l1"), ("total_ip_l2", "il_days_l2")):
        ok = np.isnan(out) & (s[ilc].fillna(0).to_numpy() < 30) & (s[ipc].fillna(0).to_numpy() >= 120)
        out[ok] = s[ipc].to_numpy()[ok]
    return np.where(np.isnan(out), fallback, out)


def _gap_x(base: np.ndarray, age: np.ndarray, kbb: np.ndarray | None = None,
           wpct: np.ndarray | None = None) -> np.ndarray:
    """Healthy-hump gap depends on age, on how big the baseline is (huge workloads regress), on how
    good the pitcher is (teams pull spots from pitchers who get hit) and on team quality."""
    kbb = np.zeros(len(base)) if kbb is None else np.nan_to_num((kbb - 0.13) / 0.04)
    wpct = np.zeros(len(base)) if wpct is None else np.nan_to_num(wpct - 0.5) * 10
    return np.column_stack([np.ones(len(base)), np.clip(age - 29, -6, 8), np.clip(base - 170, -60, 60) / 10,
                            np.clip(kbb, -3, 3), np.clip(wpct, -2, 2)])


def pre_injury_baseline(s: pd.DataFrame) -> np.ndarray:
    tot = s[["ip_l1", "ip_l2", "ip_l3", "ip_l4"]].fillna(0).to_numpy()
    return np.sort(tot, axis=1)[:, -2:].mean(axis=1)


def full_workload_starter(s: pd.DataFrame) -> pd.Series:
    """Has carried a starter's load (140+ IP in one of the last 3 seasons) and ended the season
    healthy (not a 60+ day IL stint at season end, no surgery)."""
    full = (s[["ip", "ip_l1", "ip_l2"]].fillna(0) >= 140).sum(axis=1) >= 1
    hurt = ((s.il_at_season_end == 1) & (s.il_end_spell_days.fillna(s.il_days) >= 60)) | (s.il_surgery == 1)
    return full & ~hurt & (s.absent == 0)


def established_starter(s: pd.DataFrame) -> pd.Series:
    """100+ IP this season or last, and not coming off a lost season."""
    return (s[["ip", "ip_l1"]].fillna(0).max(axis=1) >= 100) & (s.il_days < 120)


def _established_x(s: pd.DataFrame) -> np.ndarray:
    age = (s.age - 29).clip(-6, 10)
    return np.column_stack([
        np.ones(len(s)), s.ip.fillna(0) / 100, s.ip_l1.fillna(0) / 100, s.ip_l2.fillna(0) / 100,
        s.seasons_130_5y.fillna(0), (s.kbb_shrunk.fillna(.13) - .13) / .04, age, age.clip(lower=0) ** 2 / 10,
        s.il_days.fillna(0) / 100, s.il_days_l1.fillna(0) / 100, s.il_at_season_end.fillna(0),
        s.il_arm.fillna(0) + s.il_shoulder.fillna(0), s.sp_share.fillna(0), s.ip_per_gs.fillna(5),
        s.late_gs.fillna(0) / 8])


def is_starter(s: pd.DataFrame) -> pd.Series:
    """MLB starters, plus absent starters (injured all season, or upper-minors rotation arms)."""
    mlb = (s.gs >= 3) & (s.sp_share >= 0.5)
    absent = s.get("absent", 0) == 1
    milb_gs = s.get("milb_aaa_gs", 0).fillna(0) + s.get("milb_aa_gs", 0).fillna(0) if "milb_aaa_gs" in s else 0
    prior_gs = s.get("gs_l1", pd.Series(0, index=s.index)).fillna(0) + s.get("gs_l2", pd.Series(0, index=s.index)).fillna(0)
    # Also swingmen / mid-season converts and last year's starters now in another role.
    mixed = (s.gs >= 8) | (s.get("late_gs", 0) >= 3) | (s.get("gs_l1", pd.Series(0, index=s.index)).fillna(0) >= 20)
    return mlb | mixed | (milb_gs >= 8) | (absent & (s.get("il_days", 0) >= 100) & (prior_gs >= 10))


UP = {"ip", "gs", "total_ip", "ip_l1", "gs_l1", "total_ip_l1", "ip_l2", "total_ip_l2", "max_total_ip_3y",
      "seasons_130_5y", "ip_5y_mean", "kbb_shrunk", "late_gs", "late_outs", "career_gs_before", "sp_share",
      "ip_per_gs", "pitches_per_gs", "on_roster_at_end", "rehab_late"}
DOWN = {"il_days", "il_days_l1", "il_days_l2", "il_days_5y", "il_at_season_end", "il_arm", "il_shoulder",
        "il_surgery", "il_arm_l1", "il_surgery_l1", "il_stints"}


def monotonic():
    """Workload and quality can only raise projected innings; injury history can only lower them."""
    return [1 if f in UP else -1 if f in DOWN else 0 for f in FEATURES]


class StarterPlayingTime:
    QUANTILES = (0.1, 0.5, 0.9)

    def fit(self, frame: pd.DataFrame, last_target: int) -> "StarterPlayingTime":
        tr = frame[(frame.season + 1 <= last_target) & (frame.season + 1 != 2020) & (frame.season >= 2015)]
        tr = tr[(tr.gs >= 1) | (tr.milb_gs >= 5) | ((tr.absent == 1) & (tr.gs_l1.fillna(0) + tr.gs_l2.fillna(0) >= 5))]
        kw = dict(max_iter=400, learning_rate=0.04, max_leaf_nodes=24, min_samples_leaf=60,
                  l2_regularization=1.0, random_state=0, monotonic_cst=monotonic())
        X = tr[FEATURES]
        base = HistGradientBoostingRegressor(**kw)
        y = tr.ip_next
        h = ((tr.il_days_next.fillna(0) == 0) & (y > 0)).to_numpy()
        self.ip = clone(base).fit(X, y)
        self.gs = HistGradientBoostingRegressor(**kw).fit(X, tr.gs_next)
        self.healthy = clone(base).fit(X[h], y[h])
        E, H = np.empty(len(y)), np.empty(len(y))
        for season in tr.season.unique():
            te = (tr.season == season).to_numpy()
            E[te] = clone(base).fit(X[~te], y[~te]).predict(X[te])
            H[te] = clone(base).fit(X[~te & h], y[~te & h]).predict(X[te])
        H = np.maximum(H, E)
        self.blend = HealthyBlend().fit(E, H, y.to_numpy(), _blend_z(tr))
        final = E + self.blend.weight(_blend_z(tr)) * (H - E)
        # Sparse but distinct group: established starters returning from a lost season. A small
        # ridge regression on rehab timing, prior workload and age (backtests: late rehab => ~75 IP
        # mean vs ~46 without) replaces the boosted estimate there.
        rs = returning_starter(tr).to_numpy()
        Xr, yr = _returning_x(tr[rs]), y.to_numpy()[rs]
        self.ret_b = np.linalg.solve(Xr.T @ Xr + 5 * np.eye(Xr.shape[1]), Xr.T @ yr)
        oof_r = np.empty(rs.sum())
        seasons_r = tr.season.to_numpy()[rs]
        for season in np.unique(seasons_r):
            te = seasons_r == season
            b = np.linalg.solve(Xr[~te].T @ Xr[~te] + 5 * np.eye(Xr.shape[1]), Xr[~te].T @ yr[~te])
            oof_r[te] = np.clip(Xr[te] @ b, 0, 200)
        final[rs] = oof_r
        # Established starters: average the boosted estimate with a ridge model trained only on
        # them (backtest: better ranking of who carries a full workload than either alone).
        es = established_starter(tr).to_numpy()
        Xe, ye = _established_x(tr[es]), y.to_numpy()[es]
        self.est_b = np.linalg.solve(Xe.T @ Xe + 3 * np.eye(Xe.shape[1]), Xe.T @ ye)
        seasons_e = tr.season.to_numpy()[es]
        oof_e = np.empty(es.sum())
        for season in np.unique(seasons_e):
            te = seasons_e == season
            b = np.linalg.solve(Xe[~te].T @ Xe[~te] + 3 * np.eye(Xe.shape[1]), Xe[~te].T @ ye[~te])
            oof_e[te] = np.clip(Xe[te] @ b, 0, 215)
        final[es] = 0.5 * final[es] + 0.5 * oof_e
        self.q = BinnedQuantiles(self.QUANTILES, thresholds=(100, 150)).fit_oof(final, y, tr.established)

        # --- Workload thesis for full-workload starters: two humps, not one average.
        # Healthy hump: recent workload baseline minus an age-dependent gap (median of pitchers who
        # threw 100+ IP the next year). Injury hump: everything below 100 IP.
        fw = full_workload_starter(tr).to_numpy()
        base = workload_baseline(tr[fw])
        yf = y.to_numpy()[fw]
        hump = yf >= 100
        age = tr.age.to_numpy()[fw]
        kb = tr.kbb_shrunk.to_numpy()[fw]
        wp = tr.team_wpct.to_numpy()[fw]
        A = _gap_x(base[hump], age[hump], kb[hump], wp[hump])
        self.gap_b = np.linalg.lstsq(A, yf[hump] - base[hump], rcond=None)[0]
        self.hump_sd = float(np.std(yf[hump] - (base[hump] + A @ self.gap_b)))
        self.low_q = np.quantile(yf[~hump], np.linspace(0, 1, 101))
        from sklearn.ensemble import HistGradientBoostingClassifier
        self.p_hump = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.04, max_leaf_nodes=12, min_samples_leaf=60, l2_regularization=1.0,
            random_state=0, monotonic_cst=monotonic()).fit(X[fw], hump.astype(int))
        # Returning starters (lost season): P(pitch in MLB) and IP-if-returned as a share of the
        # pre-injury workload, by whether they rehabbed late.
        rs_y, rs_late = y.to_numpy()[rs], tr.rehab_late.fillna(0).to_numpy()[rs]
        pre = np.maximum(pre_injury_baseline(tr[rs]), 1)
        self.ret = {}
        for late in (0, 1):
            m = rs_late == late
            came = m & (rs_y > 0)
            self.ret[late] = (float(came.sum() / max(m.sum(), 1)), float(np.median(rs_y[came] / pre[came])),
                              np.quantile(rs_y[came] / pre[came], np.linspace(0, 1, 101)))
        return self

    def _mixture_quantiles(self, center, p, high_draws=None):
        """P10/P50/P90 of a two-component distribution: N(center, sd) w.p. p, low hump otherwise."""
        qs = np.linspace(0.005, 0.995, 200)
        out = []
        from scipy.stats import norm
        for c, pp in zip(center, p):
            hi = np.clip(norm.ppf(qs, c, self.hump_sd), 100, 230) if high_draws is None else high_draws
            lo = np.interp(qs, np.linspace(0, 1, 101), self.low_q)
            vals = np.r_[hi, lo]
            w = np.r_[np.full(len(hi), pp / len(hi)), np.full(len(lo), (1 - pp) / len(lo))]
            order = np.argsort(vals)
            cw = np.cumsum(w[order])
            out.append([vals[order][np.searchsorted(cw, q)] for q in (0.1, 0.5, 0.9)])
        return np.array(out)

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[FEATURES]
        out = pd.DataFrame(index=frame.index)
        E = np.clip(self.ip.predict(X), 0, 215)
        H = np.maximum(np.clip(self.healthy.predict(X), 0, 215), E)
        w = self.blend.weight(_blend_z(frame))
        out["IP"] = E + w * (H - E)
        rs = returning_starter(frame).to_numpy()
        if rs.any():
            out.loc[rs, "IP"] = np.clip(_returning_x(frame[rs]) @ self.ret_b, 0, 200)
        es = established_starter(frame).to_numpy()
        if es.any():
            out.loc[es, "IP"] = 0.5 * out.loc[es, "IP"] + 0.5 * np.clip(_established_x(frame[es]) @ self.est_b, 0, 215)
        out["IP_expected"] = out.IP
        out["P_full_season"] = np.nan

        out["IP_healthy"], out["healthy_weight"] = H, w
        # GS scales with the blended workload.
        out["GS"] = np.clip(self.gs.predict(X) * np.where(E > 1, out.IP / np.maximum(E, 1), 1.0), 0, 34)
        # Keep starts consistent with innings: at least ~4.5 IP per start for a starter's workload.
        out["GS"] = np.minimum(out.GS, out.IP / 4.5)
        q = self.q.predict(out.IP.to_numpy(), frame.established.to_numpy())
        for qq in self.QUANTILES:
            out[f"IP_p{int(qq * 100)}"] = q[qq].to_numpy()
        out["P_100IP"], out["P_150IP"] = q["P_100"].to_numpy(), q["P_150"].to_numpy()
        # Workload thesis: full-workload starters are projected at their healthy-hump center.
        fw = full_workload_starter(frame).to_numpy()
        if fw.any():
            base = workload_baseline(frame[fw])
            center = np.clip(base + _gap_x(base, frame.age.to_numpy()[fw], frame.kbb_shrunk.to_numpy()[fw],
                                           frame.team_wpct.to_numpy()[fw]) @ self.gap_b, 0, 215)
            p = self.p_hump.predict_proba(X[fw])[:, 1]
            q = self._mixture_quantiles(center, p)
            low_mean = float(np.mean(self.low_q))
            expected = p * center + (1 - p) * low_mean
            # Headline = healthy-hump workload (the workload thesis; quality and team now move the center).
            out.loc[fw, "IP"] = center
            out.loc[fw, "IP_expected"] = expected
            out.loc[fw, ["IP_p10", "IP_p50", "IP_p90"]] = q
            out.loc[fw, "P_full_season"] = p
            out.loc[fw, "P_100IP"] = p
        # Returning starters: center = typical IP of comparable pitchers who made it back.
        rs = returning_starter(frame).to_numpy()
        if rs.any():
            late = frame.rehab_late.fillna(0).to_numpy()[rs].astype(int)
            pre = pre_injury_baseline(frame[rs])
            p_ret = np.array([self.ret[l][0] for l in late])
            center = np.array([self.ret[l][1] for l in late]) * pre
            out.loc[rs, "IP"] = np.clip(center, 0, 200)
            out.loc[rs, "IP_expected"] = p_ret * center
            out.loc[rs, "P_full_season"] = p_ret
            qs = []
            for pr, l, pb in zip(p_ret, late, pre):
                share = np.interp(np.linspace(0.005, 0.995, 200), np.linspace(0, 1, 101), self.ret[l][2]) * pb
                vals = np.r_[share, np.zeros(200)]
                wts = np.r_[np.full(200, pr / 200), np.full(200, (1 - pr) / 200)]
                o = np.argsort(vals)
                cw = np.cumsum(wts[o])
                qs.append([vals[o][np.searchsorted(cw, qq)] for qq in (0.1, 0.5, 0.9)])
            out.loc[rs, ["IP_p10", "IP_p50", "IP_p90"]] = np.array(qs)
        # Starts follow the headline workload (~5.6 IP per start for full-workload arms).
        thesis = full_workload_starter(frame).to_numpy() | returning_starter(frame).to_numpy()
        out.loc[thesis, "GS"] = np.clip(out.loc[thesis, "IP"] / 5.6, 0, 33)
        return out


def rotation_claim(s: pd.DataFrame) -> pd.Series:
    """Claim on a rotation spot from end-of-season info: starts (late-season starts count extra) and
    last year's starts, scaled by quality. A starter who lost the season to injury keeps most of his
    prior claim (out-for-next-year pitchers are excluded upstream)."""
    gs_l1 = s.gs_l1.fillna(0)
    own = np.maximum(s.gs, np.where(s.il_days >= 100, 0.9 * gs_l1, 0))
    return (own + 1.5 * s.late_gs.fillna(0) + 0.3 * gs_l1) * (1 + 3 * (s.kbb_shrunk.fillna(.13) - .13))


def role_group(s: pd.DataFrame) -> pd.Series:
    """'convert' (moved into the rotation this year), 'mixed' (swingman), else 'starter'."""
    prior_gs = s[["gs_l1", "gs_l2", "gs_l3", "gs_l4"]].fillna(0)
    conv = (s.gs >= 8) & (prior_gs.max(axis=1) < 10)
    mixed = (s.gs >= 3) & (s.sp_share < 0.8) & ~conv
    return pd.Series(np.select([conv, mixed], ["convert", "mixed"], "starter"), index=s.index)


class ConversionModel:
    """P(a reliever becomes a starter next year: 10+ real starts), the Jax route. Base rate ~2%;
    prior starting history (MLB or MiLB), multi-inning relief, quality and a few real starts raise it.
    Also learns the typical next-year IP of relievers who did convert (relative to this year's)."""

    def _x(self, s):
        relief_g = (s.games - s.gs).clip(lower=1)
        per_relief = ((s.ip - s.sp_outs.fillna(0) / 3) / relief_g).clip(0, 4)
        prior_sp = s.gs_l1.fillna(0) + s.gs_l2.fillna(0) + s.milb_gs.fillna(0)
        relief_ip = (s.ip - s.sp_outs.fillna(0) / 3).clip(lower=0)  # heavy relief volume (e.g. 75+ IP) converts more
        return np.nan_to_num(np.column_stack([np.log1p(prior_sp), per_relief, (s.kbb_shrunk.fillna(.13) - .13) / .04,
                                              (s.age.fillna(28) - 28) / 4, s.gs.fillna(0), s.total_ip.fillna(0) / 50,
                                              relief_ip / 25]))

    def fit(self, f: pd.DataFrame, last_target: int) -> "ConversionModel":
        from sklearn.linear_model import LogisticRegression
        key = f.set_index(["mlbam_id", "season"])
        nx = key.reindex(pd.MultiIndex.from_arrays([f.mlbam_id, f.season + 1]))
        f = f.assign(gs_next=nx.gs.fillna(0).to_numpy())
        rp = f[(f.gs < 8) & ((f.games - f.gs) >= 15) & f.season.between(2015, last_target - 1)
               & ~f.season.isin([2019, 2020]) & (f.absent == 0)]
        y = (rp.gs_next >= 10).astype(int)
        self.m = LogisticRegression(C=1.0, max_iter=1000).fit(self._x(rp), y)
        conv = rp[y == 1]
        self.ip_fit = np.polyfit(conv.total_ip.clip(30, 160), conv.ip_next, 1)  # IP if converted
        return self

    def predict(self, s: pd.DataFrame) -> pd.DataFrame:
        p = self.m.predict_proba(self._x(s))[:, 1]
        ip_conv = np.clip(np.polyval(self.ip_fit, s.total_ip.fillna(0).clip(30, 160)), 40, 180)
        return pd.DataFrame({"P_convert": p, "IP_if_converted": ip_conv}, index=s.index)
