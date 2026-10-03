"""Saves as a within-team competition (conditional logit with an "outside" option).

Each team's relievers compete for next season's saves:

    share_i = exp(u_i) / (exp(a) + sum_j exp(u_j)),   u_i = x_i @ beta

where a is the utility of "someone not in the bullpen yet" (winter signings, trades, call-ups).
Saves_i = team saves pool x share_i. Features: recent hold on the job (save share, late-season
save-type finishes), the job when last healthy (prior-season saves), stuff (velocity, whiff rate,
K-BB), experience (career saves), health, handedness. Fit on team-seasons 2015+ where the
player stayed with the team.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

FEATS = ["sv_share", "late_share", "prev_closer", "career_sv", "velo_z", "whiff_z", "kbb_z", "lhp", "age_c",
         "il_frac", "hurt_end", "absent", "top_late", "top_prev", "top_stuff"]


def features(f: pd.DataFrame) -> pd.DataFrame:
    """f: reliever frame (goose.models.reliever.build_rp_frame) rows for the candidate pool."""
    x = pd.DataFrame(index=f.index)
    team_sv = f.groupby(["team", "season"]).sv.transform("sum").clip(lower=10)
    team_late = f.groupby(["team", "season"]).late_finish_close.transform("sum").clip(lower=3)
    x["sv_share"] = f.sv / team_sv
    x["late_share"] = f.late_finish_close / team_late
    x["prev_closer"] = (np.maximum(f.sv_l1, f.sv_l2) / 30).clip(upper=1.3)
    x["career_sv"] = np.log1p(f.career_sv.fillna(0))
    for c, src in (("velo_z", "fb_velo"), ("whiff_z", "whiff_rate"), ("kbb_z", "kbb_shrunk")):
        v = f[src]
        x[c] = ((v - v.groupby(f.season).transform("mean")) / v.groupby(f.season).transform("std")).fillna(0)
    x["lhp"] = (f.throws == "L").astype(float)
    x["age_c"] = (f.age - 29).clip(-6, 10) / 5
    x["il_frac"] = (f.il_days.fillna(0) / 180).clip(upper=1)
    x["hurt_end"] = f.il_at_season_end.fillna(0)
    x["absent"] = f.absent.fillna(0)
    # Closer jobs are winner-take-most: being the team's top option matters more than the level.
    key = [f.team, f.season]
    stuff = x.velo_z + x.kbb_z + x.whiff_z
    x["top_late"] = ((x.late_share == x.late_share.groupby(key).transform("max")) & (x.late_share > 0.2)).astype(float)
    x["top_prev"] = ((x.prev_closer == x.prev_closer.groupby(key).transform("max")) & (x.prev_closer > 0.3)).astype(float)
    x["top_stuff"] = (stuff == stuff.groupby(key).transform("max")).astype(float)
    return x


class SavesShare:
    def fit(self, f: pd.DataFrame, last_target: int, ridge: float = 1.0) -> "SavesShare":
        tr = f[(f.season + 1 <= last_target) & (f.season >= 2015) & (f.season + 1 != 2020) & (f.season != 2020)]
        from goose.models.reliever import is_reliever
        tr = tr[tr.team.notna() & is_reliever(tr)]  # the competition is among relievers only
        Xall = features(tr)  # computed on the full pool (team ranks), then departed players dropped
        stay = (tr.team_next == tr.team).to_numpy()
        tr, X = tr[stay], Xall[stay].to_numpy()
        y = tr.sv_next.fillna(0).to_numpy()
        groups = tr.groupby(["team", "season"]).ngroup().to_numpy()
        team_next_sv = tr.team_sv_next.to_numpy()
        # Outside saves per group = team's next-year saves not earned by candidates who stayed.
        g_ids = np.unique(groups)
        inside = np.bincount(groups, weights=y, minlength=g_ids.max() + 1)
        total = pd.Series(team_next_sv).groupby(groups).max().reindex(range(g_ids.max() + 1)).fillna(0).to_numpy()
        outside = np.clip(total - inside, 0, None)
        keep = total > 10
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd

        ng = len(total)
        scale = max(total[keep].sum(), 1)

        def nll(theta):
            """Negative log-likelihood (per save, x100) with exact gradient."""
            a, b = theta[0], theta[1:]
            u = Z @ b
            m = max(u.max(), a)
            eu, ea = np.exp(u - m), np.exp(a - m)
            denom = np.bincount(groups, weights=eu, minlength=ng) + ea
            n_g = np.bincount(groups, weights=y, minlength=ng) + np.where(keep, outside, 0)
            ll = (y * (u - m)).sum() + (np.where(keep, outside, 0) * (a - m)).sum() - (n_g * np.log(denom)).sum()
            p_i = eu / denom[groups]
            g_b = Z.T @ (y - n_g[groups] * p_i)
            g_a = (np.where(keep, outside, 0) - n_g * ea / denom).sum()
            f = -ll / scale * 100 + ridge * np.sum(b ** 2)
            grad = np.r_[-g_a / scale * 100, -g_b / scale * 100 + 2 * ridge * b]
            return f, grad

        res = minimize(nll, np.zeros(Z.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        self.converged = bool(res.success)
        self.a, self.b = res.x[0], res.x[1:]
        # Team saves pool: shrink this year's team total toward the league mean.
        ts = tr.groupby(["team", "season"]).agg(now=("sv", "sum"), nxt=("team_sv_next", "max")).dropna()
        A = np.column_stack([np.ones(len(ts)), ts.now])
        self.pool_b = np.linalg.lstsq(A, ts.nxt, rcond=None)[0]
        return self

    def predict(self, f: pd.DataFrame, eligible: np.ndarray | None = None) -> pd.DataFrame:
        """Saves share and saves for candidates; ineligible rows (e.g. departing free agents) get 0
        and their saves stay in the team pool for the remaining candidates and outside options."""
        X = (features(f).to_numpy() - self.mu) / self.sd
        u = X @ self.b
        eligible = np.ones(len(f), bool) if eligible is None else eligible
        eu = np.where(eligible, np.exp(u), 0.0)
        g = f.groupby(["team", "season"]).ngroup().to_numpy()
        denom = np.bincount(g, weights=eu) + np.exp(self.a)
        share = eu / denom[g]
        team_sv = f.groupby(["team", "season"]).sv.transform("sum").to_numpy()
        pool = self.pool_b[0] + self.pool_b[1] * team_sv
        return pd.DataFrame({"sv_share_next": share, "SV_share_model": share * pool}, index=f.index)

    def coefs(self) -> pd.Series:
        return pd.Series(np.r_[self.a, self.b], index=["outside"] + FEATS)
