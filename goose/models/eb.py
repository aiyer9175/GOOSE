"""Generic empirical-Bayes component-rate engine shared by hitters and pitchers.

For each component c:

    prior mean     mu   = sigmoid(logit(league_c) + X @ beta)      X: shrunk process stats, age, pedigree
    evidence            = sum_i w_i * MLB counts (t, t-1, t-2)
                          where the MLB numerator blends actual and expected-outcome counts
                          (spray-aware batted-ball model) on the SAME denominator, so expected
                          stats sharpen the estimate without inflating the sample size
                          + sum_L w_L * f_L * MiLB counts (AAA, AA; league-normalized upstream)
    posterior rate      = (k * mu + evidence_num) / (k + evidence_den)
    projection          = sigmoid(logit(posterior) + a1 * (age - 27) + a2 * (age - 27)^2
                                  + (gamma @ D) * exp(-career / lambda))
    where D are development covariates (prospect pedigree): tools that a former top prospect
    showed can keep surfacing in his first several hundred MLB PA; lambda (fit) sets how fast
    that fades as MLB evidence accumulates.

All parameters are fit jointly per component by binomial maximum likelihood on next-season
outcomes. k + evidence_den is the effective sample size, used downstream for uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from goose.data.http import ROOT

FEATURES = ROOT / "data" / "features"
PEDIGREE = ["log_pick_c", "undrafted"]
LAG_WEIGHTS = (1.0, 0.6, 0.35)  # process stats only; outcome lag weights are fit


@dataclass(frozen=True)
class Spec:
    components: dict[str, tuple[str, str]]      # name -> (num col, den col)
    covariates: dict[str, list[str]]            # name -> prior-mean covariates
    process: dict[str, tuple[str, str]]         # shrunk process metric -> (num, den)
    x_evidence: dict[str, str]                  # component -> expected-count column (same den)
    stabilization_file: str
    extra_design: callable = None               # frame -> DataFrame of extra covariates
    post_covariates: tuple = ()                 # development covariates, decaying with career volume
    career_col: str = "career_total"            # frame column with career volume through t


@dataclass
class ComponentFit:
    name: str
    covariates: list[str]
    params: np.ndarray = field(default=None)
    x_mean: pd.Series = field(default=None)
    x_std: pd.Series = field(default=None)
    nll: float = np.nan

    post: tuple = ()

    # layout: [beta, log_k, w1, w2, wAAA, wAA, log_fAAA, log_fAA, a1, a2, logit_omega_x, gamma..., log_lambda]
    def unpack(self, p):
        nb = len(self.covariates) + len(PEDIGREE)
        log_k, w1, w2, w_aaa, w_aa, lf_aaa, lf_aa, a1, a2, log_wx = p[nb:nb + 10]
        return (p[:nb], np.exp(log_k), expit(w1), expit(w2), expit(w_aaa), expit(w_aa), np.exp(lf_aaa),
                np.exp(lf_aa), a1, a2, expit(log_wx))

    def unpack_post(self, p):
        nb = len(self.covariates) + len(PEDIGREE) + 10
        if len(p) <= nb:
            return np.zeros(0), 1.0
        return p[nb:-1], np.exp(p[-1])


class EBRates:
    SPEC: Spec = None

    def __init__(self):
        self.fits: dict[str, ComponentFit] = {}
        self.league: pd.DataFrame | None = None

    def _league(self, frame: pd.DataFrame) -> pd.DataFrame:
        g = frame.groupby("season")
        rows = {c: g[f"{n}_l0"].sum() / g[f"{d}_l0"].sum() for c, (n, d) in self.SPEC.components.items()}
        rows |= {f"p_{m}": g[f"{n}_l0"].sum() / g[f"{d}_l0"].sum() for m, (n, d) in self.SPEC.process.items()}
        return pd.DataFrame(rows)

    def _design(self, frame: pd.DataFrame) -> pd.DataFrame:
        stab = pd.read_csv(FEATURES / self.SPEC.stabilization_file, index_col=0)
        X = pd.DataFrame(index=frame.index)
        lg = self.league.reindex(frame.season).set_index(frame.index)
        for m, (n, d) in self.SPEC.process.items():
            num = sum(w * frame[f"{n}_l{i}"].fillna(0) for i, w in enumerate(LAG_WEIGHTS))
            den = sum(w * frame[f"{d}_l{i}"].fillna(0) for i, w in enumerate(LAG_WEIGHTS))
            if f"aaasc_{n}_l0" in frame and f"aaasc_{d}_l0" in frame:
                # AAA Statcast (2025+) counts as process evidence at 60% weight.
                num = num + 0.6 * sum(w * frame[f"aaasc_{n}_l{i}"].fillna(0) for i, w in enumerate(LAG_WEIGHTS[:2]))
                den = den + 0.6 * sum(w * frame[f"aaasc_{d}_l{i}"].fillna(0) for i, w in enumerate(LAG_WEIGHTS[:2]))
            fallback = self.league[f"p_{m}"].mean()
            mu = lg[f"p_{m}"].fillna(0 if np.isnan(fallback) else fallback)
            k = stab.loc[m, "k"]
            # League-relative shrunk value; 0 (= league average) when the metric didn't exist yet.
            X[f"p_{m}"] = ((num + k * mu) / (den + k) - mu).where(den > 0, 0.0).fillna(0.0)
        X["age_c"] = frame.age - 27
        X["log_pick_c"] = frame.log_pick.fillna(np.log(300)) - np.log(300)
        X["undrafted"] = frame.undrafted.fillna(1)
        if self.SPEC.extra_design is not None:
            X = X.join(self.SPEC.extra_design(frame))
        return X

    def _evidence(self, frame, c, w1, w2, w_aaa, w_aa, f_aaa, f_aa, w_x=0.0):
        n, d = self.SPEC.components[c]

        def lagsum(col):
            return frame[f"{col}_l0"].fillna(0) + w1 * frame[f"{col}_l1"].fillna(0) + w2 * frame[f"{col}_l2"].fillna(0)
        num, den = lagsum(n), lagsum(d)
        if c in self.SPEC.x_evidence:
            # Blend actual and expected counts; the denominator (sample size) is unchanged.
            xnum = np.minimum(lagsum(self.SPEC.x_evidence[c]), den)
            num = (1 - w_x) * num + w_x * xnum
        for lvl, w, f in (("aaa", w_aaa, f_aaa), ("aa", w_aa, f_aa)):
            if f"milb_{lvl}_{n}_l0" not in frame:
                continue
            mn = frame[f"milb_{lvl}_{n}_l0"].fillna(0) + 0.5 * frame[f"milb_{lvl}_{n}_l1"].fillna(0)
            md = frame[f"milb_{lvl}_{d}_l0"].fillna(0) + 0.5 * frame[f"milb_{lvl}_{d}_l1"].fillna(0)
            num = num + w * np.minimum(f * mn, md)
            den = den + w * md
        return num.to_numpy(), den.to_numpy()

    def predict_component(self, frame, X, c, with_n=False):
        fit = self.fits[c]
        beta, k, w1, w2, w_aaa, w_aa, f_aaa, f_aa, a1, a2, w_x = fit.unpack(fit.params)
        Z = ((X[fit.covariates + PEDIGREE] - fit.x_mean) / fit.x_std).to_numpy()
        lg = self.league[c].reindex(frame.season).to_numpy()
        lg = np.where(np.isnan(lg), self.league[c].iloc[-1], lg)
        mu = expit(logit(lg) + Z @ beta)
        num, den = self._evidence(frame, c, w1, w2, w_aaa, w_aa, f_aaa, f_aa, w_x)
        post = (k * mu + num) / (k + den)
        age_next = frame.age.to_numpy() + 1 - 27
        eta = logit(np.clip(post, 1e-4, 1 - 1e-4)) + a1 * age_next + a2 * age_next ** 2
        gamma, lam = fit.unpack_post(fit.params)
        if len(gamma):
            D = X[list(fit.post)].to_numpy()
            career = frame[self.SPEC.career_col].fillna(0).to_numpy()
            eta = eta + (D @ gamma) * np.exp(-career / lam)
        pred = expit(eta)
        return (pred, k + den, mu) if with_n else pred

    def fit(self, frame: pd.DataFrame, last_target: int, max_career: float | None = None,
            career_col: str = "career_pa_before", volume_col: str = "pa") -> "EBRates":
        train = frame[frame.season + 1 <= last_target]
        if max_career is not None:
            train = train[train[career_col].fillna(0) + train[volume_col].fillna(0) <= max_career]
        self.league = self._league(frame[frame.season <= last_target - 1])
        X = self._design(train)
        for c, (n, d) in self.SPEC.components.items():
            cov = self.SPEC.covariates[c]
            fit = ComponentFit(c, cov, post=tuple(self.SPEC.post_covariates))
            cols = cov + PEDIGREE
            fit.x_mean, fit.x_std = X[cols].mean(), X[cols].std().replace(0, 1).fillna(1)
            y_n, y_d = train[f"{n}_next"].to_numpy(), train[f"{d}_next"].to_numpy()
            ok = np.isfinite(y_n) & np.isfinite(y_d) & (y_d > 0)
            tr, Xc, yn, yd = train[ok], X[ok], np.minimum(y_n[ok], y_d[ok]), y_d[ok]
            self.fits[c] = fit

            npost_ = len(self.SPEC.post_covariates)

            def nll(p):
                fit.params = p
                pr = np.clip(self.predict_component(tr, Xc, c), 1e-5, 1 - 1e-5)
                # Ridge on prior betas; stronger shrinkage on development terms, which are
                # learned from few seasons of prospect lists.
                pen = 5e-4 * np.sum(p[:len(cols)] ** 2)
                if npost_:
                    pen += 0.5 * np.sum(p[-npost_ - 1:-1] ** 2)
                return -(yn * np.log(pr) + (yd - yn) * np.log(1 - pr)).sum() / yd.sum() * 1e3 + pen

            npost = len(fit.post)
            p0 = np.r_[np.zeros(len(cols)), np.log(200.0), 0.5, -0.5, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            bounds = [(None, None)] * len(p0)
            if c not in self.SPEC.x_evidence:
                bounds[-1], p0[-1] = (-20, -20), -20  # omega = 0: no expected-outcome blend
            if npost:
                p0 = np.r_[p0, np.zeros(npost), np.log(600.0)]
                bounds += [(-1.0, 1.0)] * npost + [(np.log(100.0), np.log(4000.0))]
            res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds)
            fit.params, fit.nll = res.x, res.fun
        return self

    def summary(self) -> pd.DataFrame:
        rows = {}
        for c, fit in self.fits.items():
            beta, k, w1, w2, w_aaa, w_aa, f_aaa, f_aa, a1, a2, w_x = fit.unpack(fit.params)
            rows[c] = dict(k=round(k, 1), w_t1=round(w1, 2), w_t2=round(w2, 2), w_x=round(w_x, 2),
                           w_aaa=round(w_aaa, 2), w_aa=round(w_aa, 2), f_aaa=round(f_aaa, 2), f_aa=round(f_aa, 2),
                           age1=round(a1, 4), age2=round(a2, 5),
                           **{f"b_{n}": round(b, 3) for n, b in zip(fit.covariates + PEDIGREE, beta)})
        return pd.DataFrame(rows).T

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = self._design(frame)
        out = pd.DataFrame(index=frame.index)
        for c in self.SPEC.components:
            p, n, mu = self.predict_component(frame, X, c, with_n=True)
            out[c], out[f"{c}_n"], out[f"{c}_prior"] = p, n, mu
        return out


def lag_frame(base: pd.DataFrame, info: pd.DataFrame, components: dict) -> pd.DataFrame:
    """Join lagged (t, t-1, t-2) count columns and next-season targets onto info rows."""
    idx = base.set_index(["mlbam_id", "season"])
    frame = info.copy()
    key = pd.MultiIndex.from_frame(frame[["mlbam_id", "season"]])
    for lag in range(3):
        k = pd.MultiIndex.from_arrays([key.get_level_values(0), key.get_level_values(1) - lag])
        lagged = idx.reindex(k)
        lagged.index = frame.index
        frame = frame.join(lagged.add_suffix(f"_l{lag}"))
    nxt = idx.reindex(pd.MultiIndex.from_arrays([key.get_level_values(0), key.get_level_values(1) + 1]))
    nxt.index = frame.index
    targets = list(dict.fromkeys(c for nd in components.values() for c in nd))
    return frame.join(nxt[targets].add_suffix("_next"))


def park_lookup(frame: pd.DataFrame) -> pd.DataFrame:
    """Park factors (log scale, 0 = neutral) for the player's end-of-season team; 0 if unknown."""
    pf = pd.read_parquet(FEATURES / "park_factors.parquet").set_index(["team", "season"])
    idx = pd.MultiIndex.from_arrays([frame.team, frame.season])
    out = pf.reindex(idx)
    out.index = frame.index
    return np.log(out[["park_hr", "park_h"]]).fillna(0.0)
