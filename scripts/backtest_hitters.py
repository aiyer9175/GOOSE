"""Rolling-origin backtest of hitter rate projections vs Steamer and ZiPS.

For each target season T: fit on transitions ending <= T-1, project T from end-of-(T-1) data,
score on the focus population (100-300 MLB PA in T-1) and on everyone.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goose.eval.competitors import actual_hitters, load, score_rates  # noqa: E402
from goose.models.hitter_rates import HitterRates, HitterRatesPed  # noqa: E402
from goose.models.statline import hitter_line  # noqa: E402
from goose.data.http import ROOT  # noqa: E402

CACHE = ROOT / "data" / "models"
RATES = ["wOBA", "OBP", "SLG", "AVG", "ISO", "K%", "BB%"]


def fit_cached(frame: pd.DataFrame, last_target: int, variant: str = "base",
               max_career_pa: float | None = None) -> HitterRates:
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"hitter_rates_{variant}_{last_target}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    cls = HitterRatesPed if variant.startswith("ped") else HitterRates
    m = cls().fit(frame, last_target, max_career_pa)
    path.write_bytes(pickle.dumps(m))
    return m


def project(frame: pd.DataFrame, model: HitterRates, season: int) -> pd.DataFrame:
    rows = frame[frame.season == season]
    rates = model.predict(rows)
    line = hitter_line(rates, pd.Series(600.0, index=rows.index))
    line["mlbam_id"] = rows.mlbam_id.to_numpy()
    line["name"] = rows.name.to_numpy()
    line["pa_prev"] = rows.pa.to_numpy()
    line["career_prev"] = (rows.career_pa_before.fillna(0) + rows.pa.fillna(0)).to_numpy()
    line["season_prev"] = season
    raw = rates.rename(columns={c: f"r_{c}" for c in rates.columns})
    return line.join(raw)


def main(targets=range(2022, 2027), variant: str = "base", max_career_pa: float | None = None) -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / "data" / "features" / "hitter_frame.parquet")
    results = []
    for T in targets:
        model = fit_cached(frame, T - 1, variant, max_career_pa)
        ours = project(frame, model, T - 1)
        ours.to_parquet(CACHE / f"hitter_proj_{variant}_{T}.parquet")
        comp = load("hitters", T)
        comp = comp[comp.system.isin(["steamer", "zips"])]
        g = ours.assign(system="goose")[["mlbam_id", "system"] + RATES]
        proj = pd.concat([comp[["mlbam_id", "system"] + RATES], g], ignore_index=True)
        act = actual_hitters(T)
        focus = ours.loc[ours.pa_prev.between(100, 300), "mlbam_id"]
        for label, ids, min_pa in (("focus 100-300", pd.Index(focus), 100), ("all", None, 200)):
            sc = score_rates(proj, act, RATES, min_pa, ids=ids)
            sc["season"], sc["pop"] = T, label
            results.append(sc.reset_index(names="system"))
        print(f"{T} done", flush=True)
    res = pd.concat(results, ignore_index=True)
    res.to_csv(CACHE / f"backtest_hitter_rates_{variant}.csv", index=False)
    return res


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.precision", 4)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="base")
    ap.add_argument("--max-career-pa", type=float)
    ap.add_argument("--targets", type=int, nargs="*", default=list(range(2022, 2027)))
    a = ap.parse_args()
    r = main(targets=a.targets, variant=a.variant, max_career_pa=a.max_career_pa)
    for pop, g in r.groupby("pop"):
        print(f"\n== {pop}")
        print(g.pivot_table(index="season", columns="system", values="wOBA").to_string())
        w = g.assign(wt=g.n).groupby("system").apply(
            lambda d: pd.Series({s: np.sqrt(np.average(d[s] ** 2, weights=d.wt)) for s in RATES} | {"n": d.n.sum()}))
        print("pooled:\n" + w.sort_values("wOBA").to_string())
