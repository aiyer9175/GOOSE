"""Final hitter rate model: segment routing + calibration, scored leave-one-season-out.

Players with <= 1000 career PA use the low-sample-trained model ('xblend_low'), others the
full model ('xblend'). Calibration slopes for target season T are fit on the other seasons'
out-of-sample projections only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goose.data.http import ROOT  # noqa: E402
from goose.eval.competitors import actual_hitters, load, score_rates  # noqa: E402
from goose.models import calibrate  # noqa: E402
from goose.models.statline import hitter_line  # noqa: E402

CACHE = ROOT / "data" / "models"
RATES = ["wOBA", "OBP", "SLG", "AVG", "ISO", "K%", "BB%"]
TARGETS = range(2022, 2027)
FULL, LOW = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("xblend", "xblend_low")


def combined(T: int) -> pd.DataFrame:
    low = pd.read_parquet(CACHE / f"hitter_proj_{LOW}_{T}.parquet")
    full = pd.read_parquet(CACHE / f"hitter_proj_{FULL}_{T}.parquet").set_index("mlbam_id")
    use_low = low.career_prev <= 1000
    out = pd.concat([low[use_low], full.reindex(low.loc[~use_low, "mlbam_id"]).dropna(subset=["wOBA"]).reset_index()], ignore_index=True)
    return out


def line_from_rates(df: pd.DataFrame) -> pd.DataFrame:
    rates = df[[c for c in df if c.startswith("r_") and not c.endswith(("_n", "_prior"))]]
    rates = rates.rename(columns=lambda c: c[2:])
    line = hitter_line(rates, pd.Series(600.0, index=df.index))
    for c in ("mlbam_id", "name", "pa_prev", "career_prev", "season_prev"):
        line[c] = df[c].to_numpy()
    return line


def main():
    seasons = pd.read_parquet(ROOT / "data" / "features" / "hitter_seasons.parquet")
    actual = calibrate.hitter_actuals(seasons)
    proj = {T: combined(T) for T in TARGETS}
    results, all_slopes = [], {}
    for T in TARGETS:
        oos = pd.concat([proj[t] for t in TARGETS if t != T], ignore_index=True)
        slopes = calibrate.fit(oos, actual)
        all_slopes[T] = slopes
        variants = {"goose_raw": line_from_rates(proj[T]),
                    "goose_cal": line_from_rates(calibrate.apply(proj[T], slopes))}
        comp = load("hitters", T)
        comp = comp[comp.system.isin(["steamer", "zips"])][["mlbam_id", "system"] + RATES]
        frames = [comp] + [v.assign(system=k)[["mlbam_id", "system"] + RATES] for k, v in variants.items()]
        p = pd.concat(frames, ignore_index=True)
        act = actual_hitters(T)
        focus = variants["goose_raw"].loc[variants["goose_raw"].pa_prev.between(100, 300), "mlbam_id"]
        for label, ids, min_pa in (("focus 100-300", pd.Index(focus), 100), ("all", None, 200)):
            sc = score_rates(p, act, RATES, min_pa, ids=ids)
            sc["season"], sc["pop"] = T, label
            results.append(sc.reset_index(names="system"))
    r = pd.concat(results, ignore_index=True)
    r.to_csv(CACHE / "backtest_hitters_final.csv", index=False)
    pd.set_option("display.width", 200)
    for pop, g in r.groupby("pop"):
        print(f"\n== {pop}")
        print(g.pivot_table(index="season", columns="system", values="wOBA").round(4).to_string())
        pooled = g.groupby("system").apply(lambda d: pd.Series(
            {s: np.sqrt(np.average(d[s] ** 2, weights=d.n)) for s in RATES} | {"n": d.n.sum()}))
        print(pooled.sort_values("wOBA").round(4).to_string())
    s = pd.Series(all_slopes[2026]).unstack()
    print("\ncalibration slopes (fit on 2022-2025):\n" + s.round(2).to_string())


if __name__ == "__main__":
    main()
