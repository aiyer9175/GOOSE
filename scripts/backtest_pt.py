"""Backtest blended playing time (hitters PA, starters IP) vs expected-only and Steamer, 2022-2026."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goose.data.http import ROOT  # noqa: E402
from goose.eval.competitors import actual_hitters, actual_pitchers, load  # noqa: E402
from goose.models.pitcher_pt import StarterPlayingTime, build_pt_frame as pbf, is_starter  # noqa: E402
from goose.models.playing_time import HitterPlayingTime, build_pt_frame  # noqa: E402

FEAT = ROOT / "data" / "features"


def report(d, vol, pops):
    for pop, sel in pops:
        x = d[sel]
        rm = lambda c: np.sqrt(((x[c] - x.A) ** 2).mean())  # noqa: E731
        print(f"  {pop:16s} n={len(x):5d} blended={rm(vol):6.1f} expected-only={rm(vol + '_expected'):6.1f} "
              f"steamer={rm('st'):6.1f} | bias blended={(x[vol] - x.A).mean():6.1f} steamer={(x.st - x.A).mean():6.1f}"
              f" | P10-P90 coverage={((x.A >= x[vol + '_p10']) & (x.A <= x[vol + '_p90'])).mean():.2f}")


def main():
    f = build_pt_frame(pd.read_parquet(FEAT / "hitter_seasons.parquet"))
    rows = []
    for T in range(2022, 2027):
        cur = f[f.season == T - 1]
        pr = HitterPlayingTime().fit(f, T - 1).predict(cur)
        pr = pr.assign(id=cur.mlbam_id.to_numpy(), prev=cur.pa.to_numpy(), est=cur.established.to_numpy(),
                       x3=cur.xwoba_3y.to_numpy())
        st = load("hitters", T).query("system == 'steamer'")[["mlbam_id", "PA"]].set_axis(["id", "st"], axis=1)
        a = actual_hitters(T)[["mlbam_id", "PA"]].set_axis(["id", "A"], axis=1)
        rows.append(pr.merge(st, on="id").merge(a, on="id", how="left").fillna({"A": 0}))
    d = pd.concat(rows)
    print("HITTERS (previous model: focus 154.2, all 145.6)")
    report(d, "PA", [("focus 100-300", d.prev.between(100, 300)), ("all MLB", d.prev >= 1),
                     ("elite injured", (d.est == 1) & (d.x3 >= .36) & (d.prev < 450)), ("established", d.est == 1)])

    fp = pbf(pd.read_parquet(FEAT / "pitcher_seasons.parquet"))
    rows = []
    for T in range(2022, 2027):
        cur = fp[fp.season == T - 1]
        cur = cur[is_starter(cur)]
        pr = StarterPlayingTime().fit(fp, T - 1).predict(cur)
        pr = pr.assign(id=cur.mlbam_id.to_numpy(), prev=cur.ip.to_numpy(), s130=cur.seasons_130_5y.to_numpy(),
                       absent=cur.absent.to_numpy())
        st = load("pitchers", T).query("system == 'steamer'")[["mlbam_id", "IP"]].set_axis(["id", "st"], axis=1)
        a = actual_pitchers(T)[["mlbam_id", "IP"]].set_axis(["id", "A"], axis=1)
        rows.append(pr.merge(st, on="id").merge(a, on="id", how="left").fillna({"A": 0}))
    d = pd.concat(rows)
    print("STARTERS (previous model: focus 43.4, all 40.1)")
    report(d, "IP", [("focus 0-80", (d.prev > 0) & (d.prev <= 80)), ("all", d.prev >= 0),
                     ("durable 3+ x130", d.s130 >= 3), ("missed season", d.absent == 1)])


if __name__ == "__main__":
    main()
