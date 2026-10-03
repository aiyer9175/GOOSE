"""Rolling-origin backtest for starting pitchers with 0 < IP <= 80 in T-1 (and >= 3 GS, >= 50% starts).

Scores rates (ERA, FIP, WHIP, K%, BB%, K-BB%) on pitchers with 100+ TBF in T, and volume/counting
stats (IP, GS, K, W, QS) on everyone Steamer projected, with non-appearances as 0.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goose.data.http import ROOT  # noqa: E402
from goose.eval.competitors import actual_pitchers, load  # noqa: E402
from goose.models.pitcher_pt import StarterPlayingTime, build_pt_frame, is_starter  # noqa: E402
from goose.models.pitcher_rates import PitcherRates  # noqa: E402
from goose.models.statline import pitcher_line  # noqa: E402

CACHE = ROOT / "data" / "models"
RATES = ["ERA", "FIP", "WHIP", "K%", "BB%", "K-BB%"]
COUNTS = ["IP", "GS", "K", "W", "QS"]


def fit_rates(frame, T, variant):
    path = CACHE / f"pitcher_rates_{variant}_{T - 1}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    m = PitcherRates().fit(frame, T - 1)
    path.write_bytes(pickle.dumps(m))
    return m


def project(T: int, variant: str = "base") -> pd.DataFrame:
    seasons = pd.read_parquet(ROOT / "data" / "features" / "pitcher_seasons.parquet")
    frame = pd.read_parquet(ROOT / "data" / "features" / "pitcher_frame.parquet")
    ptf = build_pt_frame(seasons)
    coefs = json.loads((ROOT / "data" / "features" / "pitcher_line_coefs.json").read_text())
    rates_m = fit_rates(frame, T, variant)
    pt_m = StarterPlayingTime().fit(ptf, T - 1)
    cur = frame[frame.season == T - 1]
    rates = rates_m.predict(cur)
    pt = pt_m.predict(ptf.loc[ptf.season == T - 1].set_index(cur.index))
    line = pitcher_line(rates, pt.IP, pt.GS, coefs, T - 1)
    line = line.join(pt.drop(columns=["IP", "GS"]))
    line["mlbam_id"], line["name"] = cur.mlbam_id.to_numpy(), cur.name.to_numpy()
    prev = seasons[seasons.season == T - 1].set_index("mlbam_id")
    line["ip_prev"] = line.mlbam_id.map(prev.ip)
    line["focus"] = line.mlbam_id.map(is_starter(prev) & (prev.ip > 0) & (prev.ip <= 80))
    return line


def score(T: int, ours: pd.DataFrame) -> list[dict]:
    act = actual_pitchers(T)
    comp = load("pitchers", T)
    st = comp[comp.system == "steamer"]
    zp = comp[comp.system == "zips"]
    f = ours[ours.focus]
    rows = []
    m = f.merge(st, on="mlbam_id", suffixes=("", "_st")).merge(act, on="mlbam_id", how="left", suffixes=("", "_a"))
    m = m.merge(zp[["mlbam_id"] + RATES].add_suffix("_zp").rename(columns={"mlbam_id_zp": "mlbam_id"}),
                on="mlbam_id", how="left")
    r = m[m.TBF_a >= 100]
    w = r.TBF_a

    def wr(e, w):
        e = e - np.average(e, weights=w)
        return float(np.sqrt(np.average(e ** 2, weights=w)))
    for stat in RATES:
        rows.append(dict(season=T, kind="rate", stat=stat, n=len(r), goose=wr(r[stat] - r[f"{stat}_a"], w),
                         steamer=wr(r[f"{stat}_st"] - r[f"{stat}_a"], w),
                         zips=wr(r[f"{stat}_zp"] - r[f"{stat}_a"], w) if r[f"{stat}_zp"].notna().all() else np.nan))
    c = m.copy()
    for stat in COUNTS:
        c[f"{stat}_a"] = c[f"{stat}_a"].fillna(0)
        rows.append(dict(season=T, kind="count", stat=stat, n=len(c),
                         goose=float(np.sqrt(((c[stat] - c[f"{stat}_a"]) ** 2).mean())),
                         steamer=float(np.sqrt(((c[f"{stat}_st"] - c[f"{stat}_a"]) ** 2).mean())),
                         blend=float(np.sqrt(((0.5 * c[stat] + 0.5 * c[f"{stat}_st"] - c[f"{stat}_a"]) ** 2).mean()))))
    cov = ((c.IP_a >= c.IP_p10) & (c.IP_a <= c.IP_p90)).mean()
    rows.append(dict(season=T, kind="calib", stat="IP p10-p90 coverage", n=len(c), goose=float(cov)))
    return rows


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    rows = []
    variant = sys.argv[1] if len(sys.argv) > 1 else "base"
    for T in range(2022, 2027):
        ours = project(T, variant)
        ours.to_parquet(CACHE / f"pitcher_proj_{T}.parquet")
        rows += score(T, ours)
        print(f"{T} done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(CACHE / f"backtest_pitchers_{variant}.csv", index=False)
    print(r.pivot_table(index=["kind", "stat"], columns="season", values=["goose", "steamer"]).round(3).to_string())
    pooled = r[r.kind != "calib"].assign(wn=lambda d: d.n).groupby(["kind", "stat"]).apply(
        lambda d: pd.Series({s: np.sqrt(np.average(d[s] ** 2, weights=d.wn)) for s in ("goose", "steamer", "zips", "blend")
                             if d[s].notna().any()}))
    print("\npooled:\n" + pooled.round(3).to_string())
    print(r[r.kind == "calib"][["season", "goose"]].to_string())
