"""Per-component calibration slopes for the focus population (100-300 PA in T-1).

slope < 1: projections too spread out (under-shrunk); > 1: too timid.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goose.data.http import ROOT  # noqa: E402

variant = sys.argv[1]
s = pd.read_parquet(ROOT / "data/features/hitter_seasons.parquet")
s["contact"] = s.ab - s.k + s.sf
s["h"] = s.single + s.double + s.triple + s.hr
act = {"k": ("k", "pa"), "bb": ("bb", "pa"), "hr": ("hr", "contact"),
       "babip": (lambda d: d.h - d.hr, lambda d: d.contact - d.hr), "xbh": (lambda d: d.double + d.triple, lambda d: d.h - d.hr)}
rows = []
for T in range(2022, 2027):
    g = pd.read_parquet(ROOT / f"data/models/hitter_proj_{variant}_{T}.parquet")
    g = g[g.pa_prev.between(100, 300)]
    a = s[s.season == T].set_index("mlbam_id")
    for c, (n, d) in act.items():
        num = n(a) if callable(n) else a[n]
        den = d(a) if callable(d) else a[d]
        y = (num / den).reindex(g.mlbam_id).to_numpy()
        w = den.reindex(g.mlbam_id).to_numpy()
        col = {"k": "K%", "bb": "BB%"}.get(c)
        if col:
            proj = g[col].to_numpy()
        elif c == "hr":
            proj = (g.HR / (g.AB - g.K + g.SF)).to_numpy()
        elif c == "babip":
            proj = g.BABIP.to_numpy()
        else:
            proj = ((g["2B"] + g["3B"]) / (g.H - g.HR)).to_numpy()
        rows.append(pd.DataFrame({"c": c, "T": T, "y": y, "p": proj, "w": w}))
r = pd.concat(rows).dropna()
r = r[r.w >= 30]
for c, d in r.groupby("c"):
    pm, ym = np.average(d.p, weights=d.w), np.average(d.y, weights=d.w)
    b = np.sum(d.w * (d.p - pm) * (d.y - ym)) / np.sum(d.w * (d.p - pm) ** 2)
    print(f"{c:6s} n={len(d):4d} slope={b:.2f} proj_sd={np.sqrt(np.average((d.p-pm)**2, weights=d.w)):.4f}")
