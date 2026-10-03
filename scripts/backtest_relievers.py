"""Backtest reliever IP / SV / HLD / K vs Steamer for pitchers in a relief role in T-1, 2022-2026."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goose.data.http import ROOT  # noqa: E402
from goose.eval.competitors import actual_pitchers, load  # noqa: E402
from goose.models.pitcher_pt import build_pt_frame  # noqa: E402
from goose.models.reliever import RelieverModel, build_rp_frame, is_reliever  # noqa: E402

FEAT = ROOT / "data" / "features"

f = build_rp_frame(build_pt_frame(pd.read_parquet(FEAT / "pitcher_seasons.parquet")))
rows = []
for T in range(2022, 2027):
    cur = f[(f.season == T - 1) & is_reliever(f)]
    pr = RelieverModel().fit(f, T - 1).predict(cur)
    pr = pr.assign(id=cur.mlbam_id.to_numpy(), sv_prev=cur.sv.to_numpy(), T=T,
                   k_rate=(cur.k / cur.bf.replace(0, np.nan)).to_numpy())
    st = load("pitchers", T).query("system == 'steamer'")[["mlbam_id", "IP", "SV", "HLD"]]
    st.columns = ["id", "IP_st", "SV_st", "HLD_st"]
    a = actual_pitchers(T)[["mlbam_id", "IP", "SV", "HLD"]]
    a.columns = ["id", "IP_a", "SV_a", "HLD_a"]
    d = pr.merge(st, on="id").merge(a, on="id", how="left").fillna({"IP_a": 0, "SV_a": 0, "HLD_a": 0})
    rows.append(d)
d = pd.concat(rows)
print(f"relievers n={len(d)}")
for stat in ("IP", "SV", "HLD"):
    x = d
    e = lambda c: np.sqrt(((x[c] - x[f"{stat}_a"]) ** 2).mean())  # noqa: E731
    print(f"  {stat:4s} ours {e(stat):5.2f} steamer {e(stat + '_st'):5.2f} blend "
          f"{np.sqrt(((0.5 * x[stat] + 0.5 * x[stat + '_st'] - x[stat + '_a']) ** 2).mean()):5.2f}"
          f" | bias ours {(x[stat] - x[stat + '_a']).mean():5.2f} steamer {(x[stat + '_st'] - x[stat + '_a']).mean():5.2f}")
closers = d[(d.SV_a >= 10) | (d.SV_st >= 10) | (d.SV >= 10)]
print(f"  closer-relevant (n={len(closers)}): SV ours {np.sqrt(((closers.SV - closers.SV_a) ** 2).mean()):.2f} "
      f"steamer {np.sqrt(((closers.SV_st - closers.SV_a) ** 2).mean()):.2f}; P10-P90 SV coverage "
      f"{((d.SV_a >= d.SV_p10) & (d.SV_a <= d.SV_p90)).mean():.2f}; spearman SV ours "
      f"{d[['SV', 'SV_a']].corr('spearman').iloc[0, 1]:.3f} steamer {d[['SV_st', 'SV_a']].corr('spearman').iloc[0, 1]:.3f}")
