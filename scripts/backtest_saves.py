"""Backtest the team-share saves model vs the per-player model and Steamer, 2022-2026."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from goose.data.http import ROOT  # noqa: E402
from goose.eval.competitors import actual_pitchers, load  # noqa: E402
from goose.models.pitcher_pt import build_pt_frame  # noqa: E402
from goose.models.reliever import RelieverModel, build_rp_frame, is_reliever  # noqa: E402
from goose.models.saves_share import SavesShare  # noqa: E402

f = build_rp_frame(build_pt_frame(pd.read_parquet(ROOT / "data/features/pitcher_seasons.parquet")))
rows = []
for T in range(2022, 2027):
    cur = f[(f.season == T - 1) & is_reliever(f)]
    share = SavesShare().fit(f, T - 1).predict(cur)
    old = RelieverModel().fit(f, T - 1).predict(cur)
    d = pd.DataFrame({"id": cur.mlbam_id.to_numpy(), "share": share.SV_share_model.to_numpy(),
                      "old": old.SV.to_numpy(), "T": T})
    st = load("pitchers", T).query("system == 'steamer'")[["mlbam_id", "SV"]].set_axis(["id", "st"], axis=1)
    a = actual_pitchers(T)[["mlbam_id", "SV"]].set_axis(["id", "A"], axis=1)
    rows.append(d.merge(st, on="id").merge(a, on="id", how="left").fillna({"A": 0}))
d = pd.concat(rows)
d["blend_st"] = 0.5 * d.share + 0.5 * d.st
d["ours_avg"] = 0.5 * d.share + 0.5 * d.old
for pop, x in (("all relievers", d), ("closer-relevant", d[(d.A >= 10) | (d.st >= 10) | (d.share >= 10) | (d.old >= 10)])):
    print(f"{pop} (n={len(x)})")
    for c in ("share", "old", "ours_avg", "st", "blend_st"):
        e = x[c] - x.A
        print(f"   {c:9s} RMSE {np.sqrt((e ** 2).mean()):5.2f}  bias {e.mean():+5.2f}  spearman {x[[c, 'A']].corr('spearman').iloc[0, 1]:.3f}")
print(SavesShare().fit(f, 2026).coefs().round(2).to_string())
