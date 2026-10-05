"""2027 projections from end-of-2026 data: hitters and starting pitchers.

Rates: empirical-Bayes component models (low-sample model for <= 1000 career PA), calibrated
on 2022-2026 out-of-sample backtests. Playing time: gradient-boosted PA / IP / GS with quantiles.
Fragility: playing-time risk (P(<300 PA) / IP range, injury flags) + outcome volatility.

Writes data/projections/2027_hitters.csv and 2027_pitchers.csv (all players; `focus` marks
hitters with 100-300 PA and starters with 0-80 IP in 2026).
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
from goose.eval.competitors import actual_hitters  # noqa: E402
from goose.models import calibrate, injury_adjust, volatility  # noqa: E402
from goose.models.hitter_rates import HitterRates, HitterRatesPed  # noqa: E402
from goose.models.pitcher_pt import (ConversionModel, StarterPlayingTime, build_pt_frame as pitcher_pt_frame,  # noqa: E402
                                     is_starter, role_group, rotation_claim)
from goose.models.pitcher_rates import PitcherRates  # noqa: E402
from goose.models.reliever import RelieverModel, build_rp_frame, is_reliever, team_role_context  # noqa: E402
from goose.models.saves_share import SavesShare  # noqa: E402
from goose.models.platoon import Platoon  # noqa: E402
from goose.models.playing_time import (HitterPlayingTime, build_pt_frame as hitter_pt_frame,  # noqa: E402
                                       opportunity, position_depth)
from goose.models.statline import hitter_line, pitcher_line  # noqa: E402

FEAT = ROOT / "data" / "features"
MODELS = ROOT / "data" / "models"
OUT = ROOT / "data" / "projections"
BASE = 2026
FULL, LOW = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("park", "park_low")


def rate_cls(variant: str):
    """Pedigree models for variants named ped*/final*; plain otherwise."""
    return HitterRatesPed if variant.startswith(("ped", "final")) else HitterRates
OVERRIDES = ROOT / "data" / "raw" / "user" / "overrides_2027.csv"
STUFF_FILE = ROOT / "data" / "raw" / "user" / "stuffplus_2026_sep11.csv"
BACKTEST_TARGETS = range(2022, 2027)


def cached(path: Path, make):
    if path.exists():
        return pickle.loads(path.read_bytes())
    m = make()
    path.write_bytes(pickle.dumps(m))
    return m


FA_FILE = ROOT / "data" / "raw" / "user" / "free_agents_2027.csv"


def fa_proxy(s: pd.DataFrame) -> pd.Series:
    """Free agents from data/raw/user/free_agents_2027.csv (column mlbam_id or Name/name).

    Flagged players get a neutral park and no org-specific prospect threat. With no file,
    nobody is flagged: an age/service guess wrongly flagged stars on long contracts (e.g. Judge).
    """
    flag = pd.Series(False, index=s.index)
    # Payroll exports: players whose 2027 column reads FREE AGENT (contract or service time ends).
    pay = ROOT / "data" / "processed" / "contracts_2026.parquet"
    if pay.exists():
        c = pd.read_parquet(pay)
        flag |= s.index.isin(c.loc[c.status_2027 == "free_agent", "mlbam_id"].dropna().astype(int))
    if FA_FILE.exists():
        fa = pd.read_csv(FA_FILE)
        if "mlbam_id" in fa:
            flag |= s.index.isin(fa.mlbam_id)
        name_col = next((c for c in ("Name", "name", "Player", "player") if c in fa), None)
        if name_col and "name" in s:
            flag |= s.name.isin(fa[name_col])
    return flag


def oos_hitter_projections() -> pd.DataFrame:
    frames = []
    for T in BACKTEST_TARGETS:
        low = pd.read_parquet(MODELS / f"hitter_proj_{LOW}_{T}.parquet")
        full = pd.read_parquet(MODELS / f"hitter_proj_{FULL}_{T}.parquet").set_index("mlbam_id")
        use = low.career_prev <= 1000
        frames.append(pd.concat([low[use], full.reindex(low.loc[~use, "mlbam_id"]).dropna(subset=["wOBA"]).reset_index()], ignore_index=True))
    return pd.concat(frames, ignore_index=True)


def fit_volatility(oos: pd.DataFrame, seasons: pd.DataFrame) -> dict:
    rows = []
    for T, g in oos.groupby("season_prev"):
        T = int(T) + 1
        rates = g[[c for c in g if c.startswith("r_") and not c.endswith(("_n", "_prior"))]].rename(columns=lambda c: c[2:])
        line = hitter_line(rates, pd.Series(600.0, index=g.index))
        a = actual_hitters(T).set_index("mlbam_id")
        d = volatility.profile(seasons, g.mlbam_id, T - 1, g.career_prev)
        d["resid"] = g.mlbam_id.map(a.wOBA) - line.wOBA
        d = d[g.mlbam_id.map(a.PA).fillna(0).to_numpy() >= 150]
        d["resid"] -= d.resid.mean()
        rows.append(d)
    return volatility.fit(pd.concat(rows))


def project_hitters() -> pd.DataFrame:
    seasons = pd.read_parquet(FEAT / "hitter_seasons.parquet")
    frame = pd.read_parquet(FEAT / "hitter_frame.parquet")
    cur = frame[frame.season == BASE].copy()
    info = seasons[seasons.season == BASE].set_index("mlbam_id")
    fa = cur.mlbam_id.map(fa_proxy(info)).fillna(False)
    cur.loc[fa.to_numpy(), "team"] = None  # neutral park for likely free agents

    full = cached(MODELS / f"hitter_rates_{FULL}_{BASE}.pkl", lambda: rate_cls(FULL)().fit(frame, BASE))
    low = cached(MODELS / f"hitter_rates_{LOW}_{BASE}.pkl", lambda: rate_cls(LOW)().fit(frame, BASE, 1000))
    career = cur.career_pa_before.fillna(0) + cur.pa.fillna(0)
    r_full, r_low = full.predict(cur), low.predict(cur)
    use_low = (career <= 1000).to_numpy()
    rates = r_full.copy()
    rates.loc[use_low] = r_low.loc[use_low]

    # Calibration slopes from all out-of-sample backtest seasons.
    oos = oos_hitter_projections()
    slopes = calibrate.fit(oos, calibrate.hitter_actuals(seasons))
    df = rates.rename(columns=lambda c: f"r_{c}").assign(career_prev=career.to_numpy(), season_prev=BASE,
                                                       mlbam_id=cur.mlbam_id.to_numpy())
    df = calibrate.apply(df, slopes)
    rates = df[[f"r_{c}" for c in calibrate.CALIBRATED]].rename(columns=lambda c: c[2:])

    # Playing time.
    ptf = hitter_pt_frame(seasons)
    ptf.loc[ptf.mlbam_id.isin(cur.mlbam_id[fa]) & (ptf.season == BASE), "prospect_threat"] = np.nan
    pt_model = HitterPlayingTime().fit(ptf, BASE)
    # 2027 openings: departing free agents no longer block anyone.
    now = ptf.season == BASE
    ptf.loc[now, "hole_woba"] = opportunity(ptf[now], exclude_ids=set(cur.mlbam_id[fa])).hole_woba
    dep = position_depth(ptf[now], exclude_ids=set(cur.mlbam_id[fa]))
    ptf.loc[now, ["pos_rivals", "pos_rival_best"]] = dep.to_numpy()
    pt_rows = ptf[ptf.season == BASE].set_index("mlbam_id").reindex(cur.mlbam_id)
    pt = pt_model.predict(pt_rows.reset_index()).set_index(cur.index)

    # Platoon: re-weight rates for expected matchups vs LHP (LHH only; league-ish split).
    pl = Platoon().fit(BASE + 1).project(cur.mlbam_id, BASE)
    d_exp = (pl.pal_share_next - pl.pal_share_past).fillna(0)
    woba_now = hitter_line(rates, pt.PA).wOBA
    delta = -d_exp * pl.split.fillna(0)              # fewer lefty matchups -> better overall line
    f = (1 + delta / woba_now.clip(lower=0.2)).clip(0.9, 1.1)
    for c in ("hr", "babip", "bb"):
        rates[c] = rates[c] * f
    line = hitter_line(rates, pt.PA)
    coefs = json.loads((FEAT / "hitter_r_rbi_coefs.json").read_text())
    slot = cur.mlbam_id.map(info.lineup_slot).fillna(6.5)
    per_pa = pd.DataFrame({"obp": line.OBP, "iso": line.ISO, "slg": line.SLG, "sb_pa": line.SB / line.PA,
                           "hr_pa": line.HR / line.PA, "slot": slot, "top3": (slot <= 3.5).astype(float)})
    for stat in ("r", "rbi"):
        c = coefs[stat]
        line[stat.upper()] = line.PA * (c["const"] + sum(per_pa[k] * v for k, v in c.items() if k != "const")).clip(lower=0)

    vol_model = fit_volatility(oos, seasons)
    sigma = volatility.predict(volatility.profile(seasons, cur.mlbam_id, BASE, career), vol_model)

    out = pd.DataFrame({"mlbam_id": cur.mlbam_id, "Name": cur.name, "Team_2026": cur.mlbam_id.map(info.team),
                        "Age_2027": (cur.age + 1).round(1), "Pos": cur.mlbam_id.map(info.primary_position),
                        "PA_2026": cur.pa, "career_PA": career})
    out = out.join(pt.round({"PA": 0, "PA_p10": 0, "PA_p50": 0, "PA_p90": 0, "P_full_season": 2})).join(line.drop(columns=["PA"]))
    out["platoon_role"] = pl.platoon_role.to_numpy()
    out["pa_vs_lhp_share"] = pl.pal_share_next.round(3).to_numpy()
    e = pl.pal_share_next.fillna(0.29)
    sp = pl.split.fillna(0)
    out["wOBA_vsR"] = (out.wOBA + e * sp).round(3)
    out["wOBA_vsL"] = (out.wOBA - (1 - e) * sp).round(3)
    out["wOBA_p10"] = out.wOBA - 1.2816 * sigma
    out["wOBA_p90"] = out.wOBA + 1.2816 * sigma
    out["volatility_pctile"] = sigma.rank(pct=True).round(2)
    out["pt_risk"] = (1 - pt.P_300PA).round(2)
    out["fragility"] = (0.5 * out.volatility_pctile + 0.5 * out.pt_risk.rank(pct=True)).round(2)
    out["free_agent"] = fa.astype(int)
    for c in ("options_left", "prospect_threat", "il_days"):
        out[c] = cur.mlbam_id.map(info[c])
    out["focus"] = cur.pa.between(100, 300).astype(int)
    absent = cur.absent.fillna(0).to_numpy() == 1
    out["status"] = np.where(~absent, "MLB", np.where(out.il_days.fillna(0) >= 100, "Missed 2026", "Minors"))
    out["PA_healthy"] = pt.PA_healthy.round(0)
    for c in ("fv_last", "best_rank"):
        out[c] = cur.mlbam_id.map(info[c]) if c in info else np.nan
    return out


def stuff_adjust(rates: pd.DataFrame, cur: pd.DataFrame, info: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Nudge K%/BB% of low-sample pitchers toward what their Stuff+/Location+ imply.

    Uses the September 2026 Stuff+ report (MLB, AAA, some A). Single snapshot, so this is NOT
    backtested: the mapping is a cross-sectional fit on 2026 MLB pitchers, and the weight is
    deliberately small and limited to pitchers with < 600 career BF.
    """
    if not STUFF_FILE.exists():
        return rates, pd.Series(np.nan, index=rates.index)
    sp = pd.read_csv(STUFF_FILE)
    sp = sp.sort_values("n_pitches").groupby("pitcher").tail(1)  # largest sample, any level
    mlb = sp[(sp.level == "MLB") & (sp.n_pitches >= 500)].merge(
        info[["k", "bb", "bf"]].reset_index(), left_on="pitcher", right_on="mlbam_id")
    X = np.c_[np.ones(len(mlb)), mlb["S+"], mlb["L+"]]
    bk = np.linalg.lstsq(X, mlb.k / mlb.bf, rcond=None)[0]
    bb = np.linalg.lstsq(X, mlb.bb / mlb.bf, rcond=None)[0]
    row = cur.mlbam_id.map(sp.set_index("pitcher")[["S+"]]["S+"])
    loc = cur.mlbam_id.map(sp.set_index("pitcher")["L+"])
    n = cur.mlbam_id.map(sp.set_index("pitcher").n_pitches).fillna(0)
    career = cur.career_bf_before.fillna(0) + cur.bf.fillna(0)
    w = (0.35 * n / (n + 600) * (career < 600)).where(row.notna(), 0.0)
    k_s = bk[0] + bk[1] * row + bk[2] * loc
    b_s = bb[0] + bb[1] * row + bb[2] * loc
    out = rates.copy()
    out["k"] = np.where(w > 0, (1 - w) * rates.k + w * k_s, rates.k)
    out["bb"] = np.where(w > 0, (1 - w) * rates.bb + w * b_s.clip(0.03, 0.2), rates.bb)
    return out, row


def project_pitchers() -> pd.DataFrame:
    seasons = pd.read_parquet(FEAT / "pitcher_seasons.parquet")
    frame = pd.read_parquet(FEAT / "pitcher_frame.parquet")
    info = seasons[seasons.season == BASE].set_index("mlbam_id")
    cur = frame[frame.season == BASE].copy()
    ptf_now = pitcher_pt_frame(seasons)
    ptf_now = ptf_now[ptf_now.season == BASE].set_index("mlbam_id")
    starters = is_starter(ptf_now)
    cur = cur[cur.mlbam_id.map(starters).fillna(False).to_numpy()]
    fa = cur.mlbam_id.map(fa_proxy(info)).fillna(False)
    cur.loc[fa.to_numpy(), "team"] = None

    rates_m = cached(MODELS / f"pitcher_rates_xblend_nopark_{BASE}.pkl", lambda: PitcherRates().fit(frame, BASE))
    rates = rates_m.predict(cur)
    rates, stuff = stuff_adjust(rates, cur, info)
    ptf = pitcher_pt_frame(seasons)
    pt_model = StarterPlayingTime().fit(ptf, BASE)
    pt = pt_model.predict(ptf[ptf.season == BASE].set_index("mlbam_id").reindex(cur.mlbam_id).reset_index())
    pt.index = cur.index
    coefs = json.loads((FEAT / "pitcher_line_coefs.json").read_text())
    line = pitcher_line(rates, pt.IP, pt.GS, coefs, BASE)
    # Healthy workload (used when a known surgery return date sets the playing time).
    from goose.models.pitcher_pt import _gap_x, pre_injury_baseline, workload_baseline
    rows_now = ptf[ptf.season == BASE].set_index("mlbam_id").reindex(cur.mlbam_id)
    hb = np.maximum(workload_baseline(rows_now), pre_injury_baseline(rows_now))
    healthy_center = np.clip(hb + _gap_x(hb, rows_now.age.to_numpy(), rows_now.kbb_shrunk.to_numpy(),
                                         rows_now.team_wpct.to_numpy()) @ pt_model.gap_b, 0, 215)

    out = pd.DataFrame({"mlbam_id": cur.mlbam_id, "Name": cur.name, "Team_2026": cur.mlbam_id.map(info.team),
                        "Age_2027": (cur.age + 1).round(1), "Throws": cur.throws,
                        "IP_2026": cur.mlbam_id.map(info.ip).round(1), "GS_2026": cur.mlbam_id.map(info.gs),
                        "total_IP_2026": cur.mlbam_id.map(info.total_ip).round(1)})
    out = out.join(line).join(pt[["IP_p10", "IP_p50", "IP_p90", "P_100IP", "P_150IP", "IP_healthy", "IP_expected",
                                  "P_full_season"]].round(2))
    out["stuff_plus"] = stuff.round(1)
    out["IP_healthy_center"] = np.round(healthy_center, 1)
    out["SV"], out["HLD"] = 0.0, 0.0
    l1 = ptf[ptf.season == BASE].set_index("mlbam_id")
    arm = l1.il_arm.fillna(0) + l1.il_arm_l1.fillna(0) + l1.il_surgery.fillna(0) + l1.il_surgery_l1.fillna(0) \
        + l1.il_shoulder.fillna(0)
    out["arm_injury_2y"] = cur.mlbam_id.map(arm > 0).astype(int)
    out["il_days_2026"] = cur.mlbam_id.map(info.il_days)
    out["ip_range_width"] = out.IP_p90 - out.IP_p10
    # Fragility: chance of falling short of a useful workload, plus injury history.
    out["fragility"] = (0.6 * (1 - out.P_100IP).rank(pct=True) + 0.4 * out.arm_injury_2y).round(2)
    out["free_agent"] = fa.astype(int)
    out["options_left"] = cur.mlbam_id.map(info.options_left)
    out["focus"] = ((out.IP_2026 > 0) & (out.IP_2026 <= 80)).astype(int)
    absent = cur.absent.fillna(0).to_numpy() == 1
    out["status"] = np.where(~absent, "MLB", np.where(out.il_days_2026.fillna(0) >= 100, "Missed 2026", "Minors"))
    return out


def project_relievers() -> pd.DataFrame:
    """Everyone who pitched meaningfully in relief in 2026 (swingmen also appear as starters)."""
    seasons = pd.read_parquet(FEAT / "pitcher_seasons.parquet")
    frame = pd.read_parquet(FEAT / "pitcher_frame.parquet")
    info = seasons[seasons.season == BASE].set_index("mlbam_id")
    rpf = build_rp_frame(pitcher_pt_frame(seasons))
    now = rpf[(rpf.season == BASE) & is_reliever(rpf)]
    fa_ids = set(info.index[fa_proxy(info)])
    # 2027 role context: departing free agents no longer hold the 9th.
    now = now.drop(columns=["team_sv_other", "team_hld_other"]).join(team_role_context(now, exclude_ids=fa_ids))
    model = RelieverModel().fit(rpf, BASE)
    pr = model.predict(now)
    # Saves: average the per-player model with the within-team share model (backtest: better than
    # either alone). Departing free agents are out of the competition; their saves stay in the pool.
    share = SavesShare().fit(rpf, BASE).predict(now, eligible=~now.mlbam_id.isin(fa_ids).to_numpy())
    sv_scale = (0.5 * pr.SV + 0.5 * share.SV_share_model) / pr.SV.clip(lower=0.1)
    pr["SV"] = 0.5 * pr.SV + 0.5 * share.SV_share_model
    for c in ("SV_p10", "SV_p50", "SV_p90"):
        pr[c] = pr[c] * sv_scale.clip(upper=3)
    pr["team_sv_share"] = share.sv_share_next
    cur = frame[frame.season == BASE].set_index("mlbam_id").reindex(now.mlbam_id).reset_index()
    cur.index = now.index
    fa = cur.mlbam_id.isin(fa_ids)
    cur.loc[fa, "team"] = None
    rates_m = cached(MODELS / f"pitcher_rates_xblend_nopark_{BASE}.pkl", lambda: PitcherRates().fit(frame, BASE))
    rates, stuff = stuff_adjust(rates_m.predict(cur), cur, info)
    coefs = json.loads((FEAT / "pitcher_line_coefs.json").read_text())
    line = pitcher_line(rates, pr.IP, pd.Series(0.0, index=pr.index), coefs, BASE)
    out = pd.DataFrame({"mlbam_id": now.mlbam_id, "Name": now.name, "Team_2026": now.team,
                        "Age_2027": (now.age + 1).round(1), "Throws": now.mlbam_id.map(info.throws),
                        "IP_2026": now.ip.round(1), "G_2026": now.games, "GS_2026": now.gs,
                        "SV_2026": now.sv, "HLD_2026": now.hld})
    out = out.join(line.drop(columns=["GS", "QS"])).join(pr.drop(columns=["IP"]).round(2))
    out["stuff_plus"] = stuff.round(1)
    out["free_agent"] = fa.astype(int).to_numpy()
    out["swingman"] = (now.gs >= 3).astype(int).to_numpy()
    out["il_days_2026"] = now.il_days
    out["fragility"] = (0.5 * (1 - out.IP / out.IP_p90.clip(lower=1)).rank(pct=True)
                        + 0.5 * (now.il_arm.fillna(0) + now.il_days.fillna(0).gt(30)).clip(upper=1).to_numpy()).round(2)
    out["status"] = "MLB"
    return out


HIT_COUNTS = ["PA", "PA_p10", "PA_p50", "PA_p90", "HR", "R", "RBI", "SB", "CS", "BB", "K", "H", "1B", "2B", "3B",
              "AB", "HBP", "SF"]
PIT_COUNTS = ["IP", "IP_p10", "IP_p50", "IP_p90", "GS", "TBF", "K", "BB", "HBP", "HR", "H", "ER", "QS", "W", "L",
              "SV", "HLD", "SV_p10", "SV_p50", "SV_p90"]


def apply_injuries(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Scale playing time for players recovering from surgery (injury report), flagged."""
    pitcher = kind != "hitters"
    adj = injury_adjust.adjustments(df.mlbam_id, pitcher)
    df["injury_note"] = adj.injury_note
    hit = adj.injury_note != ""
    if not hit.any():
        return df
    vol = "PA" if not pitcher else "IP"
    f = (adj.avail * adj.ramp)[hit]
    # Start from the HEALTHY workload (the model's own number already discounts for injury risk),
    # then apply the share of the season he's available and a ramp-up discount.
    if not pitcher:
        healthy = df.loc[hit, "PA_healthy"].fillna(df.loc[hit, "PA"])
    elif kind == "relievers":
        # His own recent relief workload (2026, else the model), bounded to a reliever's range.
        recent = df.loc[hit, "IP_2026"].where(df.loc[hit, "IP_2026"] >= 20)
        healthy = np.maximum(recent.fillna(df.loc[hit, "IP"]), df.loc[hit, "IP"]).clip(20, 80)
    else:
        healthy = df.loc[hit, "IP_healthy_center"].fillna(df.loc[hit, "IP"])
    # Post-return pace from the historical return model: headline = "returns as planned" hump.
    ratios = json.loads((MODELS / "return_ratios.json").read_text())["pitcher" if pitcher else "hitter"]
    grp = np.where(adj.arm[hit], "arm_or_surgery", "other")
    plan = np.array([ratios[g]["plan_ratio"] for g in grp])
    p_plan = np.array([ratios[g]["p_plan"] for g in grp])
    q_lo = np.array([ratios[g]["plan_q"][0] for g in grp])
    q_hi = np.array([ratios[g]["plan_q"][1] for g in grp])
    # Everyone not yet back from surgery uses the return model (season share = 1 if due by Opening Day).
    # Hitters: only returns after Opening Day use the return-pace model (a hitter recovered over the
    # winter is ready for spring). Pitchers not yet back from surgery still need a build-up.
    if pitcher:
        returning_midseason = ((adj.avail[hit] < 1) | adj.pending[hit].astype(bool)).to_numpy()
    else:
        returning_midseason = (adj.avail[hit] < 1).to_numpy()
    span = (healthy * adj.avail[hit]).to_numpy()
    target = np.where(returning_midseason, span * plan, df.loc[hit, vol])
    new_vol = np.minimum(target, adj.cap[hit])
    scale = new_vol / df.loc[hit, vol].clip(lower=0.1)
    for c in (HIT_COUNTS if not pitcher else PIT_COUNTS):
        if c in df and not c.endswith(("_p10", "_p50", "_p90")):
            df.loc[hit, c] = df.loc[hit, c] * scale
    for c in ("P_300PA", "P_100IP", "P_150IP", "P_full_season", "P_20SV", "P_10SV"):
        if c in df:
            df.loc[hit, c] = df.loc[hit, c] * adj.avail[hit] * np.where(adj.cap[hit] < 150, 0.5, 1.0)
    # Ranges for mid-season returns: setbacks (1 - p_plan) sit at ~0; planned returns span q_lo..q_hi.
    lo, mid, hi = (f"{vol}_p10", f"{vol}_p50", f"{vol}_p90")
    if lo in df:
        idx = df.loc[hit].index[returning_midseason]
        rm = returning_midseason
        df.loc[idx, lo] = np.where(p_plan[rm] < 0.9, 0.0, span[rm] * q_lo[rm])
        df.loc[idx, mid] = np.minimum(span[rm] * plan[rm], adj.cap[hit].to_numpy()[rm])
        df.loc[idx, hi] = np.minimum(span[rm] * q_hi[rm], adj.cap[hit].to_numpy()[rm])
    if "P_full_season" in df:
        idx = df.loc[hit].index[returning_midseason]
        df.loc[idx, "P_full_season"] = np.where(adj.avail[hit].to_numpy()[returning_midseason] > 0,
                                                p_plan[returning_midseason], 0.0)
    return df


def apply_rotation(df: pd.DataFrame) -> pd.DataFrame:
    """Team rotation depth: rank each team's candidates by rotation claim (free agents and pitchers
    out for 2027 excluded), attach the historical P(rotation) for that rank, and give top-5
    converts / swingmen the rotation-hump workload (never below their 2026 total)."""
    cfg = json.loads((MODELS / "rotation_ranks.json").read_text())
    seasons = pd.read_parquet(FEAT / "pitcher_seasons.parquet")
    ptf = pitcher_pt_frame(seasons)
    now = ptf[ptf.season == BASE].set_index("mlbam_id").reindex(df.mlbam_id)
    claim = rotation_claim(now).to_numpy()
    eligible = (df.free_agent.fillna(0) == 0).to_numpy() & (df.IP.fillna(0) > 15).to_numpy()
    ranks = pd.Series(np.nan, index=df.index)
    tmp = pd.DataFrame({"team": df.Team_2026.to_numpy(), "claim": np.where(eligible, claim, -1e9)}, index=df.index)
    ranks[:] = tmp.groupby("team").claim.rank(ascending=False, method="first").to_numpy()
    ranks[~eligible] = np.nan
    df["rotation_rank"] = ranks
    # P(holds a rotation spot): logistic on rank, quality, team, workload and MLB share of innings
    # (leave-one-season-out AUC 0.90 vs 0.88 for rank alone).
    lg = cfg["p_rotation_logit"]
    Xr = np.nan_to_num(np.column_stack([
        np.minimum(ranks.fillna(9), 9), (now.kbb_shrunk.fillna(.13) - .13).to_numpy() / .04,
        now.team_wpct.fillna(.5).to_numpy() - .5, now.total_ip.fillna(0).to_numpy() / 100,
        (now.ip.fillna(0) / now.total_ip.clip(lower=1)).clip(0, 1).to_numpy(), (now.age.fillna(28).to_numpy() - 28) / 5]))
    df["P_rotation"] = np.where(ranks.notna(), 1 / (1 + np.exp(-(Xr @ np.array(lg["coef"]) + lg["intercept"]))), np.nan)
    grp = role_group(now).to_numpy()
    df["role_2026"] = grp
    total26 = now.total_ip.fillna(0).to_numpy()
    in_rot = (ranks <= 5).to_numpy()
    # In-rotation workload for converts / swingmen: fitted on those who held a rotation spot
    # (next IP = a + b * prior total IP + c * K-BB z), not a flat multiplier. Never more than 10 IP
    # above his career high; never below his 2026 total if he's still stretching out (< 140 IP).
    a, b, c = cfg["rotation_ip_fit"]
    kbb_z = ((now.kbb_shrunk.fillna(.13) - .13) / .04).to_numpy()
    career_high = seasons[seasons.season <= BASE].groupby("mlbam_id").total_ip.max()
    high = df.mlbam_id.map(career_high).fillna(0).to_numpy()
    fitted = a + b * np.clip(total26, 40, 200) + c * kbb_z
    target = np.minimum(fitted, high + 10)
    target = np.where(total26 < 140, np.maximum(target, total26), target)
    sel = in_rot & np.isin(grp, ["convert", "mixed"]) & (df.injury_note.fillna("").to_numpy() == "")
    # Swingmen with a real starter history: their healthy starter workload, same career-high cap.
    healthy = np.minimum(df.IP_healthy_center.fillna(0).to_numpy(), high + 10)
    target = np.where(grp == "mixed", np.maximum(target, healthy), target)
    target = np.clip(target, 0, 200)
    full = target.copy()
    # Headline = in-rotation workload only when a rotation spot is the likely outcome (>= 50%);
    # otherwise the expected value (out-of-rotation arms throw ~45 IP).
    p_rot = df.P_rotation.fillna(0).to_numpy()
    ramp = np.clip((p_rot - 0.3) / 0.2, 0, 1)
    expected = p_rot * target + (1 - p_rot) * 45
    target = expected + ramp * (target - expected)
    sel = np.isin(grp, ["convert", "mixed"]) & (df.injury_note.fillna("").to_numpy() == "") & (ranks <= 9).to_numpy()
    f = np.where(sel, target / np.maximum(df.IP.to_numpy(), 0.1), 1.0)
    for col in ["IP", "TBF", "K", "BB", "HBP", "HR", "H", "ER", "QS", "W", "L", "IP_p50", "IP_p90"]:
        if col in df:
            df.loc[sel, col] = (df[col].to_numpy() * f)[sel]
    # Scaling the old range by the same factor can push P90 far past any plausible workload (it was
    # built around a part-time projection); the ceiling is the full in-rotation workload.
    cap = np.maximum(df.IP.to_numpy(), full)
    df.loc[sel, "IP_p90"] = np.minimum(df.IP_p90.to_numpy(), cap)[sel]
    df.loc[sel, "IP_p50"] = np.minimum(df.IP_p50.to_numpy(), cap)[sel]
    df.loc[sel, "GS"] = np.clip(df.loc[sel, "IP"] / 5.6, 0, 33)
    df.loc[sel, "P_full_season"] = df.loc[sel, "P_rotation"]
    return df


def add_conversion_candidates(starters: pd.DataFrame, relievers: pd.DataFrame) -> pd.DataFrame:
    """Relievers with a real chance to move into the rotation (the Jax route) also get a starter row.
    Their starter IP mixes the converted workload with their relief workload, using the same
    30-50% ramp as everyone else; most stay near their relief workload because conversion is rare."""
    seasons = pd.read_parquet(FEAT / "pitcher_seasons.parquet")
    ptf = pitcher_pt_frame(seasons)
    cm = ConversionModel().fit(ptf, BASE)
    now = ptf[ptf.season == BASE].set_index("mlbam_id").reindex(relievers.mlbam_id)
    pr = cm.predict(now.reset_index()).set_index(relievers.index)
    relievers["P_convert"] = pr.P_convert.round(3)
    cand = relievers[(pr.P_convert >= 0.03) & ~relievers.mlbam_id.isin(starters.mlbam_id) & (relievers.free_agent == 0)
                     & (relievers.injury_note.fillna("") == "")].copy()
    if cand.empty:
        return starters
    p = pr.P_convert[cand.index].to_numpy()
    conv_ip, rp_ip = pr.IP_if_converted[cand.index].to_numpy(), cand.IP.to_numpy()
    expected = p * conv_ip + (1 - p) * rp_ip
    ramp = np.clip((p - 0.3) / 0.2, 0, 1)
    ip = expected + ramp * (conv_ip - expected)
    f = ip / np.maximum(rp_ip, 1)
    for c in ("K", "BB", "HBP", "HR", "H", "ER", "W", "L", "TBF"):
        if c in cand:
            cand[c] = cand[c] * f
    cand["IP"], cand["GS"] = ip, p * conv_ip / 5.3
    cand["IP_p10"], cand["IP_p50"], cand["IP_p90"] = rp_ip * 0.6, ip, conv_ip
    cand["QS"], cand["P_rotation"], cand["role_2026"] = cand.GS * 0.35, p, "conversion"
    cand["P_full_season"] = p
    cand["SV"], cand["HLD"] = 0.0, 0.0
    keep = [c for c in starters.columns if c in cand.columns]
    return pd.concat([starters, cand[keep]], ignore_index=True)


def apply_overrides(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Hand overrides (data/raw/user/overrides_2027.csv: kind,mlbam_id,field,value,note).

    Used for edge cases the model can't see (e.g. a traded top prospect with a clear rotation
    spot). Counting stats are rescaled when the volume field changes; every override is flagged.
    """
    df["override"] = ""
    if not OVERRIDES.exists():
        return df
    ov = pd.read_csv(OVERRIDES)
    ov = ov[ov.kind == kind]
    vol = "PA" if kind == "hitters" else "IP"
    counts = (["HR", "R", "RBI", "SB", "CS", "BB", "K", "H", "1B", "2B", "3B", "AB", "HBP", "SF"] if kind == "hitters"
              else ["GS", "TBF", "K", "BB", "HBP", "HR", "H", "ER", "QS", "W", "L"])
    for r in ov.itertuples():
        sel = df.mlbam_id == r.mlbam_id
        if not sel.any():
            continue
        if r.field == vol:
            f = float(r.value) / df.loc[sel, vol].clip(lower=0.1)
            for c in counts:
                if c in df:
                    df.loc[sel, c] = df.loc[sel, c] * f
        df.loc[sel, r.field] = float(r.value)
        df.loc[sel, "override"] = r.note
    return df


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    h = apply_overrides(apply_injuries(project_hitters(), "hitters"), "hitters")
    h.round(4).to_csv(OUT / "2027_hitters.csv", index=False)
    p = apply_overrides(apply_rotation(apply_injuries(project_pitchers(), "pitchers")), "pitchers")
    p.round(4).to_csv(OUT / "2027_pitchers.csv", index=False)
    r = apply_overrides(apply_injuries(project_relievers(), "relievers"), "relievers")
    p = add_conversion_candidates(p, r)
    p.round(4).to_csv(OUT / "2027_pitchers.csv", index=False)
    r.round(4).to_csv(OUT / "2027_relievers.csv", index=False)
    print(f"hitters: {len(h)} ({h.focus.sum()} focus); pitchers: {len(p)} starters ({p.focus.sum()} focus); "
          f"{len(r)} relievers ({r.swingman.sum()} also starters)")
