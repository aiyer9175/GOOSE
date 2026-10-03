"""Turn projected component rates (+ PA) into a full hitter stat line."""
from __future__ import annotations

import numpy as np
import pandas as pd

WOBA = dict(bb=0.7, hbp=0.7, single=0.9, double=1.25, triple=1.6, hr=2.0)
SF_RATE = 0.0075   # sac flies per PA (league, recent)
SH_RATE = 0.0015   # sac bunts per PA


def hitter_line(rates: pd.DataFrame, pa: pd.Series) -> pd.DataFrame:
    """rates: columns k, bb, hbp, hr, babip, xbh, tri, sb_att, sb_succ (per their denominators)."""
    o = pd.DataFrame(index=rates.index)
    o["PA"] = pa
    o["BB"] = pa * rates.bb
    o["HBP"] = pa * rates.hbp
    o["K"] = pa * rates.k
    o["SF"] = pa * SF_RATE
    o["AB"] = pa - o.BB - o.HBP - o.SF - pa * SH_RATE
    contact = o.AB - o.K + o.SF
    o["HR"] = contact * rates.hr
    hits_nohr = (contact - o.HR) * rates.babip
    xbh = hits_nohr * rates.xbh
    o["3B"] = xbh * rates.tri
    o["2B"] = xbh - o["3B"]
    o["1B"] = hits_nohr - xbh
    o["H"] = o["1B"] + o["2B"] + o["3B"] + o.HR
    o["AVG"] = o.H / o.AB
    o["OBP"] = (o.H + o.BB + o.HBP) / (o.AB + o.BB + o.HBP + o.SF)
    o["SLG"] = (o["1B"] + 2 * o["2B"] + 3 * o["3B"] + 4 * o.HR) / o.AB
    o["OPS"] = o.OBP + o.SLG
    o["ISO"] = o.SLG - o.AVG
    o["wOBA"] = (WOBA["bb"] * o.BB + WOBA["hbp"] * o.HBP + WOBA["single"] * o["1B"] + WOBA["double"] * o["2B"]
                 + WOBA["triple"] * o["3B"] + WOBA["hr"] * o.HR) / (o.AB + o.BB + o.SF + o.HBP)
    o["K%"] = rates.k
    o["BB%"] = rates.bb
    o["BABIP"] = rates.babip
    att = (o["1B"] + o.BB + o.HBP) * rates.sb_att
    o["SB"] = att * rates.sb_succ
    o["CS"] = att - o.SB
    return o


def pitcher_line(rates: pd.DataFrame, ip: pd.Series, gs: pd.Series, coefs: dict, season: int) -> pd.DataFrame:
    """rates: k, bb, hbp (per BF), hr (per contact), babip, xbh. Volume from projected IP and GS."""
    contact = 1 - rates.k - rates.bb - rates.hbp
    per = pd.DataFrame({"k": rates.k, "bb": rates.bb, "hbp": rates.hbp, "hr": contact * rates.hr})
    per["h_nohr"] = (contact - per.hr) * rates.babip
    per["xbh"] = per.h_nohr * rates.xbh
    per["const"] = 1.0

    def lin(name):
        c = coefs[name]
        return sum(per[k] * v for k, v in c.items())

    outs_bf, er_bf = lin("outs_per_bf"), lin("er_per_bf")
    o = pd.DataFrame(index=rates.index)
    o["IP"] = ip
    o["GS"] = gs
    o["TBF"] = ip * 3 / outs_bf
    for col, k in (("K", "k"), ("BB", "bb"), ("HBP", "hbp"), ("HR", "hr")):
        o[col] = o.TBF * per[k]
    o["H"] = o.TBF * (per.h_nohr + per.hr)
    o["ER"] = o.TBF * er_bf
    o["ERA"] = 27 * er_bf / outs_bf
    o["WHIP"] = 3 * (per.bb + per.h_nohr + per.hr) / outs_bf
    fipc = coefs["fip_const"].get(str(season), coefs["fip_const"].get(season, 3.15))
    o["FIP"] = 3 * (13 * per.hr + 3 * (per.bb + per.hbp) - 2 * per.k) / outs_bf + fipc
    o["K%"], o["BB%"] = per.k, per.bb
    o["K-BB%"] = per.k - per.bb
    o["HR/9"] = 27 * per.hr / outs_bf
    ipgs = (ip / gs.replace(0, np.nan)).clip(3, 7).fillna(5)
    for stat in ("qs", "w", "l"):
        c = coefs[f"{stat}_per_gs"]
        per_gs = (c["const"] + c["era"] * o.ERA.clip(1, 9) + c["ipgs"] * ipgs).clip(lower=0)
        o[stat.upper()] = per_gs * gs
    # Relief decisions for non-starter innings (small for this population).
    relief_ip = (ip - gs * ipgs).clip(lower=0)
    o["W"] += relief_ip / 9 * 0.45
    o["L"] += relief_ip / 9 * 0.45
    return o
