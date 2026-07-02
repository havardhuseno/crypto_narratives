#!/usr/bin/env python3
"""
68_regime_conditional.py  (v2)
==============================
Are content's effects on RETURN, VOLATILITY, and VaR regime-dependent? Four PRE-SPECIFIED cells
(no fishing): {bull, bear} × {low-vol, high-vol}, from HMM bull-probability and the volatility
regime, both predetermined at t (split at TRAIN medians, frozen). Models are trained POOLED on
pre-2023; we evaluate the content increment *within each test-period cell*, out-of-sample, with
block-bootstrap significance.

  A. return:     OOS dir-IC [controls] vs [+content] within each cell + bootstrap CI of the diff.
  B. volatility: OOS QLIKE [HAR+vol baseline] vs [+narrative] within each cell (reduction %).
  C. VaR:        5% exception rate + Kupiec p (empirical calibration) within each cell, baseline vs
                 narrative-augmented vol forecast.

Outputs: outputs/regime_conditional_{return,vol,var}.csv , regime_conditional_summary.md
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

PAPER_ROOT = Path(__file__).resolve().parents[1]; HERE = Path(__file__).resolve().parent
OUT = PAPER_ROOT / "outputs"
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m58 = _load("d58", "58_conditional_directional.py")
m61 = _load("e61", "61_volforecast_economic.py")
assemble_ret = m58.m53.assemble
SUB30 = m58.SUB30; zsc = m58.zsc; POL = m58.POLARITY; ATTN = m58.ATTENTION
qlike = m61.qlike; christ = m61.christoffersen_cc; REDDIT_VOL = m61.REDDIT_VOL; POLARITY = m61.POLARITY

CELLS = ["bull/hi", "bull/lo", "bear/hi", "bear/lo"]
def cell_labels(df, tr):
    pb = df["ms_regime_prob_high"].values; vo = df["vol_ord"].values
    pb_thr = np.median(pb[tr]); vo_thr = np.median(vo[tr])
    bull = pb > pb_thr; hivol = vo > vo_thr
    return np.array([("bull" if b else "bear") + "/" + ("hi" if h else "lo") for b, h in zip(bull, hivol)])

def spearman(a, b): return float(pd.Series(a).corr(pd.Series(b), method="spearman"))

def main():
    import argparse
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import norm
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=m58.COINS)
    ap.add_argument("--W", type=int, default=24); ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--H", type=int, default=24); ap.add_argument("--har-lags", nargs="+", type=int, default=[24,72,168])
    ap.add_argument("--n-boot", type=int, default=500); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); log = print
    alphas = np.logspace(-2, 4, 25)

    # ---------- A. RETURN ----------
    dfr = assemble_ret(args.coins, args.W, args.m, args.H, m58.m53.m46.MIN_ITEMS, 3, 168)
    tr = dfr["is_train"].values; te = ~tr; y = dfr["Yret"].values
    coin_d = pd.get_dummies(dfr["coin"], prefix="coin", drop_first=True).astype(float).values
    def scr(cols): raw = dfr[cols].values; return StandardScaler().fit(raw[tr]).transform(raw)
    ctrl = scr(["trail_ret", "ms_regime_prob_high", "vol_ord"]); content = scr([f"share_{s}" for s in SUB30])
    Xc = np.column_stack([coin_d, ctrl]); Xk = np.column_stack([coin_d, ctrl, content])
    pc = RidgeCV(alphas=alphas).fit(Xc[tr], y[tr]).predict(Xc[te])
    pk = RidgeCV(alphas=alphas).fit(Xk[tr], y[tr]).predict(Xk[te])
    yte = y[te]; cells_te = cell_labels(dfr, tr)[te]; hte = dfr["h"].values[te]
    rng = np.random.default_rng(args.seed); rrows = []
    for cell in CELLS:
        m = cells_te == cell
        if m.sum() < 80: continue
        icc, ick = spearman(pc[m], yte[m]), spearman(pk[m], yte[m])
        # block bootstrap of the diff within cell
        sub_h = hte[m]; cl = {}
        for i, hh in enumerate(sub_h): cl.setdefault(hh, []).append(i)
        keys = list(cl.values()); pcm, pkm, ym = pc[m], pk[m], yte[m]; d = []
        for _ in range(args.n_boot):
            idx = np.concatenate([keys[j] for j in rng.integers(0, len(keys), len(keys))])
            d.append(spearman(pkm[idx], ym[idx]) - spearman(pcm[idx], ym[idx]))
        d = np.array(d)
        rrows.append({"cell": cell, "n": int(m.sum()), "IC_controls": icc, "IC_content": ick,
                      "diff": ick - icc, "boot_lo": float(np.quantile(d,.025)),
                      "boot_hi": float(np.quantile(d,.975)), "P_diff>0": float((d>0).mean()),
                      "hit_content": float(((pkm>0)==(ym>0)).mean())})
    rdf = pd.DataFrame(rrows); rdf.to_csv(OUT / "regime_conditional_return.csv", index=False)
    log("\n=== A. RETURN: OOS dir-IC by regime cell ===\n" + rdf.to_string(index=False))

    # ---------- B+C. VOLATILITY + VaR ----------
    dfv = m61.assemble(args.coins, args.W, args.H, args.har_lags, m61.m46.MIN_ITEMS, 3)
    harrv = dfv.attrs["harrv"]; harvol = dfv.attrs["harvol"]
    trv = dfv["is_train"].values; tev = ~trv
    ylr = dfv["y_logrv"].values; yret = dfv["yret_fwd"].values
    coind = pd.get_dummies(dfv["coin"], prefix="coin", drop_first=True).astype(float).values
    def scv(cols): raw = dfv[cols].values; return StandardScaler().fit(raw[trv]).transform(raw)
    PRICE = scv(harrv + harvol); SOC = scv(REDDIT_VOL + [c for c in POLARITY if c in dfv.columns])
    CON = scv([f"share_{s}" for s in SUB30])
    M0 = np.column_stack([coind, PRICE]); M2 = np.column_stack([coind, PRICE, SOC, CON])
    p0 = RidgeCV(alphas=alphas).fit(M0[trv], ylr[trv]); p2 = RidgeCV(alphas=alphas).fit(M2[trv], ylr[trv])
    pr0_tr, pr0 = p0.predict(M0[trv]), p0.predict(M0[tev]); pr2 = p2.predict(M2[tev])
    ql0 = qlike(ylr[tev], pr0); ql2 = qlike(ylr[tev], pr2)
    cells_v = cell_labels(dfv, trv)[tev]
    # empirical VaR calibration (train standardized returns vs baseline sigma)
    e_tr = yret[trv] / np.exp(pr0_tr); q05 = np.quantile(e_tr, 0.05)
    sig0 = np.exp(pr0); sig2 = np.exp(pr2); yv = yret[tev]
    vrows = []; brows = []
    for cell in CELLS:
        m = cells_v == cell
        if m.sum() < 80: continue
        red = 100*(ql0[m].mean() - ql2[m].mean())/ql0[m].mean()
        brows.append({"cell": cell, "n": int(m.sum()), "QLIKE_baseline": float(ql0[m].mean()),
                      "QLIKE_narrative": float(ql2[m].mean()), "QLIKE_reduction_%": float(red)})
        for name, sig in [("baseline", sig0), ("narrative", sig2)]:
            exc = (yv[m] < q05*sig[m]).astype(int); r = christ(exc, 0.05)
            vrows.append({"cell": cell, "model": name, "n": int(m.sum()), "exc_rate": round(r["rate"],4),
                          "kupiec_p": round(r["kupiec_p"],4) if r["kupiec_p"]==r["kupiec_p"] else None})
    bdf = pd.DataFrame(brows); vdf = pd.DataFrame(vrows)
    bdf.to_csv(OUT / "regime_conditional_vol.csv", index=False); vdf.to_csv(OUT / "regime_conditional_var.csv", index=False)
    log("\n=== B. VOLATILITY: QLIKE reduction (narrative vs baseline) by cell ===\n" + bdf.to_string(index=False))
    log("\n=== C. VaR 5% coverage by cell (rate~0.05 & kupiec p>0.05 = adequate) ===\n" + vdf.to_string(index=False))

    md = ["# Regime-conditional effects (v2): {bull,bear}×{low,high}-vol, OOS", "",
          "Pre-specified 4 cells; models trained pooled on pre-2023, evaluated within each test cell.", "",
          "## A. Return (OOS dir-IC, content vs controls)", "", "```", rdf.to_string(index=False), "```",
          "\n## B. Volatility (QLIKE reduction from narrative)", "", "```", bdf.to_string(index=False), "```",
          "\n## C. VaR 5% coverage (baseline vs narrative)", "", "```", vdf.to_string(index=False), "```",
          "\n**Read:** return content adds OOS skill in a cell only if diff>0 with P(diff>0) high and CI "
          "excluding 0; the volatility gain is expected to concentrate in high-vol cells; narrative VaR is "
          "valuable if it brings the exception rate toward 5% (Kupiec p>0.05) where the baseline fails — "
          "especially in high-vol cells."]
    (OUT / "regime_conditional_summary.md").write_text("\n".join(md))
    log(f"\n✓ -> regime_conditional_{{return,vol,var}}.csv , _summary.md")

if __name__ == "__main__":
    main()
