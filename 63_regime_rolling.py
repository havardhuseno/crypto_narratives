#!/usr/bin/env python3
"""
63_regime_rolling.py
====================
STAGE 4D — does adapting the vol-forecast to REGIME (state) or to DRIFT (time) beat the pooled
static model, and does rolling VaR re-calibration fix the regime-shift over-coverage?

Two complementary adaptation axes, both tested as forecasters of OOS log-RV (QLIKE; DM vs pooled):
  A. POOLED          — single static fit on train (= script 62's M3 full feature set).        [reference]
  B. EXPANDING-WINDOW — walk forward through test, monthly refit on all data up to t (drift/recency).
  C. REGIME-HARD      — separate models per regime, hard-selected by the test regime:
        C_vol  : split by volatility regime (vol_ord buckets)   ← natural state for volatility
        C_bull : split by p_bull (HMM bull prob) median         ← the user's literal idea
  D. REGIME-INTERACTED— one regularised fit with features × p_bull AND features × vol-state (the
                        soft, data-efficient version of C — smooth regime-varying coefficients).

Feature set (all predetermined at t; train pre-2023 / test post-2023; leak-free): HAR-RV + HAR
trading-volume + Reddit volume + sentiment + 30 coin narrative subtypes + general-market narrative
(content+attention+sentiment). Standardisation fit on the relevant training subset each time.

Then VaR: for the expanding forecast (B), compare STATIC vs EXPANDING tail calibration (re-estimate
the standardised-return quantile monthly) — does rolling fix the static over-coverage from the
2021–22→2023+ volatility regime shift?

Outputs:
  outputs/regime_rolling_oos.csv   QLIKE/RMSE/DM for A–D
  outputs/regime_rolling_var.csv   VaR coverage: static vs expanding calibration
  outputs/regime_rolling_summary.md
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = PAPER_ROOT / "outputs"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


m61 = _load("ve61", "61_volforecast_economic.py")
m57 = _load("ma57", "57_market_augmented.py")
assemble = m61.assemble
qlike = m61.qlike
dm_test = m61.dm_test
christoffersen_cc = m61.christoffersen_cc
SUB30 = m61.SUB30
POLARITY = m61.POLARITY
REDDIT_VOL = m61.REDDIT_VOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=m61.COINS)
    ap.add_argument("--W", type=int, default=24)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--har-lags", nargs="+", type=int, default=[24, 72, 168])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = print
    if args.smoke:
        args.coins = ["BTC", "ETH", "SOL"]

    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import norm

    df = assemble(args.coins, args.W, args.H, args.har_lags, m61.m46.MIN_ITEMS, 3)
    harrv = df.attrs["harrv"]; harvol = df.attrs["harvol"]
    mfeat = m57.build_market_features(args.W)
    df, mcols = m57.merge_market(df, mfeat)
    df = df.sort_values(["ts"]).reset_index(drop=True)
    mkt_share = [c for c in mcols if c.startswith("market_share_")]
    cont_cols = (harrv + harvol + REDDIT_VOL + [c for c in POLARITY if c in df.columns]
                 + [f"share_{s}" for s in SUB30] + mkt_share + ["market_att", "market_net_sent"])
    tr = df["is_train"].values; te = ~tr
    log(f"n={len(df):,} (train {tr.sum():,}/test {te.sum():,}); features={len(cont_cols)}")

    y = df["y_logrv"].values
    yret = df["yret_fwd"].values
    pbull = df["ms_regime_prob_high"].values
    volstate = df["vol_ord"].values
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float).values
    Xc_raw = df[cont_cols].values
    alphas = np.logspace(-3, 4, 30)

    def design(fit_mask, interact=False):
        sc = StandardScaler().fit(Xc_raw[fit_mask])
        Xc = sc.transform(Xc_raw)
        blocks = [coin_d, Xc]
        if interact:
            pb = (pbull - 0.5)[:, None]
            vs = ((volstate - volstate[fit_mask].mean()) / (volstate[fit_mask].std() + 1e-9))[:, None]
            blocks += [Xc * pb, Xc * vs]
        return np.column_stack(blocks)

    def fit_pred(X, fit_mask, pred_mask):
        m = RidgeCV(alphas=alphas).fit(X[fit_mask], y[fit_mask])
        return m.predict(X[pred_mask])

    yte = y[te]
    preds = {}

    # A. pooled
    Xp = design(tr)
    preds["A_pooled"] = fit_pred(Xp, tr, te)

    # B. expanding-window (monthly refit on all data up to the month start)
    months = pd.PeriodIndex(df["ts"].dt.to_period("M"))
    test_months = sorted(months[te].unique())
    pred_B = np.full(te.sum(), np.nan)
    te_idx = np.where(te)[0]
    pos = {gi: k for k, gi in enumerate(te_idx)}
    for mth in test_months:
        fit_mask = (months < mth)
        blk = te & (months == mth)
        if fit_mask.sum() < 200 or blk.sum() == 0:
            continue
        Xb = design(fit_mask)
        p = fit_pred(Xb, fit_mask, blk)
        for gi, val in zip(np.where(blk)[0], p):
            pred_B[pos[gi]] = val
    preds["B_expanding"] = pred_B

    # C. regime-hard
    def regime_hard(buckets_train, buckets_all):
        out = np.full(te.sum(), np.nan)
        for b in np.unique(buckets_train[tr]):
            fit_mask = tr & (buckets_all == b)
            pm = te & (buckets_all == b)
            if fit_mask.sum() < 200 or pm.sum() == 0:
                continue
            X = design(fit_mask)
            p = fit_pred(X, fit_mask, pm)
            for gi, val in zip(np.where(pm)[0], p):
                out[pos[gi]] = val
        return out
    preds["C_vol"] = regime_hard(volstate, volstate)
    bull_bucket = (pbull > np.median(pbull[tr])).astype(int)
    preds["C_bull"] = regime_hard(bull_bucket, bull_bucket)

    # D. regime-interacted (soft)
    Xd = design(tr, interact=True)
    preds["D_interacted"] = fit_pred(Xd, tr, te)

    # score (drop NaN-aligned rows per model for fairness on common coverage)
    rows = []
    base_ql = qlike(yte, preds["A_pooled"])
    for name, p in preds.items():
        ok = np.isfinite(p)
        ql = qlike(yte[ok], p[ok])
        rmse = float(np.sqrt(np.mean((yte[ok] - p[ok]) ** 2)))
        dm_p = dm_test(qlike(yte[ok], p[ok]), base_ql[ok])[1] if name != "A_pooled" else np.nan
        rows.append({"model": name, "n_scored": int(ok.sum()), "QLIKE": float(np.mean(ql)),
                     "RMSE_logRV": rmse, "DM_vs_pooled_QLIKE_p": dm_p})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "regime_rolling_oos.csv", index=False)
    log("\nForecast variants (QLIKE; DM vs pooled A):\n" + res.to_string(index=False))

    # ---------------- VaR: static vs expanding tail calibration (on expanding forecast B) ----------------
    sig_all = {}
    # need σ̂ on train+test for B; reuse expanding preds on test and a pooled fit for train σ̂
    sig_te_B = np.exp(preds["B_expanding"])
    # static calibration: e-quantile from initial-train pooled fit
    p_tr_pooled = fit_pred(Xp, tr, tr)
    e_tr = yret[tr] / np.exp(p_tr_pooled)
    var_rows = []
    for alpha in (0.05, 0.01):
        # static
        q_static = {"normal": e_tr.mean() + e_tr.std() * norm.ppf(alpha),
                    "empirical": np.quantile(e_tr, alpha)}
        for cal, qa in q_static.items():
            ok = np.isfinite(sig_te_B)
            exc = (yret[te][ok] < qa * sig_te_B[ok]).astype(int)
            r = christoffersen_cc(exc, alpha)
            var_rows.append({"calibration": f"static_{cal}", "alpha": alpha, "rate": round(r["rate"], 4),
                             "kupiec_p": round(r["kupiec_p"], 4) if r["kupiec_p"] == r["kupiec_p"] else None,
                             "cc_p": round(r["cc_p"], 4) if r["cc_p"] == r["cc_p"] else None})
        # expanding: monthly-updated e-quantile from all data up to the month
        exc_n, exc_e = [], []
        for mth in test_months:
            fit_mask = (months < mth)
            blk = te & (months == mth)
            if fit_mask.sum() < 200 or blk.sum() == 0:
                continue
            # pooled σ̂ on history (consistent calibration sample) → standardised-return quantile
            sig_hist = np.exp(fit_pred(Xp, tr, fit_mask))
            e_hist = yret[fit_mask] / sig_hist
            qn = e_hist.mean() + e_hist.std() * norm.ppf(alpha)
            qe = np.quantile(e_hist, alpha)
            sel = np.where(blk)[0]
            for gi in sel:
                k = pos[gi]
                if not np.isfinite(sig_te_B[k]):
                    continue
                thr_n = qn * sig_te_B[k]; thr_e = qe * sig_te_B[k]
                exc_n.append(int(yret[gi] < thr_n)); exc_e.append(int(yret[gi] < thr_e))
        for cal, exc in [("expanding_normal", exc_n), ("expanding_empirical", exc_e)]:
            if not exc:
                continue
            r = christoffersen_cc(np.array(exc), alpha)
            var_rows.append({"calibration": cal, "alpha": alpha, "rate": round(r["rate"], 4),
                             "kupiec_p": round(r["kupiec_p"], 4) if r["kupiec_p"] == r["kupiec_p"] else None,
                             "cc_p": round(r["cc_p"], 4) if r["cc_p"] == r["cc_p"] else None})
    var = pd.DataFrame(var_rows); var.to_csv(OUT / "regime_rolling_var.csv", index=False)
    log("\nVaR coverage — static vs expanding tail calibration (forecast=expanding B):\n"
        + var.to_string(index=False))

    md = ["# Stage 4D — regime-conditional & rolling vol forecasting + rolling VaR", "",
          f"n={len(df):,} (test {te.sum():,}), H={args.H}h. Variants: A pooled (ref) / B expanding-window "
          "(monthly refit) / C regime-hard (by vol regime and by p_bull) / D regime-interacted "
          "(features × p_bull and × vol-state). Then VaR with static vs expanding tail calibration.", "",
          "## Forecast variants (QLIKE; DM vs pooled)", "", "```", res.to_string(index=False), "```",
          "\nLower QLIKE than pooled with DM p<0.05 ⇒ that adaptation helps. (Hard regime models can "
          "overfit on fragmented train data; the interacted/expanding versions are the data-efficient "
          "improvements.)", "",
          "## VaR coverage: static vs expanding calibration", "", "```", var.to_string(index=False), "```",
          "\nIf expanding (monthly-updated) tail calibration moves the breach rate toward α and restores "
          "Kupiec/CC adequacy where static over-covers, the regime-drift confound is confirmed and fixed."]
    (OUT / "regime_rolling_summary.md").write_text("\n".join(md))
    log(f"\n✓ oos -> {OUT/'regime_rolling_oos.csv'}")
    log(f"✓ var -> {OUT/'regime_rolling_var.csv'}")
    log(f"✓ summary -> {OUT/'regime_rolling_summary.md'}")


if __name__ == "__main__":
    main()
