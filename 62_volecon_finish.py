#!/usr/bin/env python3
"""
62_volecon_finish.py
====================
STAGE 4C — finish the economic layer of the narrative-volatility result.

Three additions over script 61:
  1. MARKET-AUGMENTED model variant — add the general-market Reddit stream (r/CryptoCurrency +
     r/CryptoMarkets; script 56) as shared content + attention + sentiment, so we answer "does
     MARKET-WIDE narrative improve coin volatility forecasting beyond the coin's own?".
  2. FAT-TAIL VaR — recompute Value-at-Risk with three calibrations of the standardised-return
     distribution (estimated on TRAIN): Normal, Student-t (scipy MLE), Empirical (train quantile).
     The 1% Normal failure in script 61 is a distribution problem; empirical/t calibration fixes
     the shape, and on top of a correct shape the better vol forecast should give better coverage.
  3. VOL-TIMING ECONOMIC VALUE — a volatility-targeting strategy (position ∝ target/σ̂_forecast);
     mean-variance utility gain ΔU of using the narrative/market forecast vs the price baseline
     (Fleming–Kirby–Ostdiek style "fee an investor would pay").

Nested OOS models on a common sample (train pre-2023 / test post-2023, frozen standardisation):
    M0_price   : coin FE + HAR-RV + HAR-trading-volume
    M1_social  : + Reddit volume + sentiment (coin)
    M2_coinnarr: + 30 coin narrative subtypes
    M3_market  : + general-market narrative content + attention + sentiment

Outputs:
  outputs/volecon2_oos.csv    nested OOS QLIKE/RMSE/DM incl. market variant
  outputs/volecon2_var.csv    VaR coverage (Normal/Student-t/Empirical) × models × α
  outputs/volecon2_timing.csv vol-timing utility/Sharpe per model + ΔU vs baseline
  outputs/volecon2_summary.md write-up
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
build_market_features = m57.build_market_features
merge_market = m57.merge_market


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=m61.COINS)
    ap.add_argument("--W", type=int, default=24)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--har-lags", nargs="+", type=int, default=[24, 72, 168])
    ap.add_argument("--gamma", type=float, default=2.0, help="risk aversion for MV utility")
    ap.add_argument("--maxlev", type=float, default=3.0, help="max leverage for vol-timing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = print
    if args.smoke:
        args.coins = ["BTC", "ETH", "SOL"]

    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import norm, t as student_t

    log(f"vol-econ-finish: target log-RV[t,t+{args.H}h); coins={args.coins}")
    df = assemble(args.coins, args.W, args.H, args.har_lags, m61.m46.MIN_ITEMS, 3)
    harrv = df.attrs["harrv"]; harvol = df.attrs["harvol"]
    # merge general-market features (content shares + attention + sentiment), common sample
    mfeat = build_market_features(args.W)
    df, mcols = merge_market(df, mfeat)
    mkt_share = [c for c in mcols if c.startswith("market_share_")]
    tr = df["is_train"].values; te = ~tr
    log(f"  common-sample n={len(df):,} (train {tr.sum():,}/test {te.sum():,}); market subtypes={len(mkt_share)}")

    y = df["y_logrv"].values
    yret = df["yret_fwd"].values
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float).values

    def sc(cols):
        raw = df[cols].values; return StandardScaler().fit(raw[tr]).transform(raw)
    PRICE = sc(harrv + harvol)
    SOCIAL = sc(REDDIT_VOL + [c for c in POLARITY if c in df.columns])
    COINNARR = sc([f"share_{s}" for s in SUB30])
    MARKET = sc(mkt_share + ["market_att", "market_net_sent"])

    designs = {
        "M0_price": np.column_stack([coin_d, PRICE]),
        "M1_social": np.column_stack([coin_d, PRICE, SOCIAL]),
        "M2_coinnarr": np.column_stack([coin_d, PRICE, SOCIAL, COINNARR]),
        "M3_market": np.column_stack([coin_d, PRICE, SOCIAL, COINNARR, MARKET]),
    }
    alphas = np.logspace(-3, 4, 30)
    pred_tr, pred_te, ql = {}, {}, {}
    rows = []
    yte = y[te]
    for name, X in designs.items():
        mdl = RidgeCV(alphas=alphas).fit(X[tr], y[tr])
        pred_tr[name] = mdl.predict(X[tr]); pred_te[name] = mdl.predict(X[te])
        ql[name] = qlike(yte, pred_te[name])
        rows.append({"model": name, "RMSE_logRV": float(np.sqrt(np.mean((yte - pred_te[name]) ** 2))),
                     "QLIKE": float(np.mean(ql[name]))})
    res = pd.DataFrame(rows)
    sse0 = np.sum((yte - pred_te["M0_price"]) ** 2)
    res["OOS_R2_vs_M0"] = [1 - np.sum((yte - pred_te[m]) ** 2) / sse0 for m in res["model"]]
    res.to_csv(OUT / "volecon2_oos.csv", index=False)
    log("\nOOS (incl. market-augmented M3):\n" + res.to_string(index=False))
    log(f"  coin-narr vs price : DM(QLIKE) p={dm_test(ql['M2_coinnarr'], ql['M0_price'])[1]:.3g}")
    log(f"  market beyond coin : DM(QLIKE) p={dm_test(ql['M3_market'], ql['M2_coinnarr'])[1]:.3g}")

    # ---------------- VaR with Normal / Student-t / Empirical calibration ----------------
    # standardised train returns e = yret/σ̂ (σ̂=exp(pred log-RV)); calibrate lower-tail quantile on TRAIN.
    var_rows = []
    for name in ["M0_price", "M2_coinnarr", "M3_market"]:
        sig_tr = np.exp(pred_tr[name]); sig_te = np.exp(pred_te[name])
        e_tr = yret[tr] / sig_tr
        tdf, tloc, tscale = student_t.fit(e_tr)
        for alpha in (0.05, 0.01):
            q = {"normal": e_tr.mean() + e_tr.std() * norm.ppf(alpha),
                 "studentt": student_t.ppf(alpha, tdf, tloc, tscale),
                 "empirical": np.quantile(e_tr, alpha)}
            for cal, qa in q.items():
                thr = qa * sig_te                      # VaR threshold (lower tail, negative)
                exc = (yret[te] < thr).astype(int)
                r = christoffersen_cc(exc, alpha)
                var_rows.append({"model": name, "calibration": cal, "alpha": alpha,
                                 "exceptions": r["exceptions"], "rate": round(r["rate"], 4),
                                 "kupiec_p": round(r["kupiec_p"], 4) if r["kupiec_p"] == r["kupiec_p"] else None,
                                 "cc_p": round(r["cc_p"], 4) if r["cc_p"] == r["cc_p"] else None})
    var = pd.DataFrame(var_rows); var.to_csv(OUT / "volecon2_var.csv", index=False)
    log("\nVaR coverage (rate≈alpha & p>0.05 = adequate):\n" + var.to_string(index=False))

    # ---------------- vol-timing economic value (mean-variance utility) ----------------
    target = float(np.median(np.exp(pred_te["M0_price"])))   # neutral risk target
    tim_rows = []
    for name in ["M0_price", "M2_coinnarr", "M3_market"]:
        sig = np.exp(pred_te[name])
        w = np.clip(target / sig, 0, args.maxlev)
        pnl = w * yret[te]
        mu, sd = pnl.mean(), pnl.std()
        sharpe = mu / (sd + 1e-12)
        util = mu - 0.5 * args.gamma * (sd ** 2)
        tim_rows.append({"model": name, "mean": mu, "vol": sd, "sharpe": sharpe, "MV_utility": util})
    tim = pd.DataFrame(tim_rows)
    u0 = tim.loc[tim.model == "M0_price", "MV_utility"].values[0]
    tim["dU_vs_M0"] = tim["MV_utility"] - u0
    tim.to_csv(OUT / "volecon2_timing.csv", index=False)
    log("\nVol-timing economic value (ΔU = utility gain vs price baseline):\n" + tim.to_string(index=False))

    md = ["# Stage 4C — economic-layer finish: market-augmented forecast, fat-tail VaR, vol-timing", "",
          f"Common sample n={len(df):,} (test {te.sum():,}), H={args.H}h. Models M0 price → M1 +social → "
          "M2 +coin-narrative → M3 +market-narrative (general subreddits). VaR calibrated (Normal / "
          "Student-t / Empirical) on train standardised returns. Vol-timing: position∝target/σ̂, "
          f"MV utility (γ={args.gamma}, max leverage {args.maxlev}).", "",
          "## OOS forecast (incl. market augmentation)", "", "```", res.to_string(index=False), "```",
          f"\ncoin-narrative vs price DM(QLIKE) p={dm_test(ql['M2_coinnarr'], ql['M0_price'])[1]:.3g}; "
          f"market beyond coin DM(QLIKE) p={dm_test(ql['M3_market'], ql['M2_coinnarr'])[1]:.3g}.", "",
          "## VaR coverage (Normal vs Student-t vs Empirical)", "", "```", var.to_string(index=False), "```",
          "\nEmpirical/Student-t calibration corrects the 1% Normal under-coverage; the question is "
          "whether the narrative/market forecast then delivers coverage closer to α and CC adequacy.", "",
          "## Vol-timing economic value", "", "```", tim.to_string(index=False), "```",
          "\nΔU>0 = a mean-variance investor pays to use the narrative/market vol forecast over the price "
          "baseline. (Stylised single-name vol-targeting pooled across coins; illustrative of economic value.)"]
    (OUT / "volecon2_summary.md").write_text("\n".join(md))
    log(f"\n✓ oos    -> {OUT/'volecon2_oos.csv'}")
    log(f"✓ var    -> {OUT/'volecon2_var.csv'}")
    log(f"✓ timing -> {OUT/'volecon2_timing.csv'}")
    log(f"✓ summary-> {OUT/'volecon2_summary.md'}")


if __name__ == "__main__":
    main()
