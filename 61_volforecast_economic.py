#!/usr/bin/env python3
"""
61_volforecast_economic.py
==========================
STAGE 4B — harden the vol-forecasting result (script 60) and give it economic teeth.

Three upgrades over script 60:
  1. HARDENED BASELINE — the no-text benchmark now includes trailing TRADING volume
     (HAR-of-volume at 1d/3d/7d) alongside HAR-RV, so the narrative increment is measured
     beyond price persistence AND trading volume (the MDH channel) AND Reddit volume +
     sentiment. Narrative must beat all of that.
  2. CONDITIONAL QLIKE — decompose the forecast-loss improvement by realised-vol tercile of
     the test period. QLIKE rewards avoiding variance UNDER-prediction, so the hypothesis is
     the narrative gain concentrates in the HIGH-vol regime (spike anticipation).
  3. VaR BACKTEST (economic tie-in) — convert each vol forecast to a one-period Value-at-Risk
     and backtest tail coverage: Kupiec unconditional-coverage (POF) + Christoffersen
     conditional-coverage tests at α=5% and 1%, baseline vs narrative-augmented. Better VaR
     calibration = the QLIKE gain made into risk-management value.

Nested OOS models (train pre-2023 / test post-2023, standardisation frozen on train; all
predictors predetermined at t; content = 30 subtypes, noise 3.x dropped):
    M0_price    : coin FE + HAR-RV(1d/3d/7d) + HAR-trading-volume(1d/3d/7d)
    M1_+social  : + Reddit volume (log counts) + sentiment
    M2_+narr    : + 30 narrative-content shares

Outputs (under paper_narrative/, never touches production):
  outputs/volecon_oos.csv         nested OOS QLIKE/RMSE/DM (hardened baseline)
  outputs/volecon_conditional.csv QLIKE by realised-vol tercile, M0 vs M2
  outputs/volecon_var.csv         VaR exception rates + Kupiec/Christoffersen, M0 vs M2
  outputs/volecon_summary.md      write-up + verdict
"""
from __future__ import annotations

import argparse
import importlib.util
from math import erfc, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = PAPER_ROOT / "outputs"


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


m55 = _load("mv55", "55_multivariate_apath.py")
m53 = m55.m53
m46 = m53.m46
build_bars = m53.build_bars
build_coin_caches = m53.build_coin_caches
PANEL = m46.PANEL
COINS = m53.COINS
SUB30 = m55.SUB30
WINSOR = 0.005
TRAIN_END = pd.Timestamp("2023-01-01", tz="UTC")
EPS = 1e-8
POLARITY = ["net_sent", "intent_buy_share", "intent_sell_share", "intent_fomo_share", "intent_fear_share"]
REDDIT_VOL = ["att_log_posts", "att_log_items"]


def assemble(coins, W, H, har_lags, min_items, min_posts):
    panel = pd.read_parquet(PANEL, columns=["ts", "coin", "ms_regime_prob_high", "vol_regime"])
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    panel["vol_ord"] = panel["vol_regime"].cat.codes.astype(float)
    caches = {c: build_coin_caches(c) for c in coins}
    parts = []
    for coin in coins:
        d = build_bars(coin, W)
        pc = panel[panel["coin"] == coin].sort_values("ts")
        d = pd.merge_asof(d.sort_values("ts"), pc[["ts", "ms_regime_prob_high", "vol_ord"]],
                          on="ts", direction="backward")
        bars_h = d["h"].values.astype(int)
        c = caches[coin]; cum_r2 = c["cum_r2"]; cum_vol = c["cum_vol"]; close = c["close"]
        hmin = c["hmin"]; hmax = c["hmax"]; i = bars_h - hmin; span = hmax - hmin; n = len(bars_h)

        def rv(a, b):
            out = np.full(n, np.nan); ok = (a >= 0) & (b <= span + 1) & (b > a)
            out[ok] = np.sqrt(np.clip(cum_r2[b[ok]] - cum_r2[a[ok]], 0, None)); return out

        def tvol(a, b):
            out = np.full(n, np.nan); ok = (a >= 0) & (b <= span + 1) & (b > a)
            out[ok] = np.log1p(np.clip(cum_vol[b[ok]] - cum_vol[a[ok]], 0, None)); return out

        d["y_logrv"] = np.log(rv(i + 1, i + H + 1) + EPS)            # forward target
        for h in har_lags:
            d[f"harrv_{h}"] = np.log(rv(i - h + 1, i + 1) + EPS)     # trailing log-RV
            d[f"harvol_{h}"] = tvol(i - h + 1, i + 1)                # trailing log trading-volume
        d["yret_fwd"] = close.reindex(bars_h + H).values / close.reindex(bars_h).values - 1.0  # for VaR
        parts.append(d)

    df = pd.concat(parts, ignore_index=True)
    harrv = [f"harrv_{h}" for h in har_lags]; harvol = [f"harvol_{h}" for h in har_lags]
    need = ["y_logrv", "yret_fwd"] + harrv + harvol + REDDIT_VOL + ["ms_regime_prob_high", "vol_ord"]
    keep = (df["n_items"] >= min_items) & (df["n_posts"] >= min_posts)
    for col in need:
        keep &= df[col].notna() & np.isfinite(df[col])
    df = df[keep].copy()
    df = df[df.groupby("coin")["h"].transform(lambda s: (s - s.min()) % H == 0)].copy()
    for col in ["y_logrv"] + harrv + harvol:                        # winsor features+target (NOT yret_fwd: VaR needs tails)
        lo, hi = df[col].quantile([WINSOR, 1 - WINSOR]); df[col] = df[col].clip(lo, hi)
    df["is_train"] = df["ts"] < TRAIN_END
    df.attrs["harrv"] = harrv; df.attrs["harvol"] = harvol
    return df


def dm_test(la, lb, maxlags=5):
    d = np.asarray(la) - np.asarray(lb); n = len(d); dbar = d.mean(); dd = d - dbar
    var = np.mean(dd * dd)
    for L in range(1, maxlags + 1):
        var += 2 * (1 - L / (maxlags + 1)) * np.mean(dd[L:] * dd[:-L])
    dm = dbar / (sqrt(var / n) + 1e-18)
    return float(dm), float(erfc(abs(dm) / sqrt(2)))            # erfc(|z|/√2) = two-sided p


def qlike(yt_log, yp_log):
    r = np.exp(2 * yt_log) / np.exp(2 * yp_log); return r - np.log(r) - 1.0


def chi2_sf1(x):
    return erfc(sqrt(max(x, 0) / 2))                            # P(chi2_1 > x)


def kupiec_pof(x, n, alpha):
    if x == 0 or x == n:
        return float("nan"), float("nan")
    pi = x / n
    ll0 = (n - x) * log(1 - alpha) + x * log(alpha)
    ll1 = (n - x) * log(1 - pi) + x * log(pi)
    lr = -2 * (ll0 - ll1)
    return float(lr), float(chi2_sf1(lr))


def christoffersen_cc(exc, alpha):
    """Conditional coverage = Kupiec POF + independence (exception clustering). exc = 0/1 array."""
    exc = np.asarray(exc).astype(int); n = len(exc); x = int(exc.sum())
    lr_uc, p_uc = kupiec_pof(x, n, alpha)
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(exc[:-1], exc[1:]):
        if a == 0 and b == 0: n00 += 1
        elif a == 0 and b == 1: n01 += 1
        elif a == 1 and b == 0: n10 += 1
        else: n11 += 1
    lr_ind = float("nan")
    try:
        pi01 = n01 / (n00 + n01); pi11 = n11 / (n10 + n11); pi = (n01 + n11) / (n00 + n01 + n10 + n11)
        if 0 < pi < 1 and (n00 + n01) and (n10 + n11):
            ll_ind0 = (n00 + n10) * log(1 - pi) + (n01 + n11) * log(pi)
            ll_ind1 = (n00 * log(1 - pi01) if pi01 < 1 else 0) + (n01 * log(pi01) if pi01 > 0 else 0) \
                + (n10 * log(1 - pi11) if pi11 < 1 else 0) + (n11 * log(pi11) if pi11 > 0 else 0)
            lr_ind = -2 * (ll_ind0 - ll_ind1)
    except Exception:
        pass
    lr_cc = (lr_uc + lr_ind) if (lr_uc == lr_uc and lr_ind == lr_ind) else float("nan")
    p_cc = float(np.exp(-lr_cc / 2)) if lr_cc == lr_cc else float("nan")   # chi2(2) survival = exp(-x/2)
    return {"n": n, "exceptions": x, "rate": x / n, "alpha": alpha,
            "kupiec_LR": lr_uc, "kupiec_p": p_uc, "cc_LR": lr_cc, "cc_p": p_cc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=COINS)
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

    log(f"vol-econ: target=log-RV[t,t+{args.H}h); HAR(rv+tradingvol) lags={args.har_lags}; W={args.W}h")
    df = assemble(args.coins, args.W, args.H, args.har_lags, m46.MIN_ITEMS, 3)
    harrv = df.attrs["harrv"]; harvol = df.attrs["harvol"]
    tr = df["is_train"].values; te = ~tr
    log(f"  n={len(df):,} (train {tr.sum():,}/test {te.sum():,})")
    y = df["y_logrv"].values
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float).values

    def sc(cols):
        raw = df[cols].values; return StandardScaler().fit(raw[tr]).transform(raw)
    PRICE = sc(harrv + harvol)
    SOCIAL = sc(REDDIT_VOL + [c for c in POLARITY if c in df.columns])
    CONTENT = sc([f"share_{s}" for s in SUB30])
    designs = {
        "M0_price": np.column_stack([coin_d, PRICE]),
        "M1_+social": np.column_stack([coin_d, PRICE, SOCIAL]),
        "M2_+narrative": np.column_stack([coin_d, PRICE, SOCIAL, CONTENT]),
    }
    alphas = np.logspace(-3, 4, 30)
    preds, ql = {}, {}
    rows = []
    yte = y[te]
    for name, X in designs.items():
        mdl = RidgeCV(alphas=alphas).fit(X[tr], y[tr]); p = mdl.predict(X[te]); preds[name] = p
        ql[name] = qlike(yte, p)
        rows.append({"model": name, "RMSE_logRV": float(np.sqrt(np.mean((yte - p) ** 2))),
                     "QLIKE": float(np.mean(ql[name]))})
    res = pd.DataFrame(rows)
    sse0 = np.sum((yte - preds["M0_price"]) ** 2)
    res["OOS_R2_vs_M0"] = [1 - np.sum((yte - preds[m]) ** 2) / sse0 for m in res["model"]]
    dm20 = dm_test(ql["M2_+narrative"], ql["M0_price"])
    dm21 = dm_test(ql["M2_+narrative"], ql["M1_+social"])
    res.to_csv(OUT / "volecon_oos.csv", index=False)
    log("\nHardened-baseline OOS:\n" + res.to_string(index=False))
    log(f"  narrative vs price-baseline  DM(QLIKE) stat={dm20[0]:+.2f} p={dm20[1]:.3g}")
    log(f"  narrative beyond +social     DM(QLIKE) stat={dm21[0]:+.2f} p={dm21[1]:.3g}")

    # conditional QLIKE by realised-vol tercile (of realised forward RV on test)
    rv_real = yte
    q1, q2 = np.quantile(rv_real, [1 / 3, 2 / 3])
    terc = np.where(rv_real <= q1, "low", np.where(rv_real <= q2, "mid", "high"))
    crows = []
    for t in ["low", "mid", "high"]:
        m = terc == t
        crows.append({"vol_tercile": t, "n": int(m.sum()),
                      "QLIKE_M0": float(ql["M0_price"][m].mean()),
                      "QLIKE_M2": float(ql["M2_+narrative"][m].mean()),
                      "QLIKE_reduction_%": float(100 * (ql["M0_price"][m].mean() - ql["M2_+narrative"][m].mean())
                                                 / ql["M0_price"][m].mean())})
    cond = pd.DataFrame(crows); cond.to_csv(OUT / "volecon_conditional.csv", index=False)
    log("\nConditional QLIKE by realised-vol tercile:\n" + cond.to_string(index=False))

    # VaR backtest: sigma forecast = exp(pred_logrv); VaR_alpha = z * sigma; exception if yret < -VaR
    from math import sqrt as _sqrt
    yret = df["yret_fwd"].values[te]
    z = {0.05: 1.6448536, 0.01: 2.3263479}
    vrows = []
    for alpha, za in z.items():
        for name in ["M0_price", "M2_+narrative"]:
            sigma = np.exp(preds[name])
            exc = (yret < -za * sigma).astype(int)
            r = christoffersen_cc(exc, alpha)
            r["model"] = name; r["alpha"] = alpha; vrows.append(r)
    var = pd.DataFrame(vrows)[["model", "alpha", "n", "exceptions", "rate",
                               "kupiec_LR", "kupiec_p", "cc_LR", "cc_p"]]
    var.to_csv(OUT / "volecon_var.csv", index=False)
    log("\nVaR backtest (exception rate should ≈ alpha; Kupiec/CC p>0.05 = adequate coverage):\n"
        + var.to_string(index=False))

    md = ["# Stage 4B — vol-forecast: hardened baseline, conditional QLIKE, VaR economic value", "",
          f"Target log-RV[t,t+{args.H}h). Baseline M0 = HAR-RV + HAR-trading-volume ({args.har_lags}h) "
          "+ coin FE; M1 +Reddit volume+sentiment; M2 +30 narrative subtypes. Train pre-2023/test "
          f"post-2023, frozen standardisation, leak-free. n_test={te.sum():,}.", "",
          "## OOS (narrative increment beyond price + trading-volume + social)", "",
          "```", res.to_string(index=False), "```",
          f"\nnarrative vs price baseline: DM(QLIKE) p={dm20[1]:.3g}; narrative beyond social: "
          f"DM(QLIKE) p={dm21[1]:.3g}.", "",
          "## Conditional QLIKE by realised-vol tercile", "", "```", cond.to_string(index=False), "```",
          "\nIf the QLIKE reduction concentrates in the HIGH tercile, narrative's value is "
          "spike/variance-underprediction avoidance — the risk-relevant dimension.", "",
          "## VaR backtest (economic value)", "", "```", var.to_string(index=False), "```",
          "\nException rate ≈ α and Kupiec/Christoffersen p>0.05 ⇒ adequate coverage. If the price "
          "baseline under-covers (rate>α, rejected) and the narrative model is closer/adequate, the "
          "QLIKE gain becomes concrete risk-management value (assumes zero-mean Normal returns scaled "
          "by the vol forecast — a standard first pass; Student-t / empirical refinement is a follow-up)."]
    (OUT / "volecon_summary.md").write_text("\n".join(md))
    log(f"\n✓ oos         -> {OUT/'volecon_oos.csv'}")
    log(f"✓ conditional -> {OUT/'volecon_conditional.csv'}")
    log(f"✓ var         -> {OUT/'volecon_var.csv'}")
    log(f"✓ summary     -> {OUT/'volecon_summary.md'}")


if __name__ == "__main__":
    main()
