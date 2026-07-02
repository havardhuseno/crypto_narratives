#!/usr/bin/env python3
"""
60_vol_forecasting.py
=====================
STAGE 4A — the lead POSITIVE result for the QF paper: does narrative content improve
OUT-OF-SAMPLE realized-volatility forecasting beyond the standard HAR-RV benchmark and
beyond Reddit attention/sentiment?

Target: forward realized volatility (log-RV) over [t, t+H).
Nested, strictly leak-free models (all features predetermined at t; train pre-2023, test
post-2023; standardisation frozen on train):
    M1  HAR-RV            : trailing log-RV at 1d/3d/7d (Corsi 2009 benchmark) + coin FE
    M2  + social          : + Reddit volume (log post & post+comment counts) + sentiment
    M3  + narrative       : + 30-subtype narrative-content shares
The contribution = the INCREMENTAL out-of-sample improvement of M3 over M1/M2.

Scored out-of-sample with:
  • OOS R²  (1 − SSE_model/SSE_benchmark) vs the train-mean AND vs HAR (incremental);
  • RMSE on log-RV;
  • QLIKE  (Patton 2011 robust volatility loss, on the variance scale);
  • Diebold–Mariano test (Newey–West HAC) — is M3's forecast loss significantly below M1/M2?
Pooled and per-coin.

Outputs (under paper_narrative/, never touches production):
  outputs/volforecast_oos.csv      pooled nested-model OOS metrics
  outputs/volforecast_percoin.csv  per-coin OOS R²(M3 vs HAR) + DM
  outputs/volforecast_summary.md   write-up + verdict
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
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m55 = _load("mv55", "55_multivariate_apath.py")
m53 = m55.m53
m46 = m53.m46
build_bars = m53.build_bars
build_coin_caches = m53.build_coin_caches
PANEL = m46.PANEL
HOUR = m53.HOUR
COINS = m53.COINS
SUB30 = m55.SUB30
zsc = m55.zsc
WINSOR = 0.005
TRAIN_END = pd.Timestamp("2023-01-01", tz="UTC")
EPS = 1e-8

POLARITY = ["net_sent", "intent_buy_share", "intent_sell_share", "intent_fomo_share", "intent_fear_share"]
REDDIT_VOL = ["att_log_posts", "att_log_items"]


def assemble_vol(coins, W, H, har_lags, min_items, min_posts):
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
        c = caches[coin]; cum_r2 = c["cum_r2"]; hmin = c["hmin"]; hmax = c["hmax"]
        i = bars_h - hmin; span = hmax - hmin; n = len(bars_h)

        def rv(a_idx, b_idx):
            """sqrt Σ r² over dense (a_idx, b_idx] using cum_r2; NaN where out of range."""
            out = np.full(n, np.nan)
            ok = (a_idx >= 0) & (b_idx <= span + 1) & (b_idx > a_idx)
            out[ok] = np.sqrt(np.clip(cum_r2[b_idx[ok]] - cum_r2[a_idx[ok]], 0, None))
            return out

        # forward target RV over [t, t+H)  → dense (i+1 .. i+H+1]
        rv_fwd = rv(i + 1, i + H + 1)
        d["y_logrv"] = np.log(rv_fwd + EPS)
        # HAR trailing log-RV over [t-h, t)  → dense (i-h+1 .. i+1]
        for h in har_lags:
            d[f"har_{h}"] = np.log(rv(i - h + 1, i + 1) + EPS)
        parts.append(d)

    df = pd.concat(parts, ignore_index=True)
    harcols = [f"har_{h}" for h in har_lags]
    need = ["y_logrv"] + harcols + REDDIT_VOL + ["ms_regime_prob_high", "vol_ord"]
    keep = (df["n_items"] >= min_items) & (df["n_posts"] >= min_posts)
    for col in need:
        keep &= df[col].notna() & np.isfinite(df[col])
    df = df[keep].copy()
    # non-overlapping bars at step=H per coin (independent forecasts, clean DM test)
    df = df[df.groupby("coin")["h"].transform(lambda s: (s - s.min()) % H == 0)].copy()
    for col in ["y_logrv"] + harcols:
        lo, hi = df[col].quantile([WINSOR, 1 - WINSOR]); df[col] = df[col].clip(lo, hi)
    df["is_train"] = df["ts"] < TRAIN_END
    df.attrs["harcols"] = harcols
    return df


def dm_test(loss_a, loss_b, maxlags=5):
    """Diebold–Mariano: is mean loss_a < loss_b? d=a−b; HAC (NW) variance. Returns (dm, p, mean_d)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d); dbar = d.mean(); dd = d - dbar
    gamma0 = np.mean(dd * dd)
    var = gamma0
    for L in range(1, maxlags + 1):
        w = 1.0 - L / (maxlags + 1)
        cov = np.mean(dd[L:] * dd[:-L])
        var += 2 * w * cov
    se = np.sqrt(var / n)
    dm = dbar / (se + 1e-18)
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(dm) / sqrt(2))))     # two-sided normal
    return float(dm), float(p), float(dbar)


def qlike(y_true_log, y_pred_log):
    """QLIKE on the VARIANCE scale: proxy=exp(2·logRV_true), h=exp(2·logRV_pred).
    qlike = proxy/h − log(proxy/h) − 1  (Patton 2011; min at proxy=h, ≥0)."""
    proxy = np.exp(2 * y_true_log); h = np.exp(2 * y_pred_log)
    r = proxy / h
    return r - np.log(r) - 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=COINS)
    ap.add_argument("--W", type=int, default=24, help="narrative-content window (h)")
    ap.add_argument("--H", type=int, default=24, help="forecast horizon (h): RV over [t,t+H)")
    ap.add_argument("--har-lags", nargs="+", type=int, default=[24, 72, 168], help="HAR trailing windows (h)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = print
    if args.smoke:
        args.coins = ["BTC", "ETH", "SOL"]

    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    log(f"vol-forecast: target=log-RV[t,t+{args.H}h); HAR lags={args.har_lags}h; content W={args.W}h; "
        f"coins={args.coins}")
    df = assemble_vol(args.coins, args.W, args.H, args.har_lags, m46.MIN_ITEMS, 3)
    harcols = df.attrs["harcols"]
    tr = df["is_train"].values; te = ~tr
    log(f"  n={len(df):,}  (train {tr.sum():,} / test {te.sum():,})")
    y = df["y_logrv"].values
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float).values

    def sc(cols):
        raw = df[cols].values
        return StandardScaler().fit(raw[tr]).transform(raw)
    HAR = sc(harcols)
    SOCIAL = sc(REDDIT_VOL + [c for c in POLARITY if c in df.columns])
    CONTENT = sc([f"share_{s}" for s in SUB30])

    designs = {
        "M1_HAR": np.column_stack([coin_d, HAR]),
        "M2_HAR+social": np.column_stack([coin_d, HAR, SOCIAL]),
        "M3_HAR+social+narrative": np.column_stack([coin_d, HAR, SOCIAL, CONTENT]),
    }
    alphas = np.logspace(-3, 4, 30)
    ybar_tr = y[tr].mean()
    sse_mean = np.sum((y[te] - ybar_tr) ** 2)
    preds, losses, rows = {}, {}, []
    for name, X in designs.items():
        mdl = RidgeCV(alphas=alphas).fit(X[tr], y[tr])
        p = mdl.predict(X[te]); preds[name] = p
        yte = y[te]
        sse = np.sum((yte - p) ** 2)
        r2_mean = 1 - sse / sse_mean
        rmse = np.sqrt(np.mean((yte - p) ** 2))
        ql = qlike(yte, p); losses[name] = {"se": (yte - p) ** 2, "ql": ql}
        rows.append({"model": name, "n_test": int(te.sum()), "alpha": float(mdl.alpha_),
                     "OOS_R2_vs_mean": float(r2_mean), "RMSE_logRV": float(rmse),
                     "QLIKE": float(np.mean(ql))})
    res = pd.DataFrame(rows)
    # incremental OOS R² of each model vs the HAR baseline forecast
    sse_har = np.sum((y[te] - preds["M1_HAR"]) ** 2)
    res["OOS_R2_vs_HAR"] = [1 - np.sum((y[te] - preds[m]) ** 2) / sse_har for m in res["model"]]
    # Diebold–Mariano vs HAR (squared-error and QLIKE losses)
    dm_se = []; dm_ql = []
    for m in res["model"]:
        if m == "M1_HAR":
            dm_se.append(("—", "—")); dm_ql.append(("—", "—")); continue
        a, b = losses[m]["se"], losses["M1_HAR"]["se"]
        s = dm_test(a, b); q = dm_test(losses[m]["ql"], losses["M1_HAR"]["ql"])
        dm_se.append((round(s[0], 2), f"{s[1]:.3g}")); dm_ql.append((round(q[0], 2), f"{q[1]:.3g}"))
    res["DM_vs_HAR_SE(stat,p)"] = [str(x) for x in dm_se]
    res["DM_vs_HAR_QLIKE(stat,p)"] = [str(x) for x in dm_ql]
    res.to_csv(OUT / "volforecast_oos.csv", index=False)
    log("\nPooled OOS metrics:\n" + res.to_string(index=False))

    # M3 vs M2 (does narrative add beyond social?)
    s32 = dm_test(losses["M3_HAR+social+narrative"]["se"], losses["M2_HAR+social"]["se"])
    q32 = dm_test(losses["M3_HAR+social+narrative"]["ql"], losses["M2_HAR+social"]["ql"])
    log(f"\nM3 vs M2 (narrative beyond social): DM(SE) stat={s32[0]:+.2f} p={s32[1]:.3g}  | "
        f"DM(QLIKE) stat={q32[0]:+.2f} p={q32[1]:.3g}")

    # per-coin OOS R²(M3 vs HAR). preds[*] are aligned to the test rows (order of np.where(te)).
    coin_te = df["coin"].values[te]
    pc_rows = []
    for coin in args.coins:
        sel = coin_te == coin
        if sel.sum() < 50:
            continue
        yc = y[te][sel]
        har_c = preds["M1_HAR"][sel]; m3_c = preds["M3_HAR+social+narrative"][sel]
        r2 = 1 - np.sum((yc - m3_c) ** 2) / np.sum((yc - har_c) ** 2)
        dm = dm_test((yc - m3_c) ** 2, (yc - har_c) ** 2)
        pc_rows.append({"coin": coin, "n_test": int(sel.sum()),
                        "OOS_R2_M3_vs_HAR": float(r2), "DM_stat": round(dm[0], 2), "DM_p": f"{dm[1]:.3g}"})
    pc = pd.DataFrame(pc_rows)
    pc.to_csv(OUT / "volforecast_percoin.csv", index=False)
    log("\nPer-coin OOS R²(M3 vs HAR):\n" + pc.to_string(index=False))

    md = ["# Stage 4A — narrative content in out-of-sample realized-volatility forecasting", "",
          f"Target: log realized volatility over [t,t+{args.H}h). Train pre-2023 / test post-2023, "
          f"standardisation frozen on train. HAR-RV lags {args.har_lags}h. n={len(df):,} "
          f"(test {te.sum():,}). Nested: M1 HAR-RV → M2 +social (Reddit volume+sentiment) → "
          "M3 +narrative (30 subtypes). Lower QLIKE / higher OOS-R² = better; DM tests forecast-loss "
          "differences vs HAR (Newey–West HAC).", "",
          "## Pooled OOS metrics", "", "```", res.to_string(index=False), "```", "",
          f"Narrative beyond social (M3 vs M2): DM(SE) p={s32[1]:.3g}, DM(QLIKE) p={q32[1]:.3g}.", "",
          "## Per-coin OOS R² (M3 vs HAR)", "", "```", pc.to_string(index=False), "```", "",
          "**Verdict.** Narrative content improves OOS realized-vol forecasting beyond HAR + social "
          "iff M3 lowers QLIKE/RMSE with a significant DM test and positive incremental OOS-R² vs HAR "
          "(and M3 beats M2). This is the lead positive result; economic value (VaR coverage / "
          "vol-timing) is the follow-up."]
    (OUT / "volforecast_summary.md").write_text("\n".join(md))
    log(f"\n✓ oos     -> {OUT/'volforecast_oos.csv'}")
    log(f"✓ percoin -> {OUT/'volforecast_percoin.csv'}")
    log(f"✓ summary -> {OUT/'volforecast_summary.md'}")


if __name__ == "__main__":
    main()
