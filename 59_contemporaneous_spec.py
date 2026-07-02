#!/usr/bin/env python3
"""
59_contemporaneous_spec.py
==========================
STAGE 3.9 — contemporaneous re-specification of the content/volume/return decomposition.

Scripts 53–57 used a strictly SEQUENTIAL timeline: content [t−W,t) → forward volume [t,t+m) →
return [t+m,t+H). That hard-codes "content causes FUTURE volume", which is questionable: content
(Reddit posts) and trading volume in the same window are more naturally two contemporaneous symptoms
of one attention burst. So here content and volume are measured CONCURRENTLY over [t−W,t), and we
predict the FORWARD return over [t,t+H):

    content X [t−W,t)   ┐
                        ├─→  return Y [t,t+H)      (both X and M predetermined at t; Y strictly forward)
    volume  M [t−W,t)   ┘

This drops the indefensible X-before-M lag while keeping a clean predetermined→forward structure
(leak-free, and MORE tradeable — the decision at t uses only past data). Questions:
  • a-path : content ↔ concurrent volume (an honest contemporaneous association used to split
             predictive credit, NOT a forward causal claim);
  • b-path : does concurrent volume predict the FORWARD return?
  • total / direct : does content predict the forward return at all, and does it add anything
             beyond concurrent volume (direct = content→ret controlling volume)?
  • OOS directional : nested out-of-sample models controls → +volume → +content → +content×state,
             so "does narrative add directional signal over raw volume" is tested out-of-sample.

Reuses script 55's multivariate DML (content→M, b gross/ctrl, total content→ret) verbatim for direct
comparability — only the panel construction (concurrent M, forward Y) changes.

Outputs (under paper_narrative/, never touches production):
  outputs/contemp_persubtype.csv   per-subtype a (content↔concurrent volume) + FDR + a·b
  outputs/contemp_oos.csv          nested OOS directional IC / hit-rate
  outputs/contemp_summary.md       write-up + contrast with the sequential spec
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


def assemble_contemp(coins, W, H, min_items, min_posts, z_win_h):
    """Panel with CONCURRENT volume M over [t−W,t) and FORWARD return/vol over [t,t+H)."""
    panel = pd.read_parquet(PANEL, columns=["ts", "coin", "ms_regime_prob_high", "vol_regime"])
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    panel["vol_ord"] = panel["vol_regime"].cat.codes.astype(float)
    caches = {c: build_coin_caches(c) for c in set(coins) | {"BTC"}}
    btc = caches["BTC"]["close"]

    parts = []
    for coin in coins:
        d = build_bars(coin, W)
        pc = panel[panel["coin"] == coin].sort_values("ts")
        d = pd.merge_asof(d.sort_values("ts"), pc[["ts", "ms_regime_prob_high", "vol_ord"]],
                          on="ts", direction="backward")
        bars_h = d["h"].values.astype(int)
        c = caches[coin]
        close, hmin, hmax = c["close"], c["hmin"], c["hmax"]
        cum_r2, cum_vol = c["cum_r2"], c["cum_vol"]
        i = bars_h - hmin
        span = hmax - hmin
        n = len(bars_h)
        # CONCURRENT volume over [t−W, t)  (dense hours [i−W, i))
        M = np.full(n, np.nan)
        okm = (i - W >= 0) & (i <= span + 1)
        M[okm] = np.log1p(np.clip(cum_vol[i[okm]] - cum_vol[i[okm] - W], 0, None))
        # FORWARD return over [t, t+H)
        Yret = close.reindex(bars_h + H).values / close.reindex(bars_h).values - 1.0
        # FORWARD realised vol over (t, t+H]
        Yvol = np.full(n, np.nan)
        okv = (i + 1 >= 0) & (i + H + 1 <= span + 1)
        Yvol[okv] = np.sqrt(np.clip(cum_r2[i[okv] + H + 1] - cum_r2[i[okv] + 1], 0, None))
        d["M"] = M; d["Yret"] = Yret; d["Yvol"] = Yvol
        d["Z"] = btc.reindex(bars_h).values / btc.reindex(bars_h - z_win_h).values - 1.0
        d = d[(d["h"] - d["h"].min()) % H == 0]                 # non-overlap at step=H
        parts.append(d)

    df = pd.concat(parts, ignore_index=True)
    keep = (df["n_items"] >= min_items) & (df["n_posts"] >= min_posts) \
        & df["M"].notna() & df["Yret"].notna() & df["Yvol"].notna() \
        & df["Z"].notna() & df["trail_ret"].notna() & df["ms_regime_prob_high"].notna()
    df = df[keep].copy()
    for col in ["Yret", "Yvol", "M", "Z", "trail_ret"]:
        lo, hi = df[col].quantile([WINSOR, 1 - WINSOR])
        df[col] = df[col].clip(lo, hi)
    df["is_train"] = df["ts"] < TRAIN_END
    return df


def oos_directional(df, log):
    """Nested OOS directional models: does content add over CONCURRENT volume, out-of-sample?"""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    tr = df["is_train"].values; te = ~tr
    if te.sum() < 200 or tr.sum() < 200:
        log("  [oos] insufficient rows; skip"); return pd.DataFrame()
    y = df["Yret"].values
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float).values

    def sc(cols):
        raw = df[cols].values
        return StandardScaler().fit(raw[tr]).transform(raw)
    ctrl = sc(["trail_ret", "ms_regime_prob_high", "vol_ord"])
    vol = sc(["M"])                                              # concurrent volume
    content = sc([f"share_{s}" for s in SUB30])
    Zc = sc(["Z"])
    content_x_state = content * Zc

    designs = {
        "i_controls": np.column_stack([coin_d, ctrl]),
        "ii_+concurrent_vol": np.column_stack([coin_d, ctrl, vol]),
        "iii_+content": np.column_stack([coin_d, ctrl, vol, content]),
        "iv_+content_x_state": np.column_stack([coin_d, ctrl, vol, content, content_x_state]),
    }
    alphas = np.logspace(-2, 4, 25); rows = []
    for name, X in designs.items():
        mdl = RidgeCV(alphas=alphas).fit(X[tr], y[tr])
        pred = mdl.predict(X[te]); yte = y[te]
        ic = float(pd.Series(pred).corr(pd.Series(yte), method="spearman"))
        hit = float(((pred > 0) == (yte > 0)).mean())
        ls = float(np.mean(np.sign(pred) * yte))
        rows.append({"model": name, "alpha": float(mdl.alpha_), "n_test": int(te.sum()),
                     "dir_IC_spearman": ic, "sign_hit_rate": hit, "long_short_mean": ls})
        log(f"  [oos] {name:22s}: dir-IC={ic:+.4f}  hit={hit:.4f}  L/S={ls:+.5f}")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=COINS)
    ap.add_argument("--W", type=int, default=24)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--z-win", type=int, default=168)
    ap.add_argument("--min-items", type=int, default=10)
    ap.add_argument("--min-posts", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help="W×H robustness sweep (DML-only per cell): the contemporaneous analogue "
                         "of the sequential m×H sweep. Free windows here are W (concurrent "
                         "content+volume window) and H (forward-return horizon) — there is no m.")
    ap.add_argument("--sweep-W", nargs="+", type=int, default=[4, 24, 72])
    ap.add_argument("--sweep-H", nargs="+", type=int, default=[8, 24, 48])
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = print
    if args.smoke:
        args.coins = ["BTC", "ETH", "SOL"]; args.folds = 3
        args.sweep_W, args.sweep_H = [24], [24]

    # ---------- W×H sweep mode (DML-only per cell, mirrors script 55's sweep) ----------
    if args.sweep:
        sfx = args.out_suffix
        log(f"CONTEMPORANEOUS W×H sweep: W∈{args.sweep_W} × H∈{args.sweep_H}")
        rows = []
        for W in args.sweep_W:
            for H in args.sweep_H:
                df = assemble_contemp(args.coins, W, H, args.min_items, args.min_posts, args.z_win)
                names, C = m55.content_matrix(df, "sub30")
                table, meta = m55.dml_multivariate_apath(df, names, C, args.folds, args.seed)
                nsig = int(table["sig_fdr_5pct"].sum())
                rows.append({"W": W, "H": H, "n": meta["n"], "joint_a_p": meta["joint_p"],
                             "dR2_content_on_M": meta["dR2_content_on_M"], "b_gross": meta["b"],
                             "b_ctrl_content": meta["b_ctrl"], "total_ret_p": meta["total_ret_p"],
                             "total_ret_dR2": meta["total_ret_dR2"], "nsig_fdr_30": nsig})
                log(f"  W={W} H={H}: n={meta['n']:,}  joint a-p={meta['joint_p']:.3g}  "
                    f"ΔR²(↔M)={meta['dR2_content_on_M']:+.5f}  b={meta['b']:+.4f}/{meta['b_ctrl']:+.4f}  "
                    f"tot→ret p={meta['total_ret_p']:.3g}  #sig={nsig}/30")
        sweep = pd.DataFrame(rows)
        sweep.to_csv(OUT / f"contemp_sweep{sfx}.csv", index=False)
        md = ["# Stage 3.9b — contemporaneous spec W×H robustness sweep", "",
              "content+volume concurrent over [t−W,t); return forward [t,t+H). DML per cell (sub30). "
              "The contemporaneous analogue of script 55's sequential m×H sweep (no m here).", "",
              "```", sweep.to_string(index=False), "```", "",
              "a-path (content↔concurrent volume) strong across cells + TOTAL content→forward-return "
              "null across cells ⇒ the contemporaneous conclusions are not W24/H24-specific."]
        (OUT / f"contemp_sweep_summary{sfx}.md").write_text("\n".join(md))
        log(f"\n✓ sweep   -> {OUT/f'contemp_sweep{sfx}.csv'}")
        log(f"✓ summary -> {OUT/f'contemp_sweep_summary{sfx}.md'}")
        return

    log(f"CONTEMPORANEOUS spec: content[t-{args.W}h,t) + volume[t-{args.W}h,t) -> return[t,t+{args.H}h)")
    df = assemble_contemp(args.coins, args.W, args.H, args.min_items, args.min_posts, args.z_win)
    log(f"  pooled bars n={len(df):,}  (train {int(df['is_train'].sum()):,} / "
        f"test {int((~df['is_train']).sum()):,})  corr(content-mean? n/a); corr(M,Yret)="
        f"{np.corrcoef(df['M'], df['Yret'])[0,1]:+.3f}")

    # multivariate DML (content -> concurrent M; b gross/ctrl; TOTAL content -> forward return)
    names, C = m55.content_matrix(df, "sub30")
    table, meta = m55.dml_multivariate_apath(df, names, C, args.folds, args.seed)
    table.insert(0, "granularity", "sub30")
    table.to_csv(OUT / "contemp_persubtype.csv", index=False)
    nsig = int(table["sig_fdr_5pct"].sum())
    log(f"  [DML] content↔concurrent-vol: joint a-p={meta['joint_p']:.3g}  ΔR²={meta['dR2_content_on_M']:+.5f}"
        f"  #sig={nsig}/30")
    log(f"        b(vol→fwd-ret gross)={meta['b']:+.4f}  b(ctrl content)={meta['b_ctrl']:+.4f}  | "
        f"TOTAL content→fwd-ret p={meta['total_ret_p']:.3g}  ΔR²={meta['total_ret_dR2']:+.5f}")

    log("OOS directional (does content add over concurrent volume?):")
    oos = oos_directional(df, log)
    if not oos.empty:
        oos.to_csv(OUT / "contemp_oos.csv", index=False)

    md = ["# Stage 3.9 — contemporaneous content+volume → forward return", "",
          f"content[t−{args.W}h,t) and volume[t−{args.W}h,t) measured CONCURRENTLY; return over "
          f"[t,t+{args.H}h) forward. n={len(df):,}. Drops the sequential spec's 'content→future "
          "volume' lag. DML reused from script 55 for comparability.", "",
          "## Multivariate decomposition", "",
          f"- content ↔ concurrent volume (a): joint p={meta['joint_p']:.3g}, ΔR²={meta['dR2_content_on_M']:+.5f}, "
          f"{nsig}/30 FDR.",
          f"- concurrent volume → forward return (b): gross {meta['b']:+.4f}, ctrl-content {meta['b_ctrl']:+.4f}.",
          f"- **TOTAL content → forward return: p={meta['total_ret_p']:.3g}, ΔR²={meta['total_ret_dR2']:+.5f}**.",
          "", "Per-subtype in `contemp_persubtype.csv`."]
    if not oos.empty:
        md += ["", "## OOS directional (nested)", "", "```", oos.to_string(index=False), "```", "",
               "Content adds directional signal over raw concurrent volume iff `iii_+content` / "
               "`iv_+content_x_state` beat `ii_+concurrent_vol` out-of-sample."]
    (OUT / "contemp_summary.md").write_text("\n".join(md))
    log(f"\n✓ persubtype -> {OUT/'contemp_persubtype.csv'}")
    log(f"✓ oos        -> {OUT/'contemp_oos.csv'}")
    log(f"✓ summary    -> {OUT/'contemp_summary.md'}")


if __name__ == "__main__":
    main()
