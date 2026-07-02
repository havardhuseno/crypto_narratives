#!/usr/bin/env python3
"""
57_market_augmented.py
======================
STAGE 3.8 — add the GENERAL-MARKET Reddit stream (r/CryptoCurrency + r/CryptoMarkets) as SHARED,
non-coin-specific inputs and ask whether market-wide narrative adds anything the coin-specific
streams missed. Per user: the general subreddits "enter as content features and as volume/attention
features — just like the other coin-specific ones but not subsetted into coin-specific effects."

Market features (common to ALL coins at a given timestamp t, broadcast by hour index; leak-free):
  • market content: 30-subtype shares of general POSTS over the trailing [t−W, t) window
    (topics from script 56's general.parquet; noise family 3.x dropped).
  • market attention/volume + polarity: from the production `market_4h_sentiment.parquet`
    (market_n_total, market_net_sentiment), merged as-of the last 4h bar COMPLETED by t.

Two tests (reusing scripts 55/58 machinery), each asking "beyond the coin-specific content":
  A) a-path augmentation — does market content predict the coin's forward-volume mediator M
     beyond coin content + controls? (multivariate DML joint test on the market-content block.)
  B) directional augmentation (OOS) — nested out-of-sample directional models:
       controls → +coin content → +market content → +market content × market-state.
     Decisive test of whether market-wide narrative adds DIRECTIONAL return signal.

Outputs (under paper_narrative/, never touches production):
  outputs/market_aug_apath.csv     coin-only vs +market-content joint a-path on M
  outputs/market_aug_oos.csv       nested OOS directional IC / hit-rate
  outputs/market_aug_summary.md    write-up + verdict
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
m54 = m55.m54
assemble = m53.assemble
COINS = m53.COINS
SUB30 = m55.SUB30
zsc = m55.zsc
HOUR = m53.HOUR
TOPIC_DIR = m46.TOPIC_DIR
MARKET_SENT = PAPER_ROOT / "data" / "market_4h_sentiment.parquet"  # v2: freshly rebuilt (56b), not stale parent
make_folds = m54.make_folds
crossfit_resid = m54.crossfit_resid
bh_fdr = m55.bh_fdr


# ============================================================ market feature builder
def build_market_features(W: int):
    """Hour-indexed market features broadcast across coins:
       market_share_<sub> : general-POST 30-subtype shares over [t-W, t)
       market_att         : log(market_n_total) from last 4h bar completed by t
       market_net_sent    : market net sentiment, same trailing bar
    Returns a DataFrame indexed by integer hour h with these columns."""
    # --- market content shares over [t-W,t) from general post topics ---
    gen = pd.read_parquet(TOPIC_DIR / "general.parquet", columns=["created_utc", "pred_subtype"])
    gen["h"] = (gen["created_utc"].astype("int64") // HOUR).astype(int)
    h0, h1 = int(gen["h"].min()), int(gen["h"].max())
    idx = np.arange(h0, h1 + 1)
    sub_cnt = (gen.groupby(["h", "pred_subtype"]).size().unstack(fill_value=0)
               .reindex(index=idx, fill_value=0))
    keep = [s for s in SUB30 if s in sub_cnt.columns]
    mat = sub_cnt[keep].values.astype(float)                 # (Hh, 30) hourly counts
    # cumulative window-sum over [t-W, t) for every hour t
    C = np.vstack([np.zeros((1, mat.shape[1])), np.cumsum(mat, axis=0)])
    def winsum(bars_h):
        a = (bars_h - W - h0).clip(0, mat.shape[0])
        b = (bars_h - h0).clip(0, mat.shape[0])
        return C[b] - C[a]
    win = winsum(idx)
    psum = win.sum(1, keepdims=True)
    shares = np.divide(win, psum, out=np.zeros_like(win), where=psum > 0)
    mfeat = pd.DataFrame(shares, columns=[f"market_share_{s}" for s in keep], index=idx)

    # --- market attention + sentiment from the 4h aggregate, as-of last bar completed by t ---
    ms = pd.read_parquet(MARKET_SENT, columns=["bar_start_utc", "market_n_total", "market_net_sentiment"])
    ms["bar_start_utc"] = pd.to_datetime(ms["bar_start_utc"], utc=True)
    ms["ready_h"] = (ms["bar_start_utc"].view("int64") // 10**9 // HOUR).astype(int) + 4  # bar completes +4h
    ms = ms.sort_values("ready_h")
    ms["market_att"] = np.log1p(ms["market_n_total"].astype(float))
    asof = pd.merge_asof(pd.DataFrame({"h": idx}), ms[["ready_h", "market_att", "market_net_sentiment"]],
                         left_on="h", right_on="ready_h", direction="backward")
    mfeat["market_att"] = asof["market_att"].values
    mfeat["market_net_sent"] = asof["market_net_sentiment"].values
    mfeat.index.name = "h"
    return mfeat.reset_index()


def merge_market(df, mfeat):
    out = df.merge(mfeat, on="h", how="left")
    mcols = [c for c in mfeat.columns if c != "h"]
    # drop rows with no market coverage; fill rare sentiment gaps with 0 (neutral)
    out = out[out[[c for c in mcols if c.startswith("market_share_")]].notna().all(axis=1)].copy()
    out["market_att"] = out["market_att"].fillna(out["market_att"].median())
    out["market_net_sent"] = out["market_net_sent"].fillna(0.0)
    return out, mcols


# ============================================================ A) a-path augmentation
def apath_augment(df, mcols, folds, seed, log):
    from sklearn.ensemble import HistGradientBoostingRegressor
    import statsmodels.api as sm
    rng = np.random.default_rng(seed)
    fset = make_folds(len(df), folds, rng)

    def gbr():
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                             min_samples_leaf=50, l2_regularization=1.0, random_state=seed)
    M = zsc(df["M"].values)
    coin_content = np.column_stack([zsc(df[f"share_{s}"].values) for s in SUB30])
    mkt_share_cols = [c for c in mcols if c.startswith("market_share_")]
    mkt_content = np.column_stack([zsc(df[c].values) for c in mkt_share_cols])
    mkt_other = np.column_stack([zsc(df["market_att"].values), zsc(df["market_net_sent"].values)])
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float).values
    W = np.column_stack([zsc(df["ms_regime_prob_high"].values), zsc(df["vol_ord"].values),
                         zsc(df["trail_ret"].values), coin_d, coin_content, mkt_other])
    # residualise M and each market-content col on (controls + coin content + market other)
    M_r, _ = crossfit_resid(M, W, fset, gbr)
    Cm_r = np.column_stack([crossfit_resid(mkt_content[:, j], W, fset, gbr)[0]
                            for j in range(mkt_content.shape[1])])
    Xc = sm.add_constant(Cm_r, has_constant="add")
    res = sm.OLS(M_r, Xc).fit(cov_type="cluster", cov_kwds={"groups": df["h"].values})
    R = np.zeros((mkt_content.shape[1], Xc.shape[1])); R[:, 1:] = np.eye(mkt_content.shape[1])
    jp = float(np.asarray(res.f_test(R).pvalue).ravel()[0])
    dR2 = 1.0 - np.sum(res.resid ** 2) / np.sum((M_r - M_r.mean()) ** 2)
    pv = res.pvalues[1:]; fdr = bh_fdr(pv); nsig = int((fdr < 0.05).sum())
    log(f"  [a-path] market-content → coin M | beyond coin content+controls: "
        f"joint p={jp:.3g}  incremental ΔR²={dR2:+.5f}  #sig_FDR={nsig}/{len(mkt_share_cols)}")
    tab = pd.DataFrame({"market_subtype": [c.replace("market_share_", "") for c in mkt_share_cols],
                        "coef": res.params[1:], "p": pv, "fdr": fdr, "sig_fdr_5pct": fdr < 0.05})
    tab.to_csv(OUT / "market_aug_apath.csv", index=False)
    return {"joint_p": jp, "incr_dR2": float(dR2), "nsig": nsig, "n_market_feat": len(mkt_share_cols)}


# ============================================================ B) directional augmentation (OOS)
def oos_augment(df, mcols, log):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    tr = df["is_train"].values; te = ~tr
    if te.sum() < 200 or tr.sum() < 200:
        log("  [oos] insufficient rows; skip"); return pd.DataFrame()
    y = df["Yret"].values
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float).values

    def sc(cols):
        raw = df[cols].values
        s = StandardScaler().fit(raw[tr]); return s.transform(raw)
    ctrl = sc(["trail_ret", "ms_regime_prob_high", "vol_ord"])
    coin_content = sc([f"share_{s}" for s in SUB30])
    mkt_share_cols = [c for c in mcols if c.startswith("market_share_")]
    mkt_content = sc(mkt_share_cols)
    mkt_state = sc(["market_att", "market_net_sent"])
    # market content × market-state (net sentiment) interaction block
    mkt_inter = mkt_content * mkt_state[:, 1:2]

    designs = {
        "i_controls": np.column_stack([coin_d, ctrl]),
        "ii_coin_content": np.column_stack([coin_d, ctrl, coin_content]),
        "iii_+market_content": np.column_stack([coin_d, ctrl, coin_content, mkt_content, mkt_state]),
        "iv_+market_x_state": np.column_stack([coin_d, ctrl, coin_content, mkt_content, mkt_state, mkt_inter]),
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
    out = pd.DataFrame(rows); out.to_csv(OUT / "market_aug_oos.csv", index=False)
    return out


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=COINS)
    ap.add_argument("--W", type=int, default=24)
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--H", type=int, default=24)
    ap.add_argument("--z-win", type=int, default=168)
    ap.add_argument("--min-items", type=int, default=10)
    ap.add_argument("--min-posts", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = print
    if args.smoke:
        args.coins = ["BTC", "ETH", "SOL"]; args.folds = 3

    log(f"assembling coin panel W={args.W}h m={args.m}h H={args.H}h coins={args.coins}")
    df = assemble(args.coins, args.W, args.m, args.H, args.min_items, args.min_posts, args.z_win)
    log(f"  coin bars n={len(df):,}")
    log("building market features (general-stream content [t-W,t) + 4h attention/sentiment)…")
    mfeat = build_market_features(args.W)
    df, mcols = merge_market(df, mfeat)
    nshare = len([c for c in mcols if c.startswith('market_share_')])
    log(f"  merged market features; n with market coverage={len(df):,}  ({nshare} market subtypes)")

    log("A) a-path augmentation — does market content predict coin M beyond coin content?")
    a = apath_augment(df, mcols, args.folds, args.seed, log)
    log("B) OOS directional augmentation — nested models:")
    oos = oos_augment(df, mcols, log)

    md = ["# Stage 3.8 — market-augmented decomposition (general-stream as shared inputs)", "",
          f"Coin panel W={args.W}h→m={args.m}h→H={args.H}h, n={len(df):,} (with market coverage). "
          "Market features broadcast across coins by hour (leak-free): general-post 30-subtype content "
          "shares over [t−W,t); market attention log(n_total) + net sentiment from the 4h aggregate "
          "as-of last bar completed by t.", "",
          "## A) Market content → coin volume mediator M (beyond coin content + controls)", "",
          f"- joint p = **{a['joint_p']:.3g}**, incremental ΔR² = **{a['incr_dR2']:+.5f}**, "
          f"#sig FDR = {a['nsig']}/{a['n_market_feat']}. Per-subtype in `market_aug_apath.csv`."]
    if not oos.empty:
        md += ["", "## B) Out-of-sample directional return (nested models)", "",
               "```", oos.to_string(index=False), "```", "",
               "Market-wide narrative adds directional signal iff `iii_+market_content` / "
               "`iv_+market_x_state` beat `ii_coin_content` ≈ `i_controls` out-of-sample. "
               "Otherwise the market stream — like the coin streams — moves activity, not signed return."]
    (OUT / "market_aug_summary.md").write_text("\n".join(md))
    log(f"\n✓ a-path  -> {OUT/'market_aug_apath.csv'}")
    log(f"✓ oos     -> {OUT/'market_aug_oos.csv'}")
    log(f"✓ summary -> {OUT/'market_aug_summary.md'}")


if __name__ == "__main__":
    main()
