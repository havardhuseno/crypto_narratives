#!/usr/bin/env python3
"""
46_intraday_decomposition.py
============================
Leak-free INTRADAY decomposition of the Reddit social-media return signal, the Design-B
upgrade of `44_narrative_decomposition.py`. Where 44 used the production panel's monthly
`narr_cb_*` shares (which carry a monthly-clustering look-ahead leak), this script builds
features from the per-post topic classifier of script 45 — no clustering at inference, so
the rolling `[t-W, t)` topic/sentiment/volume shares are strictly trailing and leak-free.

Question (the QF paper's empirical core): does narrative CONTENT — and its DYNAMICS —
add return-predictability beyond attention + polarity, at intraday cadence?

Five layers (hierarchical, added in order):
  L1 ATTENTION  log post/comment counts in window
  L2 POLARITY   net_sentiment (bull-bear share), intent shares (buy/sell/fomo/fear)
  L3 CONTENT    34 codebook-subtype shares (posts only — topic exists for posts)
  L4 DYNAMICS   topic HHI concentration, topic-shift (1-cosine vs prior window), novelty
  L5 BREADTH    author activity / item ratio (herding proxy), engagement (comments/post)
Controls (always in, never a "layer"): coin FE, market regime (ms_regime_prob_high),
vol regime, and TRAILING RETURN over [t-W,t) (reverse causality is acute intraday —
posting reflects past price).

Data units (per the design): topic = posts only; attention/polarity/breadth = posts +
comments (comments dominate volume and match the production sentiment panel).

Eval (BOTH, per design):
  * HEADLINE   non-overlapping bars (step = H): SEs clustered by timestamp, no HAC needed.
  * ROBUSTNESS dense overlapping bars (step = 1h): Driscoll-Kraay HAC SEs (handles the
               overlap-induced MA autocorrelation + cross-coin correlation).

Inputs (read-only main repo + script-45 outputs under paper_narrative/):
  paper_narrative/data/post_topics/{coin}.parquet         (id, created_utc, pred_subtype, pred_proba)
  data/processed/labeled_full/{coin}/{stance,intent}/{posts,comments}_b*.parquet
  data/processed/reddit_clean/{coin}/posts.parquet        (id, score for engagement)
  data/raw/prices/{coin}_1h.parquet                       (hourly close)
  data/panel/full_panel_4h_v11.parquet                    (ms_regime_prob_high, vol_regime)
Outputs:
  paper_narrative/outputs/intraday_decomposition_{layers,partial_ic,summary.md}
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

TOPIC_DIR = PAPER_ROOT / "data" / "post_topics"
LABELED = PROJECT_ROOT / "data" / "processed" / "labeled_full"
REDDIT = PROJECT_ROOT / "data" / "processed" / "reddit_clean"
PRICES = PROJECT_ROOT / "data" / "raw" / "prices"
PANEL = PROJECT_ROOT / "data" / "panel" / "full_panel_4h_v11.parquet"
OUT = PAPER_ROOT / "outputs"

COINS = ["BTC", "ETH", "DOGE", "XRP", "ADA", "SOL"]
CB_LABELS = [f"3.{i}" for i in range(1, 5)] + [f"4.{i}" for i in range(1, 11)] + \
            [f"5.{i}" for i in range(1, 21)]
CONTENT_COLS = [f"share_{l}" for l in CB_LABELS]
HOUR = 3600
MIN_ITEMS = 10        # drop bars with < this many items (posts+comments) in window
WINSOR = 0.005

# default grid
WINDOWS_H = [4, 2]    # window W in hours
HORIZONS_H = [1, 2, 4]  # forward horizon H in hours


# ============================================================ stream assembly
def _read_batches(coin: str, kind: str, ctype: str) -> pd.DataFrame:
    """Concat stance|intent batch files for a content-type ('posts'|'comments')."""
    files = sorted(glob.glob(str(LABELED / coin / kind / f"{ctype}_b*.parquet")))
    if not files:
        return pd.DataFrame(columns=["id", "created_utc", "author", kind])
    cols = ["id", "created_utc", "author", kind]
    return pd.concat([pd.read_parquet(f, columns=cols) for f in files], ignore_index=True)


def load_stream(coin: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (posts_df, all_df).
      posts_df: text-bearing posts with topic + stance + intent + score (content/dynamics).
      all_df  : posts+comments with stance + intent + author (attention/polarity/breadth)."""
    topic = pd.read_parquet(TOPIC_DIR / f"{coin}.parquet")  # id, created_utc, pred_subtype
    st_p = _read_batches(coin, "stance", "posts")
    in_p = _read_batches(coin, "intent", "posts")
    st_c = _read_batches(coin, "stance", "comments")
    in_c = _read_batches(coin, "intent", "comments")

    # posts: topic + stance + intent + score
    posts = topic.merge(st_p[["id", "stance"]], on="id", how="left") \
                 .merge(in_p[["id", "intent"]], on="id", how="left")
    try:
        score = pd.read_parquet(REDDIT / coin / "posts.parquet", columns=["id", "score"])
        posts = posts.merge(score, on="id", how="left")
    except Exception:
        posts["score"] = np.nan

    # all items (posts + comments): stance + intent + author + created_utc
    a_st = pd.concat([st_p[["id", "created_utc", "author", "stance"]],
                      st_c[["id", "created_utc", "author", "stance"]]], ignore_index=True)
    a_in = pd.concat([in_p[["id", "intent"]], in_c[["id", "intent"]]], ignore_index=True)
    all_df = a_st.merge(a_in, on="id", how="left")
    return posts, all_df


# ============================================================ hourly aggregation
def hourly_matrix(posts: pd.DataFrame, all_df: pd.DataFrame,
                  h0: int, h1: int) -> pd.DataFrame:
    """Additive per-hour aggregates on a contiguous integer-hour grid [h0, h1].
    Everything here is summable, so window sums come from cumsums (fast + exact)."""
    idx = np.arange(h0, h1 + 1)
    out = pd.DataFrame(index=idx)

    # ---- posts (content / dynamics / engagement) ----
    p = posts.dropna(subset=["created_utc"]).copy()
    p["h"] = (p["created_utc"].astype("int64") // HOUR)
    p = p[(p["h"] >= h0) & (p["h"] <= h1)]
    out["n_posts"] = p.groupby("h").size().reindex(idx, fill_value=0)
    out["sum_score"] = p.groupby("h")["score"].sum().reindex(idx, fill_value=0.0)
    # per-subtype counts
    sub = (p.groupby(["h", "pred_subtype"]).size().unstack(fill_value=0)
           .reindex(index=idx, fill_value=0))
    for l in CB_LABELS:
        out[f"cnt_{l}"] = sub[l] if l in sub.columns else 0

    # ---- all items (attention / polarity / breadth) ----
    a = all_df.dropna(subset=["created_utc"]).copy()
    a["h"] = (a["created_utc"].astype("int64") // HOUR)
    a = a[(a["h"] >= h0) & (a["h"] <= h1)]
    out["n_items"] = a.groupby("h").size().reindex(idx, fill_value=0)
    for s in ["bullish", "bearish", "neutral", "mixed"]:
        out[f"st_{s}"] = a[a["stance"] == s].groupby("h").size().reindex(idx, fill_value=0)
    for it in ["buy", "sell", "hold", "fomo", "fear", "none"]:
        out[f"in_{it}"] = a[a["intent"] == it].groupby("h").size().reindex(idx, fill_value=0)
    # hourly distinct authors (additive proxy for breadth/herding when summed over window)
    out["auth_hours"] = a.groupby("h")["author"].nunique().reindex(idx, fill_value=0)
    return out.fillna(0.0)


def _cumwin(mat: np.ndarray, bars_h: np.ndarray, h0: int, W: int) -> np.ndarray:
    """Sum of rows over hour-window [t-W, t) for each bar hour t. mat is (H,K) on
    contiguous hours starting at h0. Returns (len(bars),K)."""
    C = np.vstack([np.zeros((1, mat.shape[1])), np.cumsum(mat, axis=0)])  # (H+1,K)
    a = (bars_h - W - h0).clip(0, mat.shape[0])
    b = (bars_h - h0).clip(0, mat.shape[0])
    return C[b] - C[a]


# ============================================================ per-coin feature build
def build_bars(coin: str, W: int) -> pd.DataFrame:
    """One row per hourly bar t (dense). Features summarise posts/items in [t-W, t);
    targets/ trailing return from 1h close. Regime merged trailing from the 4h panel."""
    price = pd.read_parquet(PRICES / f"{coin}_1h.parquet")
    price = price[~price.index.duplicated(keep="last")].sort_index()
    price.index = pd.to_datetime(price.index, utc=True)
    ph = (price.index.view("int64") // 10**9 // HOUR).astype(int)
    close = pd.Series(price["close"].values, index=ph)
    close = close[~close.index.duplicated(keep="last")]

    posts, all_df = load_stream(coin)
    tmin = int(min(posts["created_utc"].min(), all_df["created_utc"].min()) // HOUR)
    tmax = int(max(posts["created_utc"].max(), all_df["created_utc"].max()) // HOUR)
    h0, h1 = min(tmin, int(close.index.min())), max(tmax, int(close.index.max()))
    mat = hourly_matrix(posts, all_df, h0, h1)

    # bars = hours that have a close price (so targets/trailing are well defined)
    bars_h = np.array(sorted(close.index[(close.index >= h0) & (close.index <= h1)]))
    cols = mat.columns.tolist()
    win = _cumwin(mat.values, bars_h, h0, W)
    W2 = _cumwin(mat.values, bars_h - W, h0, W)   # prior window [t-2W, t-W) for dynamics
    df = pd.DataFrame(win, columns=cols)
    df.insert(0, "h", bars_h)
    df["coin"] = coin
    df["ts"] = pd.to_datetime(df["h"] * HOUR, unit="s", utc=True)

    # ---- attention ----
    df["att_log_posts"] = np.log1p(df["n_posts"])
    df["att_log_items"] = np.log1p(df["n_items"])

    # ---- polarity ----
    denom_st = (df[["st_bullish", "st_bearish", "st_neutral", "st_mixed"]].sum(1)).replace(0, np.nan)
    df["net_sent"] = (df["st_bullish"] - df["st_bearish"]) / denom_st
    denom_in = df[["in_buy", "in_sell", "in_hold", "in_fomo", "in_fear", "in_none"]].sum(1).replace(0, np.nan)
    for it in ["buy", "sell", "fomo", "fear"]:
        df[f"intent_{it}_share"] = df[f"in_{it}"] / denom_in
    POLARITY = ["net_sent", "intent_buy_share", "intent_sell_share",
                "intent_fomo_share", "intent_fear_share"]

    # ---- content shares (posts only) ----
    cnt = df[[f"cnt_{l}" for l in CB_LABELS]].values
    psum = cnt.sum(1, keepdims=True)
    shares = np.divide(cnt, psum, out=np.zeros_like(cnt, float), where=psum > 0)
    for j, l in enumerate(CB_LABELS):
        df[f"share_{l}"] = shares[:, j]

    # ---- dynamics: HHI, topic-shift (1-cosine vs prior window), novelty ----
    df["topic_hhi"] = (shares ** 2).sum(1)
    cnt_prev = W2[:, [cols.index(f"cnt_{l}") for l in CB_LABELS]]
    psum_prev = cnt_prev.sum(1, keepdims=True)
    shares_prev = np.divide(cnt_prev, psum_prev, out=np.zeros_like(cnt_prev, float),
                            where=psum_prev > 0)
    dot = (shares * shares_prev).sum(1)
    nrm = np.linalg.norm(shares, axis=1) * np.linalg.norm(shares_prev, axis=1)
    df["topic_shift"] = 1.0 - np.divide(dot, nrm, out=np.zeros_like(dot), where=nrm > 0)
    novel = ((shares > 0) & (shares_prev == 0))
    df["topic_novelty"] = (shares * novel).sum(1)   # share of posts in newly-present subtypes
    DYNAMICS = ["topic_hhi", "topic_shift", "topic_novelty"]

    # ---- breadth / engagement ----
    df["author_ratio"] = df["auth_hours"] / df["n_items"].replace(0, np.nan)
    df["comments_per_post"] = (df["n_items"] - df["n_posts"]) / df["n_posts"].replace(0, np.nan)
    df["mean_score"] = df["sum_score"] / df["n_posts"].replace(0, np.nan)
    BREADTH = ["author_ratio", "comments_per_post", "mean_score"]

    # ---- trailing return control [t-W, t) ----
    c_t = close.reindex(bars_h).values
    c_tw = close.reindex(bars_h - W).values
    df["trail_ret"] = c_t / c_tw - 1.0
    df["_close"] = c_t

    df.attrs["layers"] = {
        "attention": ["att_log_posts", "att_log_items"],
        "polarity": POLARITY,
        "content": CONTENT_COLS,
        "dynamics": DYNAMICS,
        "breadth": BREADTH,
    }
    return df


# ============================================================ stats
def attach_target(df: pd.DataFrame, coin_close: dict, H: int) -> pd.DataFrame:
    d = df.copy()
    bars_h = d["h"].values
    close = coin_close[d["coin"].iloc[0]]
    c_t = close.reindex(bars_h).values
    c_tH = close.reindex(bars_h + H).values
    d[f"fwd_ret"] = c_tH / c_t - 1.0
    return d


def zscore_within_coin(df, cols):
    df = df.copy()
    for c in cols:
        g = df.groupby("coin")[c]
        sd = g.transform("std").replace(0, np.nan)
        df[c] = ((df[c] - g.transform("mean")) / sd).fillna(0.0)
    return df


def bh_fdr(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    r = p[o] * n / (np.arange(n) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(n); out[o] = np.clip(r, 0, 1); return out


def _fit(y, X, df, overlap: bool, step_h: int, H: int):
    Xc = sm.add_constant(X, has_constant="add")
    if not overlap:
        return sm.OLS(y.values, Xc.values).fit(
            cov_type="cluster", cov_kwds={"groups": df["h"].values})
    # Driscoll-Kraay: time = integer bar period; maxlags ~ overlap length H/step
    tindex = (df["h"].values - df["h"].values.min()) // step_h
    L = max(1, int(np.ceil(H / step_h)))
    return sm.OLS(y.values, Xc.values).fit(
        cov_type="nw-groupsum", cov_kwds={"time": tindex.astype(int), "maxlags": L})


def decompose(df, layers, overlap, step_h, H):
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float)
    base = pd.concat([coin_d, df[["ms_regime_prob_high", "vol_ord", "trail_ret"]]], axis=1)
    y = df["fwd_ret"]
    order = ["attention", "polarity", "content", "dynamics", "breadth"]
    X = base.copy()
    rows, prev = [], 0.0
    feats_so_far = []
    block_tests = {}
    for name in ["controls"] + order:
        if name != "controls":
            feats_so_far = feats_so_far + layers[name]
            X = pd.concat([base, df[feats_so_far]], axis=1)
        res = _fit(y, X, df, overlap, step_h, H)
        r2 = res.rsquared
        rows.append({"layer": name, "n": int(res.nobs), "r2": r2, "delta_r2": r2 - prev})
        prev = r2
        if name != "controls":
            # joint Wald that this block's coefs == 0
            names = sm.add_constant(X, has_constant="add").columns.tolist()
            idx = [names.index(c) for c in layers[name]]
            R = np.zeros((len(idx), len(names)))
            for i, j in enumerate(idx):
                R[i, j] = 1.0
            w = res.wald_test(R, scalar=True)
            block_tests[name] = (float(w.statistic), float(w.pvalue))
    return rows, block_tests


def partial_ic(df, content_cols, overlap, step_h, H):
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float)
    ctrl = pd.concat([coin_d, df[["ms_regime_prob_high", "vol_ord", "trail_ret",
                                   "att_log_items", "net_sent"]]], axis=1)
    y = df["fwd_ret"]
    Xall = pd.concat([ctrl, df[content_cols]], axis=1)
    res = _fit(y, Xall, df, overlap, step_h, H)
    names = sm.add_constant(Xall, has_constant="add").columns.tolist()
    coefs = dict(zip(names, res.params)); ses = dict(zip(names, res.bse))
    pvs = dict(zip(names, res.pvalues))
    Xc = sm.add_constant(ctrl, has_constant="add").values
    y_res = y.values - Xc @ np.linalg.lstsq(Xc, y.values, rcond=None)[0]
    rows = []
    for c in content_cols:
        xr = df[c].values - Xc @ np.linalg.lstsq(Xc, df[c].values, rcond=None)[0]
        rho, _ = spearmanr(xr, y_res)
        rows.append({"feature": c, "beta": coefs.get(c, np.nan), "se": ses.get(c, np.nan),
                     "p_raw": pvs.get(c, np.nan), "partial_ic": float(rho)})
    out = pd.DataFrame(rows)
    out["p_fdr"] = bh_fdr(out["p_raw"].fillna(1.0).values)
    out["sig_fdr_5pct"] = out["p_fdr"] < 0.05
    return out.sort_values("p_raw").reset_index(drop=True)


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=COINS)
    ap.add_argument("--windows", nargs="+", type=int, default=WINDOWS_H)
    ap.add_argument("--horizons", nargs="+", type=int, default=HORIZONS_H)
    ap.add_argument("--min-items", type=int, default=MIN_ITEMS)
    ap.add_argument("--min-posts", type=int, default=3,
                    help="content/dynamics need a few text posts for the 34-way shares to "
                         "be meaningful (topic is posts-only)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # regime controls from the 4h panel (trailing merge_asof per coin)
    panel = pd.read_parquet(PANEL, columns=["ts", "coin", "ms_regime_prob_high", "vol_regime"])
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    panel["vol_ord"] = panel["vol_regime"].cat.codes.astype(float)

    # cache hourly close per coin for target attach
    coin_close = {}

    layer_rows, summary = [], []
    summary += ["# Intraday narrative decomposition (Design B, leak-free)", "",
                "Per-post topic classifier (script 45) -> rolling [t-W,t) features; targets "
                "[t,t+H) from 1h close. Five layers over controls (coin FE, regime, vol, "
                "trailing return). Headline = non-overlapping bars (cluster SE); robustness "
                "= overlapping 1h grid (Driscoll-Kraay HAC).", ""]
    pic_store = {}

    for W in args.windows:
        # build dense bars per coin once per W
        per_coin = []
        for coin in args.coins:
            d = build_bars(coin, W)
            layers = d.attrs["layers"]
            price = pd.read_parquet(PRICES / f"{coin}_1h.parquet")
            ph = (pd.to_datetime(price.index, utc=True).view("int64") // 10**9 // HOUR).astype(int)
            cl = pd.Series(price["close"].values, index=ph)
            coin_close[coin] = cl[~cl.index.duplicated(keep="last")]
            # merge trailing regime
            pc = panel[panel["coin"] == coin].sort_values("ts")
            d = pd.merge_asof(d.sort_values("ts"), pc[["ts", "ms_regime_prob_high", "vol_ord"]],
                              on="ts", direction="backward")
            per_coin.append(d)
        allbars = pd.concat(per_coin, ignore_index=True)

        for H in args.horizons:
            for overlap in [False, True]:
                step_h = H if not overlap else 1
                parts = []
                for coin in args.coins:
                    d = allbars[allbars["coin"] == coin]
                    d = attach_target(d, coin_close, H)
                    if not overlap:                       # keep non-overlapping bars only
                        d = d[(d["h"] - d["h"].min()) % H == 0]
                    parts.append(d)
                dd = pd.concat(parts, ignore_index=True)
                dd = dd[(dd["n_items"] >= args.min_items) & (dd["n_posts"] >= args.min_posts)
                        & dd["fwd_ret"].notna() & dd["trail_ret"].notna()
                        & dd["ms_regime_prob_high"].notna()].copy()
                if len(dd) < 500:
                    continue
                lo, hi = dd["fwd_ret"].quantile([WINSOR, 1 - WINSOR])
                dd["fwd_ret"] = dd["fwd_ret"].clip(lo, hi)
                zcols = (layers["attention"] + layers["polarity"] + layers["content"]
                         + layers["dynamics"] + layers["breadth"] + ["trail_ret"])
                dd = zscore_within_coin(dd, zcols)

                rows, blocks = decompose(dd, layers, overlap, step_h, H)
                tag = f"W{W}h_H{H}h_{'overlap' if overlap else 'nonoverlap'}"
                for r in rows:
                    r.update({"tag": tag, "W": W, "H": H, "overlap": overlap})
                    layer_rows.append(r)
                pic = partial_ic(dd, layers["content"], overlap, step_h, H)
                pic_store[tag] = pic
                nsig = int(pic["sig_fdr_5pct"].sum())

                cwald = blocks.get("content", (np.nan, np.nan))
                dwald = blocks.get("dynamics", (np.nan, np.nan))
                print(f"{tag}: n={len(dd):,}  content Wald p={cwald[1]:.3g}  "
                      f"dynamics Wald p={dwald[1]:.3g}  #sig/34={nsig}")
                summary += [f"## {tag}  (n={len(dd):,})", "",
                            "| layer | ΔR² | block Wald p |", "|---|---|---|"]
                for r in rows:
                    bp = blocks.get(r["layer"], (None, None))[1]
                    bp = f"{bp:.3g}" if bp is not None else "—"
                    summary.append(f"| {r['layer']} | {r['delta_r2']:+.6f} | {bp} |")
                summary += ["", f"content: **{nsig}/34** subtypes survive BH-FDR 5%.", ""]

    pd.DataFrame(layer_rows).to_csv(OUT / "intraday_decomposition_layers.csv", index=False)
    # save the headline-cell partial IC tables
    if pic_store:
        cat = pd.concat([v.assign(tag=k) for k, v in pic_store.items()], ignore_index=True)
        cat.to_csv(OUT / "intraday_decomposition_partial_ic.csv", index=False)
    (OUT / "intraday_decomposition_summary.md").write_text("\n".join(summary))
    print(f"\n✓ layers   -> {OUT/'intraday_decomposition_layers.csv'}")
    print(f"✓ partial  -> {OUT/'intraday_decomposition_partial_ic.csv'}")
    print(f"✓ summary  -> {OUT/'intraday_decomposition_summary.md'}")


if __name__ == "__main__":
    main()
