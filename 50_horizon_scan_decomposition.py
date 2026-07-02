#!/usr/bin/env python3
"""
50_horizon_scan_decomposition.py
================================
STAGE 1 of the long-horizon S1 investigation. Extends the leak-free intraday
decomposition (`46_intraday_decomposition.py`) into a *horizon scan*: does the Reddit
social-media signal — and specifically narrative CONTENT/DYNAMICS — add predictability
beyond attention + polarity at LONGER horizons and against NON-return targets?

Three things change vs script 46:
  1. HORIZONS  H ∈ {1, 2, 4, 8, 12, 24, 48, 72, 168} hours (intraday → weekly).
  2. WINDOWS   feature window W ∈ {4h, 24h, 72h} (short → multi-day rolling shares).
  3. TARGETS   three families, all forward over [t, t+H):
       - ret    : forward simple return            close[t+H]/close[t] - 1
       - absret : |forward return|                 (magnitude / vol proxy)
       - rvol   : realized vol                      sqrt(Σ hourly-logret² over (t,t+H])
       - fvol   : forward volume (log1p)            log1p(Σ hourly volume over (t,t+H])
     ret is the headline (alpha question). absret/rvol/fvol probe whether S1 content
     predicts ACTIVITY/RISK even where it does not predict signed return — the most
     defensible place a social-media effect could still live (Shiller: narratives move
     attention and volatility before they move price).

Everything else is inherited from script 46 unchanged (5 hierarchical layers over the
same controls — coin FE, ms_regime_prob_high, vol_ord, trailing return; non-overlap
cluster-SE headline + overlapping Driscoll-Kraay HAC robustness; per-subtype partial IC
with BH-FDR). We reuse 46's functions directly via importlib so the two scripts can never
silently diverge.

Outputs (all under paper_narrative/, never touches production):
  outputs/horizon_scan_layers.csv        one row per (target, W, H, overlap, layer)
  outputs/horizon_scan_partial_ic.csv    per-subtype partial IC + FDR per cell
  outputs/horizon_scan_content_curve.csv ΔR²(content) & ΔR²(dynamics) vs H headline curve
  outputs/horizon_scan_summary.md        human-readable per-cell tables + the curve
"""
from __future__ import annotations

import argparse
import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# ---- import script 46's machinery (module name starts with a digit -> importlib) ----
_spec = importlib.util.spec_from_file_location(
    "dec46", HERE / "46_intraday_decomposition.py")
m46 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m46)

# cache the (expensive) per-coin stream assembly so build_bars across multiple W reuses it
_orig_load_stream = m46.load_stream
m46.load_stream = lru_cache(maxsize=None)(_orig_load_stream)

PRICES = m46.PRICES
PANEL = m46.PANEL
OUT = m46.OUT
HOUR = m46.HOUR
COINS = m46.COINS
WINSOR = m46.WINSOR

# scan grid
WINDOWS_H = [4, 24, 72]
HORIZONS_H = [1, 2, 4, 8, 12, 24, 48, 72, 168]
TARGETS = ["ret", "absret", "rvol", "fvol"]
CURVE_W = 24   # window used for the headline ΔR²-vs-H curve


# ============================================================ target caches
def build_coin_caches(coin: str) -> dict:
    """Dense (gap-free) hourly arrays for forward targets: close, cum Σ logret², cum Σ vol.
    Indexed so that for a bar at integer-hour h the forward window (t, t+H] uses hours
    h+1..h+H. Missing hours are forward-filled (close) / zero-filled (vol)."""
    price = pd.read_parquet(PRICES / f"{coin}_1h.parquet")
    price = price[~price.index.duplicated(keep="last")].sort_index()
    price.index = pd.to_datetime(price.index, utc=True)
    ph = (price.index.view("int64") // 10**9 // HOUR).astype(int)
    close = pd.Series(price["close"].values, index=ph)
    vol = pd.Series(price["volume"].values, index=ph)
    close = close[~close.index.duplicated(keep="last")]
    vol = vol[~vol.index.duplicated(keep="last")]

    hmin, hmax = int(close.index.min()), int(close.index.max())
    full = np.arange(hmin, hmax + 1)
    close_d = close.reindex(full).ffill().bfill()
    vol_d = vol.reindex(full).fillna(0.0)

    logc = np.log(close_d.values)
    r = np.diff(logc, prepend=logc[0])
    r2 = r ** 2
    cum_r2 = np.concatenate([[0.0], np.cumsum(r2)])      # cum_r2[j] = Σ_{k<j} r2[k]
    cum_vol = np.concatenate([[0.0], np.cumsum(vol_d.values)])
    # close indexed by hour for ret target (sparse reindex is fine — NaN -> dropped)
    return {"close": close, "hmin": hmin, "hmax": hmax,
            "cum_r2": cum_r2, "cum_vol": cum_vol}


def attach_targets(df: pd.DataFrame, caches: dict, H: int) -> pd.DataFrame:
    """Attach all target families for horizon H, plus a unified 'fwd_ret' = the chosen
    target column is set later by the caller. Here we compute every family."""
    d = df.copy()
    coin = d["coin"].iloc[0]
    c = caches[coin]
    bars_h = d["h"].values.astype(int)

    # signed / abs return from sparse close
    close = c["close"]
    c_t = close.reindex(bars_h).values
    c_tH = close.reindex(bars_h + H).values
    ret = c_tH / c_t - 1.0
    d["tgt_ret"] = ret
    d["tgt_absret"] = np.abs(ret)

    # realized vol & forward volume from dense cumsums over (t, t+H]
    hmin, hmax = c["hmin"], c["hmax"]
    i = bars_h - hmin                       # index of t in dense grid
    j = i + H                               # index of t+H
    valid = (i >= 0) & (j <= hmax - hmin)
    cum_r2, cum_vol = c["cum_r2"], c["cum_vol"]
    rvol = np.full(len(bars_h), np.nan)
    fvol = np.full(len(bars_h), np.nan)
    iv, jv = i[valid], j[valid]
    rvol[valid] = np.sqrt(np.clip(cum_r2[jv + 1] - cum_r2[iv + 1], 0, None))
    fvol[valid] = np.log1p(np.clip(cum_vol[jv + 1] - cum_vol[iv + 1], 0, None))
    d["tgt_rvol"] = rvol
    d["tgt_fvol"] = fvol
    return d


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=COINS)
    ap.add_argument("--windows", nargs="+", type=int, default=WINDOWS_H)
    ap.add_argument("--horizons", nargs="+", type=int, default=HORIZONS_H)
    ap.add_argument("--targets", nargs="+", default=TARGETS)
    ap.add_argument("--min-items", type=int, default=m46.MIN_ITEMS)
    ap.add_argument("--min-posts", type=int, default=3)
    ap.add_argument("--min-bars", type=int, default=500)
    ap.add_argument("--no-overlap-only", action="store_true",
                    help="headline only (skip the overlapping Driscoll-Kraay robustness)")
    ap.add_argument("--overlap-only", action="store_true",
                    help="robustness only (overlapping Driscoll-Kraay; skip the non-overlap "
                         "headline). Use with --out-suffix to chunk by window into separate "
                         "files that survive the background wall-clock cap.")
    ap.add_argument("--overlap-step", type=int, default=1,
                    help="bar step (h) for the overlapping DK grid. 1 = full dense grid "
                         "(expensive at long H); >1 thins the grid so n and DK maxlags "
                         "(=ceil(H/step)) shrink, keeping long-H cells tractable while still "
                         "overlapping. step=4 ≈ 16x cheaper on the H=168 cell.")
    ap.add_argument("--out-suffix", default="",
                    help="suffix appended to all output filenames so parallel chunked jobs "
                         "do not clobber each other (e.g. '_overlap_W4').")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid for a fast end-to-end check")
    ap.add_argument("--substantive-only", action="store_true",
                    help="drop the four noise subtypes (share_3.x) from the content block, so the "
                         "scan uses the L=30 substantive subtypes consistent with the rest of the paper.")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.coins = ["BTC", "ETH"]
        args.windows = [24]
        args.horizons = [4, 24]
        args.targets = ["ret", "fvol"]

    # regime controls from the 4h panel (trailing merge_asof per coin)
    panel = pd.read_parquet(PANEL, columns=["ts", "coin", "ms_regime_prob_high", "vol_regime"])
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    panel["vol_ord"] = panel["vol_regime"].cat.codes.astype(float)

    caches = {coin: build_coin_caches(coin) for coin in args.coins}

    layer_rows, pic_rows, curve_rows = [], [], []
    summary = ["# Horizon scan — leak-free social-media decomposition (Stage 1)", "",
               "Per-post topic classifier (script 45) → rolling [t-W,t) features; targets "
               "forward [t,t+H). Five layers over controls (coin FE, regime, vol, trailing "
               "return), reusing script 46 verbatim. Headline = non-overlapping bars "
               "(cluster SE); robustness = overlapping 1h grid (Driscoll-Kraay HAC). "
               "Targets: ret (signed return), absret (|return|), rvol (realized vol), "
               "fvol (log forward volume).", ""]

    if args.overlap_only:
        overlaps = [True]
    elif args.no_overlap_only:
        overlaps = [False]
    else:
        overlaps = [False, True]

    for W in args.windows:
        # build dense bars per coin once per W (features only; targets attached per H below)
        per_coin = {}
        for coin in args.coins:
            d = m46.build_bars(coin, W)
            layers = d.attrs["layers"]
            pc = panel[panel["coin"] == coin].sort_values("ts")
            d = pd.merge_asof(d.sort_values("ts"),
                              pc[["ts", "ms_regime_prob_high", "vol_ord"]],
                              on="ts", direction="backward")
            d.attrs["layers"] = layers
            per_coin[coin] = d
        layers = per_coin[args.coins[0]].attrs["layers"]
        if args.substantive_only:
            _noise = {"share_3.1", "share_3.2", "share_3.3", "share_3.4"}
            layers = {**layers, "content": [c for c in layers["content"] if c not in _noise]}

        for H in args.horizons:
            for overlap in overlaps:
                # non-overlap headline steps by H; overlap robustness uses a (thinned)
                # dense grid of step = --overlap-step (default 1). Thinning cuts both n and
                # the DK maxlags (=ceil(H/step)) so long-H cells stay tractable while the grid
                # remains genuinely overlapping (adjacent windows share data whenever step<H).
                step_h = H if not overlap else max(1, args.overlap_step)
                # assemble pooled frame with all target families for this (W,H)
                parts = []
                for coin in args.coins:
                    d = attach_targets(per_coin[coin], caches, H)
                    if not overlap:
                        d = d[(d["h"] - d["h"].min()) % H == 0]
                    elif step_h > 1:
                        d = d[(d["h"] - d["h"].min()) % step_h == 0]
                    parts.append(d)
                dd0 = pd.concat(parts, ignore_index=True)
                # common feature/availability guard (target-independent)
                dd0 = dd0[(dd0["n_items"] >= args.min_items)
                          & (dd0["n_posts"] >= args.min_posts)
                          & dd0["trail_ret"].notna()
                          & dd0["ms_regime_prob_high"].notna()].copy()

                for target in args.targets:
                    tcol = f"tgt_{target}"
                    dd = dd0[dd0[tcol].notna()].copy()
                    if len(dd) < args.min_bars:
                        continue
                    # winsorise target, route into the 'fwd_ret' name decompose expects
                    lo, hi = dd[tcol].quantile([WINSOR, 1 - WINSOR])
                    dd["fwd_ret"] = dd[tcol].clip(lo, hi)
                    zcols = (layers["attention"] + layers["polarity"] + layers["content"]
                             + layers["dynamics"] + layers["breadth"] + ["trail_ret"])
                    dd = m46.zscore_within_coin(dd, zcols)

                    rows, blocks = m46.decompose(dd, layers, overlap, step_h, H)
                    tag = f"{target}_W{W}h_H{H}h_{'overlap' if overlap else 'nonoverlap'}"
                    for r in rows:
                        r.update({"tag": tag, "target": target, "W": W, "H": H,
                                  "overlap": overlap})
                        bp = blocks.get(r["layer"], (None, None))
                        r["block_wald_stat"] = bp[0]
                        r["block_wald_p"] = bp[1]
                        layer_rows.append(r)

                    pic = m46.partial_ic(dd, layers["content"], overlap, step_h, H)
                    pic["tag"] = tag; pic["target"] = target; pic["W"] = W
                    pic["H"] = H; pic["overlap"] = overlap
                    pic_rows.append(pic)
                    nsig = int(pic["sig_fdr_5pct"].sum())

                    cwald = blocks.get("content", (np.nan, np.nan))
                    dwald = blocks.get("dynamics", (np.nan, np.nan))
                    print(f"{tag}: n={len(dd):,}  content Wald p={cwald[1]:.3g}  "
                          f"dynamics Wald p={dwald[1]:.3g}  #sig/34={nsig}")

                    # headline curve: non-overlap cells at the chosen window
                    if (not overlap) and W == CURVE_W:
                        dr = {r["layer"]: r["delta_r2"] for r in rows}
                        curve_rows.append({
                            "target": target, "W": W, "H": H, "n": int(rows[0]["n"]),
                            "dR2_attention": dr.get("attention"),
                            "dR2_polarity": dr.get("polarity"),
                            "dR2_content": dr.get("content"),
                            "dR2_dynamics": dr.get("dynamics"),
                            "dR2_breadth": dr.get("breadth"),
                            "content_wald_p": cwald[1], "dynamics_wald_p": dwald[1],
                            "content_nsig_fdr": nsig})

                    summary += [f"## {tag}  (n={len(dd):,})", "",
                                "| layer | ΔR² | block Wald p |", "|---|---|---|"]
                    for r in rows:
                        bp = blocks.get(r["layer"], (None, None))[1]
                        bp = f"{bp:.3g}" if bp is not None else "—"
                        summary.append(f"| {r['layer']} | {r['delta_r2']:+.6f} | {bp} |")
                    summary += ["", f"content: **{nsig}/34** subtypes survive BH-FDR 5%.", ""]

    sfx = args.out_suffix
    pd.DataFrame(layer_rows).to_csv(OUT / f"horizon_scan_layers{sfx}.csv", index=False)
    if pic_rows:
        pd.concat(pic_rows, ignore_index=True).to_csv(
            OUT / f"horizon_scan_partial_ic{sfx}.csv", index=False)
    curve = pd.DataFrame(curve_rows)
    if not curve.empty:
        curve = curve.sort_values(["target", "H"]).reset_index(drop=True)
        curve.to_csv(OUT / f"horizon_scan_content_curve{sfx}.csv", index=False)
        # append the headline curve to the summary
        summary += ["", f"# Headline ΔR² curve (non-overlap, W={CURVE_W}h)", ""]
        for target in args.targets:
            sub = curve[curve["target"] == target]
            if sub.empty:
                continue
            summary += [f"## target = {target}", "",
                        "| H (h) | n | ΔR² att | ΔR² pol | ΔR² content | ΔR² dyn | "
                        "content Wald p | #sig/34 |",
                        "|---|---|---|---|---|---|---|---|"]
            for _, r in sub.iterrows():
                summary.append(
                    f"| {int(r['H'])} | {int(r['n'])} | {r['dR2_attention']:+.5f} | "
                    f"{r['dR2_polarity']:+.5f} | {r['dR2_content']:+.5f} | "
                    f"{r['dR2_dynamics']:+.5f} | {r['content_wald_p']:.3g} | "
                    f"{int(r['content_nsig_fdr'])} |")
            summary += [""]

    (OUT / f"horizon_scan_summary{sfx}.md").write_text("\n".join(summary))
    print(f"\n✓ layers   -> {OUT/f'horizon_scan_layers{sfx}.csv'}")
    print(f"✓ partial  -> {OUT/f'horizon_scan_partial_ic{sfx}.csv'}")
    print(f"✓ curve    -> {OUT/f'horizon_scan_content_curve{sfx}.csv'}")
    print(f"✓ summary  -> {OUT/f'horizon_scan_summary{sfx}.md'}")


if __name__ == "__main__":
    main()
