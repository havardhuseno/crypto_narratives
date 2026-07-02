#!/usr/bin/env python3
"""
44_narrative_decomposition.py
=============================
Paper-grade decomposition of the Reddit social-media return signal in crypto into
three layers — ATTENTION, POLARITY (sentiment), and NARRATIVE CONTENT (34 codebook
subtypes) — answering the QF paper question: *does narrative content add
predictability beyond attention + polarity?*

Upgrades over the first-pass `43_taxonomy_ic_diagnostic.py` (which was univariate,
had no controls, no inference, and a likely look-ahead leak):

  (1) LEAK-FREE features. Daily-derived Reddit features (content shares, polarity,
      attention) are consumed with an explicit one-UTC-day trailing lag: the value
      used at a 4h decision bar on day d is the aggregate as of the END of day d-1.
      We report BOTH "as-stored" (panel as-is) and "lag1d" so the conclusion is
      shown robust to the lag choice (this IS the paper's leak-control pillar,
      demonstrated on our own prior leak).

  (2) DECOMPOSITION. Hierarchical OLS of fwd_ret_4h:
        L0 coin-FE (+ regime)  ->  L1 +attention  ->  L2 +polarity  ->  L3 +content
      Incremental R² / adj-R² and a cluster-robust joint Wald test on the 34-subtype
      block are the headline decomposition result.

  (3) INFERENCE. SEs clustered by timestamp (absorbs market-wide cross-sectional
      dependence; 8 coins is too few to cluster on). Forward returns are
      NON-overlapping (4h returns on 4h bars) so no overlap-HAC is needed.
      Per-subtype conditional coefficients get Benjamini-Hochberg FDR across the 34
      tests. Complementary rank-based partial IC (Spearman of residuals) reported too.

  (4) REGIME CONDITIONING. The content-block joint test and per-subtype partial IC
      are re-run within high- vs low-regime subsamples (ms_regime_prob_high), the one
      place a conditional content effect could hide.

Inputs : data/panel/full_panel_4h_v11.parquet
Outputs: outputs/diagnostics/narrative_decomposition_layers.csv
         outputs/diagnostics/narrative_decomposition_partial_ic.csv
         outputs/diagnostics/narrative_decomposition_summary.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

# paper_narrative/scripts/44_*.py -> parents[1]=paper_narrative, parents[2]=repo root.
# READ from the main pipeline (read-only); WRITE only inside paper_narrative/.
PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL = PROJECT_ROOT / "data" / "panel" / "full_panel_4h_v11.parquet"  # read-only pipeline data
OUT = PAPER_ROOT / "outputs"

COINS = ["ADA", "BTC", "DOGE", "ETH", "LINK", "LTC", "SOL", "XRP"]
BAR = pd.Timedelta(hours=4)
WINSOR = 0.005  # winsorize forward returns at 0.5% / 99.5% for regression robustness

CB_NAMES = {
    "narr_cb_3_1": "3.1 Structural noise", "narr_cb_3_2": "3.2 Bot noise",
    "narr_cb_3_3": "3.3 Adversarial noise", "narr_cb_3_4": "3.4 Off-topic substantive",
    "narr_cb_4_1": "4.1 Retail wallet/exchange UX", "narr_cb_4_2": "4.2 Staking-delegator",
    "narr_cb_4_3": "4.3 Staking-operator", "narr_cb_4_4": "4.4 Airdrop/snapshot prep",
    "narr_cb_4_5": "4.5 DeFi yield farming", "narr_cb_4_6": "4.6 DeFi arbitrage",
    "narr_cb_4_7": "4.7 DeFi DEX selection", "narr_cb_4_8": "4.8 Memecoin trading culture",
    "narr_cb_4_9": "4.9 Payments/Lightning", "narr_cb_4_10": "4.10 Governance participation",
    "narr_cb_5_1": "5.1 Regulatory/litigation", "narr_cb_5_2": "5.2 Adoption-institutional",
    "narr_cb_5_3": "5.3 Adoption-crypto B2B", "narr_cb_5_4": "5.4 Adoption-RWA",
    "narr_cb_5_5": "5.5 Market cycle/macro", "narr_cb_5_6": "5.6 Macro/Fed reactions",
    "narr_cb_5_7": "5.7 Catalyst anticipation", "narr_cb_5_8": "5.8 Protocol upgrade",
    "narr_cb_5_9": "5.9 Long-term value thesis", "narr_cb_5_10": "5.10 Price predictions",
    "narr_cb_5_11": "5.11 Sector market state", "narr_cb_5_12": "5.12 Community/emotional",
    "narr_cb_5_13": "5.13 Project governance scrutiny", "narr_cb_5_14": "5.14 Cross-ecosystem promo",
    "narr_cb_5_15": "5.15 L1 dominance debate", "narr_cb_5_16": "5.16 L2 positioning",
    "narr_cb_5_17": "5.17 Hacks/exploits", "narr_cb_5_18": "5.18 BTC mining economics",
    "narr_cb_5_19": "5.19 Self-custody vs CEX", "narr_cb_5_20": "5.20 Tactical market analysis",
}
CONTENT = list(CB_NAMES.keys())


# ----------------------------------------------------------------------------- helpers
def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(PANEL)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df = df[df["coin"].isin(COINS)].sort_values(["coin", "ts"]).reset_index(drop=True)
    return df


def add_fwd_ret(df: pd.DataFrame) -> pd.DataFrame:
    """fwd_ret_4h = close[t+1]/close[t]-1 per coin; masked where next bar gap != 4h."""
    out = []
    for c, g in df.groupby("coin", sort=False):
        g = g.sort_values("ts").copy()
        nxt_close = g["close"].shift(-1)
        nxt_ts = g["ts"].shift(-1)
        gap_ok = (nxt_ts - g["ts"]) == BAR
        r = nxt_close / g["close"] - 1.0
        g["fwd_ret_4h"] = np.where(gap_ok, r, np.nan)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def lag_one_day(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Leak-free: value at a bar on UTC day d := aggregate as of END of day d-1.
    Daily features are (near-)constant within a day, so 'last value of prior day'
    is the right trailing snapshot. Returns df with `<col>__lag1d` columns."""
    df = df.copy()
    df["_date"] = df["ts"].dt.floor("D")
    # last observation per (coin, date)
    daily = (df.sort_values("ts")
               .groupby(["coin", "_date"])[cols].last()
               .reset_index())
    daily = daily.sort_values(["coin", "_date"])
    for col in cols:
        daily[f"{col}__lag1d"] = daily.groupby("coin")[col].shift(1)
    lagcols = [f"{c}__lag1d" for c in cols]
    df = df.merge(daily[["coin", "_date"] + lagcols], on=["coin", "_date"], how="left")
    return df.drop(columns="_date")


def zscore_within_coin(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        g = df.groupby("coin")[c]
        df[c] = ((df[c] - g.transform("mean")) / g.transform("std")).fillna(0.0)
    return df


def fit_cluster(y: pd.Series, X: pd.DataFrame, groups: pd.Series):
    Xc = sm.add_constant(X, has_constant="add")
    return sm.OLS(y.values, Xc.values).fit(
        cov_type="cluster", cov_kwds={"groups": groups.values}), Xc.columns.tolist()


def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


# ----------------------------------------------------------------------------- core
def decomposition(df: pd.DataFrame, content_cols: list[str], tag: str) -> dict:
    """Hierarchical L0->L3 regression. Returns layer R² table + content joint Wald."""
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float)
    regime = df[["ms_regime_prob_high"]].fillna(df["ms_regime_prob_high"].median())
    att = df[["att_log"]]
    pol = df[["net_sent", "has_sent"]]
    con = df[content_cols]
    y = df["fwd_ret_4h"]
    grp = df["ts"]

    layers = {
        "L0 coin-FE + regime": pd.concat([coin_d, regime], axis=1),
        "L1 +attention": pd.concat([coin_d, regime, att], axis=1),
        "L2 +polarity": pd.concat([coin_d, regime, att, pol], axis=1),
        "L3 +content(34)": pd.concat([coin_d, regime, att, pol, con], axis=1),
    }
    rows, prev_r2 = [], 0.0
    res_full = names_full = None
    for name, X in layers.items():
        res, names = fit_cluster(y, X, grp)
        r2 = res.rsquared
        rows.append({"layer": name, "n": int(res.nobs), "r2": r2,
                     "adj_r2": res.rsquared_adj, "delta_r2": r2 - prev_r2})
        prev_r2 = r2
        if name.startswith("L3"):
            res_full, names_full = res, names

    # Joint Wald test that all 34 content coefficients == 0 (cluster-robust)
    idx = [names_full.index(c) for c in content_cols]
    R = np.zeros((len(idx), len(names_full)))
    for i, j in enumerate(idx):
        R[i, j] = 1.0
    wald = res_full.wald_test(R, scalar=True)
    return {"tag": tag, "layers": rows,
            "content_wald_F": float(wald.statistic), "content_wald_p": float(wald.pvalue),
            "full_res": res_full, "full_names": names_full, "content_idx": idx}


def partial_ic(df: pd.DataFrame, content_cols: list[str]) -> pd.DataFrame:
    """Per-subtype conditional coef (cluster-robust, BH-FDR) + rank partial IC.
    Partial IC = Spearman(resid(y|controls), resid(subtype|controls))."""
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float)
    regime = df[["ms_regime_prob_high"]].fillna(df["ms_regime_prob_high"].median())
    ctrl = pd.concat([coin_d, regime, df[["att_log", "net_sent", "has_sent"]]], axis=1)
    grp = df["ts"]
    y = df["fwd_ret_4h"]

    # (a) one multivariate regression: each subtype's conditional standardized beta
    Xall = pd.concat([ctrl, df[content_cols]], axis=1)
    res, names = fit_cluster(y, Xall, grp)
    coefs = dict(zip(names, res.params))
    ses = dict(zip(names, res.bse))
    pvs = dict(zip(names, res.pvalues))

    # (b) rank partial IC via residualization on controls
    Xc = sm.add_constant(ctrl, has_constant="add").values
    y_resid = y.values - Xc @ np.linalg.lstsq(Xc, y.values, rcond=None)[0]

    rows = []
    for c in content_cols:
        xc_resid = df[c].values - Xc @ np.linalg.lstsq(Xc, df[c].values, rcond=None)[0]
        rho, _ = spearmanr(xc_resid, y_resid)
        rows.append({"feature": c, "name": CB_NAMES[c],
                     "beta_std": coefs.get(c, np.nan), "se": ses.get(c, np.nan),
                     "p_raw": pvs.get(c, np.nan), "partial_ic": float(rho)})
    out = pd.DataFrame(rows)
    out["p_fdr"] = bh_fdr(out["p_raw"].values)
    out["sig_fdr_5pct"] = out["p_fdr"] < 0.05
    return out.sort_values("p_raw").reset_index(drop=True)


# ----------------------------------------------------------------------------- main
def prep(df: pd.DataFrame, content_cols: list[str]) -> pd.DataFrame:
    d = df.copy()
    d["att_log"] = np.log1p(d["n_total"].clip(lower=0))
    d["has_sent"] = d["net_sentiment"].notna().astype(float)
    d["net_sent"] = d["net_sentiment"].fillna(0.0)
    d = d.dropna(subset=["fwd_ret_4h"]).copy()
    # winsorize returns
    lo, hi = d["fwd_ret_4h"].quantile([WINSOR, 1 - WINSOR])
    d["fwd_ret_4h"] = d["fwd_ret_4h"].clip(lo, hi)
    # z-score continuous regressors within coin (content shares + attention + polarity)
    d = zscore_within_coin(d, content_cols + ["att_log", "net_sent"])
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Loading panel …")
    df = load_panel()
    df = add_fwd_ret(df)
    print(f"  rows={len(df):,}  coins={df['coin'].nunique()}  "
          f"valid fwd_ret={df['fwd_ret_4h'].notna().mean():.1%}")

    # leak-free lag1d versions of all daily-derived features
    daily_feats = CONTENT + ["net_sentiment", "n_total"]
    df = lag_one_day(df, daily_feats)

    variants = {
        "as_stored": {"content": CONTENT, "nt": "n_total", "ns": "net_sentiment"},
        "lag1d": {"content": [f"{c}__lag1d" for c in CONTENT],
                  "nt": "n_total__lag1d", "ns": "net_sentiment__lag1d"},
    }

    all_layers, summary_md = [], []
    summary_md += ["# Narrative content decomposition (paper-grade)", "",
                   "Hierarchical OLS of non-overlapping fwd_ret_4h; SEs clustered by "
                   "timestamp; forward returns winsorized at 0.5/99.5%. Content = 34 "
                   "codebook subtypes (z-scored within coin). Reported for both the "
                   "panel-as-stored features and an explicit one-UTC-day trailing lag "
                   "(leak-control).", ""]

    partial_tables = {}
    for vtag, spec in variants.items():
        if vtag == "lag1d":
            # drop the as-stored originals, then promote lagged cols to canonical names
            d = df.drop(columns=CONTENT + ["n_total", "net_sentiment"]).rename(
                columns={**{f"{c}__lag1d": c for c in CONTENT},
                         "n_total__lag1d": "n_total",
                         "net_sentiment__lag1d": "net_sentiment"})
        else:
            d = df.copy()
        content_cols = CONTENT
        d = prep(d, content_cols)
        print(f"\n=== variant: {vtag}  (n={len(d):,}) ===")

        dec = decomposition(d, content_cols, vtag)
        for r in dec["layers"]:
            r["variant"] = vtag
            all_layers.append(r)
            print(f"  {r['layer']:<24} R²={r['r2']:.5f}  ΔR²={r['delta_r2']:+.5f}")
        print(f"  content-block joint Wald: F={dec['content_wald_F']:.2f} "
              f"p={dec['content_wald_p']:.4g}")

        pic = partial_ic(d, content_cols)
        partial_tables[vtag] = pic
        n_sig = int(pic["sig_fdr_5pct"].sum())
        print(f"  per-subtype: {n_sig}/34 significant at BH-FDR 5%")

        summary_md += [f"## Variant: `{vtag}`", "",
                       "| Layer | n | R² | adj-R² | ΔR² |",
                       "|---|---|---|---|---|"]
        for r in dec["layers"]:
            summary_md.append(f"| {r['layer']} | {r['n']:,} | {r['r2']:.5f} | "
                              f"{r['adj_r2']:.5f} | {r['delta_r2']:+.5f} |")
        summary_md += ["",
                       f"**Content block (34 subtypes) joint Wald:** "
                       f"F={dec['content_wald_F']:.2f}, p={dec['content_wald_p']:.4g}. "
                       f"**{n_sig}/34** subtypes significant after BH-FDR (5%).", "",
                       "Top 8 subtypes by |conditional standardized β| (FDR-adjusted p):", "",
                       "| Subtype | β (std) | partial IC | p_raw | p_FDR | sig |",
                       "|---|---|---|---|---|---|"]
        top = pic.reindex(pic["beta_std"].abs().sort_values(ascending=False).index).head(8)
        for r in top.itertuples():
            summary_md.append(f"| {r.name} | {r.beta_std:+.4f} | {r.partial_ic:+.4f} | "
                              f"{r.p_raw:.3g} | {r.p_fdr:.3g} | "
                              f"{'**yes**' if r.sig_fdr_5pct else 'no'} |")
        summary_md.append("")

    # regime split on the lag1d (leak-free) variant
    summary_md += ["## Regime conditioning (lag1d, leak-free)",
                   "Content-block joint Wald within high vs low market regime "
                   "(`ms_regime_prob_high` median split):", ""]
    d = df.drop(columns=CONTENT + ["n_total", "net_sentiment"]).rename(
        columns={**{f"{c}__lag1d": c for c in CONTENT},
                 "n_total__lag1d": "n_total",
                 "net_sentiment__lag1d": "net_sentiment"})
    d = prep(d, CONTENT)
    med = d["ms_regime_prob_high"].median()
    summary_md += ["| Regime | n | content Wald F | p | #sig FDR |", "|---|---|---|---|---|"]
    for label, mask in [("high (>med)", d["ms_regime_prob_high"] > med),
                        ("low (<=med)", d["ms_regime_prob_high"] <= med)]:
        sub = d[mask]
        dec = decomposition(sub, CONTENT, f"regime_{label}")
        pic = partial_ic(sub, CONTENT)
        summary_md.append(f"| {label} | {len(sub):,} | {dec['content_wald_F']:.2f} | "
                          f"{dec['content_wald_p']:.4g} | {int(pic['sig_fdr_5pct'].sum())} |")
    summary_md.append("")

    # write outputs
    pd.DataFrame(all_layers)[["variant", "layer", "n", "r2", "adj_r2", "delta_r2"]] \
        .to_csv(OUT / "narrative_decomposition_layers.csv", index=False)
    partial_tables["lag1d"].to_csv(OUT / "narrative_decomposition_partial_ic.csv", index=False)
    (OUT / "narrative_decomposition_summary.md").write_text("\n".join(summary_md))
    print(f"\n✓ layers   → {OUT/'narrative_decomposition_layers.csv'}")
    print(f"✓ partial IC → {OUT/'narrative_decomposition_partial_ic.csv'}")
    print(f"✓ summary  → {OUT/'narrative_decomposition_summary.md'}")


if __name__ == "__main__":
    main()
