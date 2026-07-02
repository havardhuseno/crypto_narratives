#!/usr/bin/env python3
"""
58_conditional_directional.py
=============================
STAGE 3.7 — can narrative content predict the DIRECTION (sign) of returns CONDITIONAL on market
characteristics? The unconditional content→return effect is null (Stage 1; multivariate TOTAL in
script 55, p≈0.20). But every return test so far either (a) entered market state only as ADDITIVE
controls, or (b) collapsed content to the 1-D frozen scalar — which script 55 proved is lossy.
A state-dependent directional signal (content moves price differently in bull vs bear, calm vs
turbulent, rising vs falling market) would be invisible to both. The frequentist companion already
hinted at it: content×ambient-direction `Xs·Z = +0.058, p=0.013`.

So this script tests content × MARKET-CHARACTERISTIC interactions on signed return, multivariately,
with two pillars:

  PILLAR 1 — in-sample interaction decomposition (exploratory, FDR-controlled):
    Yret ~ controls + attention + polarity + content(30) [main]
              + content(30)×Z + content(30)×regime + content(30)×vol
    Each interaction block joint-Wald tested (cluster SE on bar timestamp), per-subtype BH-FDR,
    incremental ΔR². Shows whether/where conditional directional structure lives.

  PILLAR 2 — OUT-OF-SAMPLE directional validation (decisive, leak-free):
    Train on pre-2023, test on post-2023 (production boundary). Standardisation frozen on train.
    RidgeCV fit on train; predict test. Nested models compared on TEST:
      (i)  controls + attention + polarity         (no content)
      (ii) + content main effects                  (unconditional content)
      (iii)+ content × {Z, regime, vol}            (state-conditional content)
    Report directional IC (Spearman pred vs realised), sign hit-rate, and a long–short return.
    If (iii) beats (ii)≈(i) OUT-OF-SAMPLE, the state-dependent directional signal is real, not an
    interaction overfit. Content features use only [t−W,t); the topic classifier was train-time
    separated — so this is leak-free.

Moderators ("market characteristics"): Z = trailing-168h BTC ambient direction, ms_regime_prob_high
(HMM bull probability), vol_ord (volatility regime). Content = 30 subtypes (noise 3.x dropped).

Outputs (all under paper_narrative/, never touches production):
  outputs/cond_dir_insample.csv     per-subtype interaction coefs + FDR per moderator block
  outputs/cond_dir_oos.csv          nested-model OOS directional IC / hit-rate / long-short
  outputs/cond_dir_summary.md       write-up + verdict
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
assemble = m53.assemble
COINS = m53.COINS
SUB30 = m55.SUB30
bh_fdr = m55.bh_fdr
zsc = m55.zsc

MODERATORS = ["Z", "ms_regime_prob_high", "vol_ord"]   # ambient direction, HMM regime, vol regime
MOD_LABEL = {"Z": "ambient_dir", "ms_regime_prob_high": "regime", "vol_ord": "vol"}
POLARITY = ["net_sent", "intent_buy_share", "intent_sell_share",
            "intent_fomo_share", "intent_fear_share"]
ATTENTION = ["att_log_posts", "att_log_items"]


def build_blocks(df):
    """Return standardised design pieces: content (30), attention, polarity, moderators, controls,
    coin dummies; plus the standardised signed-return target."""
    def Z(cols):
        return np.column_stack([zsc(df[c].values) for c in cols])
    content = Z([f"share_{l}" for l in SUB30])
    att = Z(ATTENTION)
    pol = Z([c for c in POLARITY if c in df.columns])
    mods = {m: zsc(df[m].values) for m in MODERATORS}
    ctrl = Z(["trail_ret"])
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float).values
    y = zsc(df["Yret"].values)
    return content, att, pol, mods, ctrl, coin_d, y


def interactions(content, mods):
    """content × each moderator → dict block_name -> (n, 30) interaction matrix."""
    return {MOD_LABEL[m]: content * mods[m][:, None] for m in MODERATORS}


# ============================================================ Pillar 1: in-sample
def insample_decomposition(df, log):
    import statsmodels.api as sm
    content, att, pol, mods, ctrl, coin_d, y = build_blocks(df)
    inter = interactions(content, mods)
    mod_main = np.column_stack([mods[m] for m in MODERATORS])

    base = np.column_stack([coin_d, ctrl, mod_main, att, pol, content])  # controls+main(content too)
    blocks = list(inter.items())
    Xfull = np.column_stack([base] + [b for _, b in blocks])
    Xc = sm.add_constant(Xfull, has_constant="add")
    res = sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": df["h"].values})

    # locate each interaction block's columns in Xc (after const + base)
    off = 1 + base.shape[1]
    rows = []
    summary = []
    for name, b in blocks:
        idx = np.arange(off, off + b.shape[1]); off += b.shape[1]
        R = np.zeros((len(idx), Xc.shape[1]))
        for r, j in enumerate(idx):
            R[r, j] = 1.0
        jp = float(np.asarray(res.f_test(R).pvalue).ravel()[0])
        coefs = res.params[idx]; pvals = res.pvalues[idx]; fdr = bh_fdr(pvals)
        nsig = int((fdr < 0.05).sum())
        summary.append((name, jp, nsig))
        for k, sub in enumerate(SUB30):
            rows.append({"moderator": name, "subtype": sub, "coef": coefs[k],
                         "p": pvals[k], "fdr": fdr[k], "sig_fdr_5pct": bool(fdr[k] < 0.05)})
        log(f"  [insample] content×{name}: joint p={jp:.3g}  #sig_FDR={nsig}/30")
    return pd.DataFrame(rows), summary


# ============================================================ Pillar 2: out-of-sample
def oos_directional(df, log, seed):
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    tr = df["is_train"].values
    te = ~tr
    if te.sum() < 200 or tr.sum() < 200:
        log("  [oos] insufficient train/test rows; skipping"); return pd.DataFrame()

    # raw (unstandardised) pieces; standardise with TRAIN stats only
    content_raw = df[[f"share_{l}" for l in SUB30]].values
    att_raw = df[ATTENTION].values
    pol_raw = df[[c for c in POLARITY if c in df.columns]].values
    mod_raw = df[MODERATORS].values
    ctrl_raw = df[["trail_ret"]].values
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float).values
    y = df["Yret"].values

    def scale(train_block, full_block):
        sc = StandardScaler().fit(train_block)
        return sc.transform(full_block)

    C = scale(content_raw[tr], content_raw)
    A = scale(att_raw[tr], att_raw)
    P = scale(pol_raw[tr], pol_raw)
    Mm = scale(mod_raw[tr], mod_raw)
    Ct = scale(ctrl_raw[tr], ctrl_raw)
    inter = np.column_stack([C * Mm[:, j:j + 1] for j in range(Mm.shape[1])])  # content×each mod

    designs = {
        "i_controls": np.column_stack([coin_d, Ct, Mm, A, P]),
        "ii_content_main": np.column_stack([coin_d, Ct, Mm, A, P, C]),
        "iii_content_x_market": np.column_stack([coin_d, Ct, Mm, A, P, C, inter]),
    }
    alphas = np.logspace(-2, 4, 25)
    yt = y[tr]
    rows = []
    for name, X in designs.items():
        mdl = RidgeCV(alphas=alphas).fit(X[tr], yt)
        pred = mdl.predict(X[te])
        yte = y[te]
        ic = float(pd.Series(pred).corr(pd.Series(yte), method="spearman"))
        hit = float(((pred > 0) == (yte > 0)).mean())
        # long-short: sign(pred) * realised return, mean per bar (already ~SD-scaled returns)
        ls = float(np.mean(np.sign(pred) * yte))
        rows.append({"model": name, "alpha": float(mdl.alpha_), "n_test": int(te.sum()),
                     "dir_IC_spearman": ic, "sign_hit_rate": hit, "long_short_mean": ls})
        log(f"  [oos] {name:22s}: dir-IC={ic:+.4f}  hit={hit:.4f}  L/S={ls:+.5f}  (α={mdl.alpha_:.3g})")
    return pd.DataFrame(rows)


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = print

    if args.smoke:
        args.coins = ["BTC", "ETH", "SOL"]

    log(f"assembling W={args.W}h m={args.m}h H={args.H}h coins={args.coins}")
    df = assemble(args.coins, args.W, args.m, args.H, args.min_items, args.min_posts, args.z_win)
    log(f"  pooled bars n={len(df):,}  (train {int(df['is_train'].sum()):,} / "
        f"test {int((~df['is_train']).sum()):,})")

    log("PILLAR 1 — in-sample content×market interaction decomposition:")
    insamp, summ = insample_decomposition(df, log)
    insamp.to_csv(OUT / "cond_dir_insample.csv", index=False)

    log("PILLAR 2 — out-of-sample directional validation (nested models):")
    oos = oos_directional(df, log, args.seed)
    if not oos.empty:
        oos.to_csv(OUT / "cond_dir_oos.csv", index=False)

    # ---- summary ----
    md = ["# Stage 3.7 — conditional / state-dependent directional return decomposition", "",
          f"Cell W={args.W}h → outcome over [t,t+H), H={args.H}h. Target = signed forward return. "
          "Content = 30 subtypes (noise 3.x dropped). Market characteristics: ambient direction Z "
          f"(trailing {args.z_win}h BTC return), HMM regime, vol regime. Pooled n={len(df):,} "
          f"(train {int(df['is_train'].sum()):,} / test {int((~df['is_train']).sum()):,}).", "",
          "## Pillar 1 — in-sample interaction blocks (cluster SE, BH-FDR per block)", "",
          "| content × moderator | joint p | #sig FDR / 30 |", "|---|---|---|"]
    for name, jp, nsig in summ:
        md.append(f"| content×{name} | {jp:.3g} | {nsig} |")
    md += ["", "Per-subtype interaction coefficients + FDR in `cond_dir_insample.csv`.", ""]
    if not oos.empty:
        md += ["## Pillar 2 — OUT-OF-SAMPLE directional validation (the decisive test)", "",
               "```", oos.to_string(index=False), "```", "",
               "Nested models trained pre-2023, tested post-2023 (standardisation frozen on train). "
               "dir-IC = Spearman(prediction, realised return) on test; hit = sign accuracy; "
               "L/S = mean of sign(pred)·realised. **The signal is real iff "
               "`iii_content_x_market` beats `ii_content_main` ≈ `i_controls` out-of-sample.** "
               "Equal or worse OOS-IC ⇒ any in-sample interaction significance is overfitting, and "
               "the directional null stands even conditional on market state."]
    (OUT / "cond_dir_summary.md").write_text("\n".join(md))
    log(f"\n✓ insample -> {OUT/'cond_dir_insample.csv'}")
    log(f"✓ oos      -> {OUT/'cond_dir_oos.csv'}")
    log(f"✓ summary  -> {OUT/'cond_dir_summary.md'}")


if __name__ == "__main__":
    main()
