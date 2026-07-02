#!/usr/bin/env python3
"""
55_multivariate_apath.py
========================
STAGE 3.5b — does the a-path null survive a MULTIVARIATE / per-subtype test?

Scripts 53/54 collapsed narrative content into a single frozen 1-D scalar Xs and found the
within-coin a-path (content→forward-volume M) ≈ 0. That could be lossy: a single global
projection can wash out a within-coin, subtype-specific channel. This script tests the a-path
*multivariately* — regressing the mediator M on the full content vector within-coin (controls +
coin FE partialled out, cross-fitted DML), with a joint block test, per-subtype coefficients +
BH-FDR, and propagation of any surviving subtype through the b-path (M→return) to a per-subtype
indirect effect a_k·b.

Per user decisions:
  • content granularity: (i) 30 SUBTYPES (drop the 4 noise classes 3.x) and (ii) MAIN FAMILIES
    (family 4, family 5 aggregated shares). Both reported.
  • moderator Z stays the price-based trailing-168h BTC return (other proxies tested later).
  • a small m×H SWEEP (m∈{2,4,8} activity windows × H∈{8,24,48} horizons) to show the joint
    a-path / b-path conclusion is not specific to the W24/m4/H24 cell.

Method: cross-fitted partial-linear DML (reuses script 54's crossfit machinery). The multivariate
a-path block test = cluster-robust Wald on the content coefficients in the cross-fit-residualised
OLS M_res ~ C_res; per-subtype coefs get BH-FDR. b-path = M_res→Yret_res slope. Indirect effect
per subtype = a_k·b; joint mediated index = (Σ_k a_k·c̄_k)·b is not meaningful, so we report the
per-subtype a_k·b and the joint a-block significance instead.

Outputs (all under paper_narrative/, never touches production):
  outputs/apath_multivariate_persubtype.csv   per-subtype a_k, p, FDR, a_k·b at the base cell
  outputs/apath_multivariate_sweep.csv         joint a-block p + ΔR² + b across the m×H sweep
  outputs/apath_multivariate_summary.md        write-up + verdict vs the 1-D scalar null
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


m54 = _load("ml54", "54_ml_mediation_triangulation.py")
m53 = m54.m53
assemble = m53.assemble
COINS = m53.COINS
CB_LABELS = m53.CB_LABELS
DYN_COLS = m53.DYN_COLS
crossfit_resid = m54.crossfit_resid
make_folds = m54.make_folds
zsc = m54.zsc

SUB30 = [l for l in CB_LABELS if not l.startswith("3.")]          # drop the 4 noise classes
FAMILIES = sorted({l.split(".")[0] for l in SUB30})               # ['4','5']


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def content_matrix(df, kind):
    """Return (feature_names, standardised content matrix) for kind in {'sub30','family'}."""
    if kind == "sub30":
        cols = [f"share_{l}" for l in SUB30]
        names = SUB30
    else:  # family-aggregated shares (sum of subtype shares within each main family)
        names, mats = [], []
        for fam in FAMILIES:
            sub = [f"share_{l}" for l in CB_LABELS if l.startswith(f"{fam}.")]
            mats.append(df[sub].sum(axis=1).values)
            names.append(f"fam{fam}")
        X = np.column_stack(mats)
        return names, np.column_stack([zsc(X[:, j]) for j in range(X.shape[1])])
    X = df[cols].values
    return names, np.column_stack([zsc(X[:, j]) for j in range(X.shape[1])])


def dml_multivariate_apath(df, names, C, k_folds, seed):
    """Cross-fitted multivariate a-path + b-path. Returns per-subtype residual-OLS results and the
    scalar b, from cached cross-fit residuals (cluster inference done by caller)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    import statsmodels.api as sm

    rng = np.random.default_rng(seed)
    n = len(df)
    folds = make_folds(n, k_folds, rng)

    def gbr():
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=50, l2_regularization=1.0, random_state=seed)

    M = zsc(df["M"].values)
    Yr = zsc(df["Yret"].values)
    Z = zsc(df["Z"].values)
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float).values
    # DEFAULT control: Reddit volume (log post & post+comment counts over [t-W,t)) so the a-path
    # measures the narrative-COMPOSITION effect on trading volume, net of how MUCH was posted.
    reddit_vol = np.column_stack([zsc(df["att_log_posts"].values), zsc(df["att_log_items"].values)])
    W = np.column_stack([
        zsc(df["ms_regime_prob_high"].values), zsc(df["vol_ord"].values),
        zsc(df["trail_ret"].values), reddit_vol, coin_d])
    WZ = np.column_stack([W, Z])

    # residualise M and each content col on controls W (cross-fitted) -> within-coin a-path
    M_rW, _ = crossfit_resid(M, W, folds, gbr)
    C_rW = np.column_stack([crossfit_resid(C[:, j], W, folds, gbr)[0] for j in range(C.shape[1])])

    # multivariate OLS M_rW ~ C_rW with cluster-robust SE (cluster on bar timestamp h)
    Xc = sm.add_constant(C_rW, has_constant="add")
    res = sm.OLS(M_rW, Xc).fit(cov_type="cluster", cov_kwds={"groups": df["h"].values})
    a_coef = res.params[1:]
    a_p = res.pvalues[1:]
    # joint Wald on the whole content block
    R = np.zeros((C.shape[1], Xc.shape[1])); R[:, 1:] = np.eye(C.shape[1])
    joint = res.f_test(R)
    joint_p = float(np.asarray(joint.pvalue).ravel()[0])
    # incremental R^2 of the content block over controls (both already residualised on W)
    sst = np.sum((M_rW - M_rW.mean()) ** 2)
    sse = np.sum(res.resid ** 2)
    dR2 = 1.0 - sse / sst

    # b-path (gross): M -> Yr residualised on (W,Z); scalar slope (NOT controlling content)
    Yr_rWZ, _ = crossfit_resid(Yr, WZ, folds, gbr)
    M_rWZ, _ = crossfit_resid(M, WZ, folds, gbr)
    b = float(np.dot(M_rWZ, Yr_rWZ) / np.dot(M_rWZ, M_rWZ))

    # ---- propagation diagnostics so a_k·b is interpretable against the Stage-1 return null ----
    # TOTAL content -> return (within-coin): joint DML, residualise on (W,Z). Expect ~null.
    C_rWZ = np.column_stack([crossfit_resid(C[:, j], WZ, folds, gbr)[0] for j in range(C.shape[1])])
    Xt = sm.add_constant(C_rWZ, has_constant="add")
    rt = sm.OLS(Yr_rWZ, Xt).fit(cov_type="cluster", cov_kwds={"groups": df["h"].values})
    Rt = np.zeros((C.shape[1], Xt.shape[1])); Rt[:, 1:] = np.eye(C.shape[1])
    total_p = float(np.asarray(rt.f_test(Rt).pvalue).ravel()[0])
    total_dR2 = 1.0 - np.sum(rt.resid ** 2) / np.sum((Yr_rWZ - Yr_rWZ.mean()) ** 2)

    # b-path CONTROLLING content: residualise M and Yr on (W,Z,content) -> return-relevant
    # volume slope net of the narrative-driven component. If << gross b, content-driven volume
    # is the part that does NOT transmit to return (direction-neutral, MDH).
    WZC = np.column_stack([WZ, C])
    Yr_rWZC, _ = crossfit_resid(Yr, WZC, folds, gbr)
    M_rWZC, _ = crossfit_resid(M, WZC, folds, gbr)
    b_ctrl = float(np.dot(M_rWZC, Yr_rWZC) / np.dot(M_rWZC, M_rWZC))

    fdr = bh_fdr(a_p)
    table = pd.DataFrame({"feature": names, "a_coef": a_coef, "a_p": a_p,
                          "a_fdr": fdr, "sig_fdr_5pct": fdr < 0.05,
                          "indirect_ak_b": a_coef * b, "indirect_ak_bctrl": a_coef * b_ctrl})
    return table, {"joint_p": joint_p, "dR2_content_on_M": float(dR2), "b": b, "b_ctrl": b_ctrl,
                   "total_ret_p": total_p, "total_ret_dR2": float(total_dR2),
                   "n": n, "n_features": C.shape[1]}


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
    ap.add_argument("--sweep-m", nargs="+", type=int, default=[2, 4, 8])
    ap.add_argument("--sweep-H", nargs="+", type=int, default=[8, 24, 48])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-base", action="store_true", help="skip the per-subtype base cell")
    ap.add_argument("--no-sweep", action="store_true", help="skip the m×H sweep")
    ap.add_argument("--out-suffix", default="", help="suffix for sweep/summary outputs so chunked "
                    "per-m sweep jobs (each surviving the wall-clock cap) don't clobber each other")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = print

    if args.smoke:
        args.coins = ["BTC", "ETH", "SOL"]
        args.folds = 3
        args.sweep_m, args.sweep_H = [4], [24]

    # ---------- base cell: per-subtype multivariate a-path (sub30 + families) ----------
    persub, base_meta = None, {}
    if not args.no_base:
        log(f"BASE CELL  W={args.W}h m={args.m}h H={args.H}h  coins={args.coins}")
        df0 = assemble(args.coins, args.W, args.m, args.H, args.min_items, args.min_posts, args.z_win)
        log(f"  pooled bars n={len(df0):,}")
        persub_rows = []
        for kind in ["sub30", "family"]:
            names, C = content_matrix(df0, kind)
            table, meta = dml_multivariate_apath(df0, names, C, args.folds, args.seed)
            table.insert(0, "granularity", kind)
            persub_rows.append(table)
            base_meta[kind] = meta
            nsig = int(table["sig_fdr_5pct"].sum())
            log(f"  [{kind}] {meta['n_features']} feats: joint a-block p={meta['joint_p']:.3g}  "
                f"ΔR²(content→M)={meta['dR2_content_on_M']:+.5f}  #sig_FDR={nsig}/{meta['n_features']}")
            log(f"      b(gross)={meta['b']:+.4f}  b(ctrl content)={meta['b_ctrl']:+.4f}  | "
                f"TOTAL content→ret p={meta['total_ret_p']:.3g} ΔR²={meta['total_ret_dR2']:+.5f}")
            top = table.reindex(table["a_p"].abs().sort_values().index).head(5)
            for _, r in top.iterrows():
                log(f"      {r['feature']}: a={r['a_coef']:+.4f} p={r['a_p']:.3g} "
                    f"fdr={r['a_fdr']:.3g} a·b={r['indirect_ak_b']:+.5f}")
        persub = pd.concat(persub_rows, ignore_index=True)
        persub.to_csv(OUT / "apath_multivariate_persubtype.csv", index=False)

    # ---------- m×H sweep: joint a-block test + b (sub30 only, to stay tractable) ----------
    sweep = pd.DataFrame()
    if args.no_sweep:
        args.sweep_m = []
    log(f"\nSWEEP m×H  m∈{args.sweep_m} × H∈{args.sweep_H}  (sub30 joint a-block + b)")
    sweep_rows = []
    for m in args.sweep_m:
        for H in args.sweep_H:
            df = assemble(args.coins, args.W, m, H, args.min_items, args.min_posts, args.z_win)
            names, C = content_matrix(df, "sub30")
            table, meta = dml_multivariate_apath(df, names, C, args.folds, args.seed)
            nsig = int(table["sig_fdr_5pct"].sum())
            sweep_rows.append({"m": m, "H": H, "n": meta["n"],
                               "joint_a_p": meta["joint_p"],
                               "dR2_content_on_M": meta["dR2_content_on_M"],
                               "b_gross": meta["b"], "b_ctrl_content": meta["b_ctrl"],
                               "total_ret_p": meta["total_ret_p"],
                               "total_ret_dR2": meta["total_ret_dR2"], "nsig_fdr_30": nsig})
            log(f"  m={m} H={H}: n={meta['n']:,}  joint a-p={meta['joint_p']:.3g}  "
                f"ΔR²(→M)={meta['dR2_content_on_M']:+.5f}  b={meta['b']:+.4f}/{meta['b_ctrl']:+.4f}  "
                f"tot→ret p={meta['total_ret_p']:.3g}  #sig={nsig}/30")
    sfx = args.out_suffix
    if sweep_rows:
        sweep = pd.DataFrame(sweep_rows)
        sweep.to_csv(OUT / f"apath_multivariate_sweep{sfx}.csv", index=False)

    # ---------- summary ----------
    md = ["# Stage 3.5b — multivariate / per-subtype a-path (does the 1-D-scalar null survive?)",
          "", "Multivariate within-coin a-path: mediator M (forward volume) regressed on the full "
          "content vector with controls (regime, vol, trailing return) + coin FE partialled out via "
          "cross-fitted DML; cluster-robust Wald block test; per-subtype BH-FDR; b-path M→return "
          "(gross and controlling content); TOTAL content→return; per-subtype indirect a_k·b. Noise "
          "family 3.x dropped (30 subtypes); families = aggregated family-4/family-5 shares. "
          "Moderator Z = trailing-168h BTC return."]
    if base_meta:
        md += ["", f"## Base cell W={args.W}h → m={args.m}h → H={args.H}h  (n={len(df0):,})", "",
               "| granularity | #feat | joint a p | ΔR²(content→M) | b(gross) | b(ctrl content) | "
               "TOTAL content→ret p | TOTAL ΔR² | #sig FDR |",
               "|---|---|---|---|---|---|---|---|---|"]
        for kind in ["sub30", "family"]:
            mt = base_meta[kind]
            ns = int(persub[(persub.granularity == kind)]["sig_fdr_5pct"].sum())
            md.append(f"| {kind} | {mt['n_features']} | {mt['joint_p']:.3g} | "
                      f"{mt['dR2_content_on_M']:+.5f} | {mt['b']:+.4f} | {mt['b_ctrl']:+.4f} | "
                      f"{mt['total_ret_p']:.3g} | {mt['total_ret_dR2']:+.5f} | {ns} |")
        md += ["", "Per-subtype coefficients, FDR, and indirect effects (a_k·b gross and "
               "a_k·b-ctrl) in `apath_multivariate_persubtype.csv`."]
    if sweep_rows:
        md += ["", "## m×H sweep (sub30 joint a-block + b-path)", "",
               "```", pd.DataFrame(sweep_rows).to_string(index=False), "```"]
    md += ["", "## How to read this", "",
           "The multivariate a-path (content→M) is the corrected test of whether narrative content "
           "predicts the within-coin volume mediator, after the 1-D frozen scalar of scripts 53/54 "
           "may have washed out a subtype-specific channel. The decisive contrast for the RETURN "
           "question is between a strong a-path and the **TOTAL content→return** column: if content "
           "predicts volume (a≠0) and volume predicts return (b≠0) yet content does NOT predict "
           "return (TOTAL p large), then the narrative-driven component of volume is "
           "direction-neutral — it moves activity/volatility but not signed return — so the naive "
           "indirect a·b overstates a channel that does not exist in the total effect. b(ctrl "
           "content) vs b(gross) shows how much of the volume→return slope is orthogonal to content."]
    (OUT / f"apath_multivariate_summary{sfx}.md").write_text("\n".join(md))

    if base_meta:
        log(f"\n✓ per-subtype -> {OUT/'apath_multivariate_persubtype.csv'}")
    if sweep_rows:
        log(f"✓ sweep       -> {OUT/f'apath_multivariate_sweep{sfx}.csv'}")
    log(f"✓ summary     -> {OUT/f'apath_multivariate_summary{sfx}.md'}")


if __name__ == "__main__":
    main()
