#!/usr/bin/env python3
"""
54_ml_mediation_triangulation.py
================================
STAGE 3.6 — machine-learning TRIANGULATION of the moderated-mediation result (script 53).

Same estimands, orthogonal method. Where script 53 is a fully-parametric hierarchical
Bayesian path model, this script estimates the *same* content→activity→price chain with
double/debiased machine learning (Chernozhukov et al. 2018) and a data-driven test for the
amplification (moderation) the Bayesian model imposes parametrically. If both agree, the
conclusion is method-robust; if they disagree, that is itself informative.

Three pieces (all reuse script 53's leak-free panel + frozen content scalar Xs verbatim):

  1. DML partial-linear path coefficients (cross-fitted, Neyman-orthogonal / Robinson 1988):
       a-path :  Xs → M                          residualise both on controls W, regress
       b,d    :  M, M·Z → Y  (+ direct Xs, Z)    partialling-out, moderation by Z
       bv     :  M → V                            magnitude channel
     Nuisances E[·|W] via HistGradientBoostingRegressor (flexible, no linearity assumption
     on the CONTROLS). Indirect effect IE(Z)=â·(b̂+d̂·Z); index of moderated mediation = â·d̂.
     Inference by a CLUSTER block-bootstrap over bar-timestamps h (matches script 53's SE
     clustering and the non-overlap design).

  2. R-learner CATE τ(Z) = ∂Y/∂M as a smooth function of the ambient direction Z
     (Nie & Wager 2021). This is the data-driven version of the d-path: orthogonalise Y and
     M on (W, Z), form the R-learner pseudo-response, fit a RandomForest of the local M→Y
     effect on Z, and read off the Johnson-Neyman threshold (where τ(Z) crosses zero). The
     amplification hypothesis predicts τ(Z) increasing in Z and sign-flipping — a pattern a
     linear M·Z term can only approximate.

  3. Activity-hinge robustness: add relu(M−knot) to the b-equation to let the tail of
     activity carry a different M→Y slope (the one targeted nonlinearity the scan motivates).

Outputs (all under paper_narrative/, never touches production):
  outputs/ml_mediation_dml.csv        DML path coefficients + bootstrap CIs + IE(Z)
  outputs/ml_mediation_cate.csv       R-learner τ(Z) curve on a Z grid (+ JN threshold)
  outputs/ml_mediation_summary.md     human-readable write-up, cross-referenced to 53
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


m53 = _load("med53", "53_bayes_moderated_mediation.py")
assemble = m53.assemble
frozen_content_score = m53.frozen_content_score
COINS = m53.COINS


def zsc(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


# ============================================================ cross-fitting helpers
def crossfit_resid(y, X, folds, make_model):
    """Out-of-fold residuals y − Ê[y|X] (cross-fitted nuisance, no own-fold leakage)."""
    yhat = np.empty_like(y, dtype=float)
    for tr, te in folds:
        mdl = make_model()
        mdl.fit(X[tr], y[tr])
        yhat[te] = mdl.predict(X[te])
    return y - yhat, yhat


def make_folds(n, k, rng):
    idx = rng.permutation(n)
    parts = np.array_split(idx, k)
    folds = []
    for i in range(k):
        te = parts[i]
        tr = np.concatenate([parts[j] for j in range(k) if j != i])
        folds.append((tr, te))
    return folds


# ============================================================ DML path estimation
def dml_paths(df, Xs, k_folds, seed, hinge_knot_q=0.90):
    """Cross-fitted partial-linear DML for a, (b,d), bv. Returns point estimates and the
    per-row orthogonal scores needed for the cluster bootstrap."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    rng = np.random.default_rng(seed)
    n = len(df)
    folds = make_folds(n, k_folds, rng)

    def gbr():
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=50, l2_regularization=1.0, random_state=seed)

    # standardised structural variables
    M = zsc(df["M"].values)
    Yr = zsc(df["Yret"].values)
    Yv = zsc(df["Yvol"].values)
    Z = zsc(df["Z"].values)
    Xs = np.asarray(Xs, float)

    # controls W: regime, vol, trailing return, coin dummies (full FE — ML is robust to many)
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float).values
    W = np.column_stack([
        zsc(df["ms_regime_prob_high"].values), zsc(df["vol_ord"].values),
        zsc(df["trail_ret"].values), coin_d])
    WZ = np.column_stack([W, Z])               # controls + ambient direction

    # ---- a-path: Xs -> M, partialling out W ----
    Xs_rW, _ = crossfit_resid(Xs, W, folds, gbr)
    M_rW, _ = crossfit_resid(M, W, folds, gbr)
    a_hat = float(np.dot(Xs_rW, M_rW) / np.dot(Xs_rW, Xs_rW))

    # ---- b,d-path: M, M*Z -> Y, partialling out (W,Z) and the direct Xs, Xs*Z ----
    # build treatments T = [M, M*Z, Xs, Xs*Z]; residualise each + Y on WZ; OLS Y_r ~ T_r
    MZ = M * Z
    XsZ = Xs * Z
    Yr_r, _ = crossfit_resid(Yr, WZ, folds, gbr)
    T_cols = {"M": M, "MZ": MZ, "Xs": Xs, "XsZ": XsZ}
    T_r = {k: crossfit_resid(v, WZ, folds, gbr)[0] for k, v in T_cols.items()}
    Tmat = np.column_stack([T_r["M"], T_r["MZ"], T_r["Xs"], T_r["XsZ"]])
    coef_r, *_ = np.linalg.lstsq(Tmat, Yr_r, rcond=None)
    b_hat, d_hat, c_hat, e_hat = (float(coef_r[0]), float(coef_r[1]),
                                  float(coef_r[2]), float(coef_r[3]))

    # ---- activity-hinge robustness: add relu(M - knot) to the b-equation ----
    knot = float(np.quantile(M, hinge_knot_q))
    hinge = np.clip(M - knot, 0, None)
    hinge_r, _ = crossfit_resid(hinge, WZ, folds, gbr)
    Tmat_h = np.column_stack([T_r["M"], T_r["MZ"], hinge_r, T_r["Xs"], T_r["XsZ"]])
    coef_h, *_ = np.linalg.lstsq(Tmat_h, Yr_r, rcond=None)
    b_lin_h, d_h, b_hinge_h = float(coef_h[0]), float(coef_h[1]), float(coef_h[2])

    # ---- bv-path: M -> V, partialling out W and direct Xs ----
    WXs = np.column_stack([W, Xs])
    Yv_r, _ = crossfit_resid(Yv, WXs, folds, gbr)
    M_rWXs, _ = crossfit_resid(M, WXs, folds, gbr)
    bv_hat = float(np.dot(M_rWXs, Yv_r) / np.dot(M_rWXs, M_rWXs))

    return {
        "resid": {"Xs_rW": Xs_rW, "M_rW": M_rW, "Yr_r": Yr_r, "Tmat": Tmat,
                  "M_rWXs": M_rWXs, "Yv_r": Yv_r},
        "point": {"a": a_hat, "b": b_hat, "d": d_hat, "c_dir": c_hat, "e_dir": e_hat,
                  "bv": bv_hat, "b_lin_hinge": b_lin_h, "d_hinge": d_h,
                  "b_hinge_relu": b_hinge_h, "hinge_knot_z": knot},
    }


def _refit_from_resid(resid, idx):
    """Recompute the path point-estimates on a bootstrap index from cached residuals (fast —
    nuisances are NOT refit; valid because the orthogonal scores are already cross-fitted)."""
    Xs_rW, M_rW = resid["Xs_rW"][idx], resid["M_rW"][idx]
    a = np.dot(Xs_rW, M_rW) / np.dot(Xs_rW, Xs_rW)
    Tmat, Yr_r = resid["Tmat"][idx], resid["Yr_r"][idx]
    coef, *_ = np.linalg.lstsq(Tmat, Yr_r, rcond=None)
    b, d = coef[0], coef[1]
    M_rWXs, Yv_r = resid["M_rWXs"][idx], resid["Yv_r"][idx]
    bv = np.dot(M_rWXs, Yv_r) / np.dot(M_rWXs, M_rWXs)
    return a, b, d, bv


def cluster_bootstrap(df, resid, point, n_boot, seed):
    """Block bootstrap over bar-timestamps h (clusters): resample whole-h blocks, recompute
    the path estimates and derived IE/index from cached orthogonal residuals."""
    rng = np.random.default_rng(seed + 1)
    h = df["h"].values
    clusters = {}
    for pos, hh in enumerate(h):
        clusters.setdefault(hh, []).append(pos)
    cl_keys = np.array(list(clusters.keys()))
    cl_idx = [np.asarray(clusters[k]) for k in cl_keys]
    ncl = len(cl_keys)

    keys = ["a", "b", "d", "bv", "index_modmed", "IE_z0", "IE_zpos1", "IE_zneg1"]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        pick = rng.integers(0, ncl, size=ncl)
        idx = np.concatenate([cl_idx[p] for p in pick])
        a, b, d, bv = _refit_from_resid(resid, idx)
        draws["a"].append(a); draws["b"].append(b); draws["d"].append(d); draws["bv"].append(bv)
        draws["index_modmed"].append(a * d)
        draws["IE_z0"].append(a * b)
        draws["IE_zpos1"].append(a * (b + d))
        draws["IE_zneg1"].append(a * (b - d))

    pt = {"a": point["a"], "b": point["b"], "d": point["d"], "bv": point["bv"],
          "index_modmed": point["a"] * point["d"], "IE_z0": point["a"] * point["b"],
          "IE_zpos1": point["a"] * (point["b"] + point["d"]),
          "IE_zneg1": point["a"] * (point["b"] - point["d"])}
    rows = []
    for k in keys:
        arr = np.asarray(draws[k])
        rows.append({"quantity": k, "estimate": pt[k],
                     "boot_mean": float(arr.mean()), "se": float(arr.std()),
                     "ci_lo": float(np.quantile(arr, 0.025)),
                     "ci_hi": float(np.quantile(arr, 0.975)),
                     "P(>0)": float((arr > 0).mean())})
    return pd.DataFrame(rows)


# ============================================================ R-learner CATE in Z
def rlearner_cate(df, Xs, k_folds, seed, n_grid=41):
    """τ(Z) = local M→Y effect as a function of ambient direction Z (Nie–Wager R-learner).
    Orthogonalise Y and M on (W, Z, Xs); fit a forest of the pseudo-effect on Z with the
    R-learner weights (M−m̂)². Returns the τ(Z) curve and the Johnson–Neyman zero crossing."""
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

    rng = np.random.default_rng(seed + 7)
    n = len(df)
    folds = make_folds(n, k_folds, rng)

    def gbr():
        return HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=50, l2_regularization=1.0, random_state=seed)

    M = zsc(df["M"].values)
    Yr = zsc(df["Yret"].values)
    Z = zsc(df["Z"].values)
    Xs = np.asarray(Xs, float)
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float).values
    WZX = np.column_stack([
        zsc(df["ms_regime_prob_high"].values), zsc(df["vol_ord"].values),
        zsc(df["trail_ret"].values), coin_d, Z, Xs])

    M_res, _ = crossfit_resid(M, WZX, folds, gbr)
    Y_res, _ = crossfit_resid(Yr, WZX, folds, gbr)

    # R-learner: minimise Σ w·(pseudo − τ(Z))² with pseudo = Y_res/M_res, w = M_res²
    w = M_res ** 2
    eps = 1e-6
    pseudo = Y_res / (M_res + np.sign(M_res) * eps + (M_res == 0) * eps)
    # guard against exploding pseudo where M_res ~ 0 (weights already downweight these)
    finite = np.isfinite(pseudo)
    forest = RandomForestRegressor(
        n_estimators=400, min_samples_leaf=200, max_features=1,
        random_state=seed, n_jobs=-1)
    forest.fit(Z[finite].reshape(-1, 1), pseudo[finite], sample_weight=w[finite])

    zgrid = np.linspace(np.quantile(Z, 0.02), np.quantile(Z, 0.98), n_grid)
    tau = forest.predict(zgrid.reshape(-1, 1))

    # Johnson–Neyman: first zero crossing of τ(Z)
    jn = None
    s = np.sign(tau)
    cross = np.where(np.diff(s) != 0)[0]
    if len(cross):
        i0 = cross[0]
        z0, z1, t0, t1 = zgrid[i0], zgrid[i0 + 1], tau[i0], tau[i0 + 1]
        jn = float(z0 - t0 * (z1 - z0) / (t1 - t0))

    curve = pd.DataFrame({"Z": zgrid, "tau_M_to_ret": tau})
    return curve, jn


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
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = print

    if args.smoke:
        args.coins = ["BTC", "ETH", "SOL"]
        args.folds, args.n_boot = 3, 50

    log(f"assembling panel: coins={args.coins} W={args.W}h m={args.m}h H={args.H}h "
        f"Z=trailing{args.z_win}h BTC ret")
    df = assemble(args.coins, args.W, args.m, args.H, args.min_items, args.min_posts, args.z_win)
    Xs = frozen_content_score(df)
    log(f"  pooled bars: {len(df):,}  | corr(Xs,M)={np.corrcoef(Xs, df['M'])[0,1]:+.3f}")

    # ---- DML path estimation ----
    log("DML partial-linear path estimation (cross-fitted)…")
    dml = dml_paths(df, Xs, args.folds, args.seed)
    p = dml["point"]
    log(f"  a={p['a']:+.4f}  b={p['b']:+.4f}  d={p['d']:+.4f}  bv={p['bv']:+.4f}")
    log(f"  hinge: b_lin={p['b_lin_hinge']:+.4f}  relu(M>{p['hinge_knot_z']:+.2f})="
        f"{p['b_hinge_relu']:+.4f}  d={p['d_hinge']:+.4f}")
    log(f"  cluster block bootstrap ({args.n_boot} reps over h)…")
    dml_tab = cluster_bootstrap(df, dml["resid"], p, args.n_boot, args.seed)
    dml_tab.to_csv(OUT / "ml_mediation_dml.csv", index=False)
    log("\nDML effects:\n" + dml_tab.to_string(index=False))

    # ---- R-learner CATE τ(Z) ----
    log("\nR-learner CATE τ(Z) = M→ret effect vs ambient direction…")
    curve, jn = rlearner_cate(df, Xs, args.folds, args.seed)
    curve.to_csv(OUT / "ml_mediation_cate.csv", index=False)
    tlo, thi = curve["tau_M_to_ret"].iloc[0], curve["tau_M_to_ret"].iloc[-1]
    log(f"  τ(Z=low)={tlo:+.4f}  τ(Z=high)={thi:+.4f}  slope sign={'up' if thi>tlo else 'down'}"
        f"  | Johnson-Neyman zero crossing Z*={jn if jn is None else round(jn,3)}")

    # ---- summary.md ----
    md = ["# Stage 3.6 — ML triangulation of the moderated mediation (DML + R-learner)", "",
          f"Same leak-free panel + frozen content scalar Xs as script 53. Cell: content "
          f"W={args.W}h → activity m={args.m}h → outcome [t+m,t+H), H={args.H}h; moderator "
          f"Z = standardised trailing {args.z_win}h BTC return. Pooled bars n={len(df):,}. "
          f"DML nuisances: HistGradientBoosting, {args.folds}-fold cross-fitting; inference "
          f"by cluster block-bootstrap over bar-timestamps ({args.n_boot} reps).", "",
          "## DML partial-linear path coefficients & indirect effects", "",
          "```", dml_tab.to_string(index=False), "```", "",
          f"Activity hinge (relu of M above its {int(0.90*100)}th pct, z={p['hinge_knot_z']:+.2f}): "
          f"linear b={p['b_lin_hinge']:+.4f}, hinge slope={p['b_hinge_relu']:+.4f}, "
          f"d={p['d_hinge']:+.4f}. A non-zero hinge slope = the activity *tail* carries a "
          "different M→return effect than the body.", "",
          "## R-learner CATE τ(Z) = local M→return effect vs ambient direction", "",
          f"τ at low Z = {tlo:+.4f}; τ at high Z = {thi:+.4f}; Johnson–Neyman zero crossing "
          f"Z* = {jn if jn is None else round(jn,3)}. The amplification hypothesis predicts τ(Z) "
          "increasing in Z with a sign flip — i.e. activity pushes price UP under bullish "
          "ambient direction and DOWN under bearish — which a single linear M·Z term can only "
          "approximate. Full curve in `ml_mediation_cate.csv`.", "",
          "## Cross-reference to script 53 (hierarchical Bayesian)", "",
          "Compare DML `index_modmed`/`IE_z{0,±1}` and the τ(Z) slope to the Bayesian "
          "`index_modmed`/`IE_ret_z*`. Agreement ⇒ the conclusion is method-robust (not an "
          "artifact of the parametric path model); disagreement localises where the linear "
          "moderation mis-specifies the heterogeneity."]
    (OUT / "ml_mediation_summary.md").write_text("\n".join(md))

    log(f"\n✓ dml      -> {OUT/'ml_mediation_dml.csv'}")
    log(f"✓ cate     -> {OUT/'ml_mediation_cate.csv'}")
    log(f"✓ summary  -> {OUT/'ml_mediation_summary.md'}")


if __name__ == "__main__":
    main()
