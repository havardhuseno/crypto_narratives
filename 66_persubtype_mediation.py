#!/usr/bin/env python3
"""
66_persubtype_mediation.py  (v2)
================================
Proper indirect-vs-direct decomposition of content's effect on RETURN and VOLATILITY through the
single continuous mediator = trading VOLUME. Fixes the lossy-scalar mediation (53/54): content enters
as the 30 subtype shares (a vector), volume stays one continuous mediator.

Cross-fitted DML partialling on controls W (regime, vol, trailing return, Reddit volume, coin FE):
  a-vector:  M_res  ~ Σ_k a_k · content_k_res                          (content -> volume)
  return eq: Yret_res ~ b·M_res + Σ_k c_k · content_k_res              (b: vol->ret | content; c_k: direct)
  vol eq:    Yvol_res ~ bv·M_res + Σ_k cv_k · content_k_res            (bv: vol->vol | content; cv_k: direct)
Per subtype: indirect_return_k = a_k·b ; indirect_vol_k = a_k·bv ; direct = c_k / cv_k.
Aggregate (difference method, single mediator): for each outcome,
  total  = ΔR² of content block alone;  direct = content block's marginal ΔR² controlling M;
  mediated = total − direct;  fraction_mediated = mediated / total.
Cluster block-bootstrap (over bar timestamp h) for b, bv and the mediated fractions.

Outputs (v2): outputs/persubtype_mediation_{table,aggregate}.csv, outputs/persubtype_mediation_summary.md
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

PAPER_ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = PAPER_ROOT / "outputs"


def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


m55 = _load("mv55", "55_multivariate_apath.py")
m53 = m55.m53
assemble = m53.assemble
COINS = m53.COINS
SUB30 = m55.SUB30
zsc = m55.zsc
bh_fdr = m55.bh_fdr
crossfit_resid = m55.m54.crossfit_resid
make_folds = m55.m54.make_folds
m59 = _load("c59", "59_contemporaneous_spec.py")


def r2(y, X):
    """R^2 of OLS y~X (X already includes whatever columns; intercept added)."""
    import numpy as np
    Xc = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    resid = y - Xc @ beta
    return 1.0 - np.sum(resid ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=COINS)
    ap.add_argument("--W", type=int, default=24); ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--H", type=int, default=24); ap.add_argument("--z-win", type=int, default=168)
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contemp", action="store_true",
                    help="contemporaneous spec: volume concurrent with content over [t-W,t), "
                         "forward return/vol over [t,t+H) (no mediator window m)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); log = print
    import statsmodels.api as sm
    from sklearn.ensemble import HistGradientBoostingRegressor

    spec = "contemp" if args.contemp else "sequential"
    sfx = "_contemp" if args.contemp else ""
    if args.contemp:
        df = m59.assemble_contemp(args.coins, args.W, args.H, m53.m46.MIN_ITEMS, 3, args.z_win)
    else:
        df = assemble(args.coins, args.W, args.m, args.H, m53.m46.MIN_ITEMS, 3, args.z_win)
    log(f"[{spec}] n={len(df):,}")
    rng = np.random.default_rng(args.seed); folds = make_folds(len(df), args.folds, rng)
    def gbr():
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                             min_samples_leaf=50, l2_regularization=1.0, random_state=args.seed)
    M = zsc(df["M"].values); Yr = zsc(df["Yret"].values); Yv = zsc(df["Yvol"].values)
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float).values
    W = np.column_stack([zsc(df["ms_regime_prob_high"].values), zsc(df["vol_ord"].values),
                         zsc(df["trail_ret"].values),
                         zsc(df["att_log_posts"].values), zsc(df["att_log_items"].values), coin_d])
    C = np.column_stack([zsc(df[f"share_{s}"].values) for s in SUB30])

    # cross-fitted residuals on controls W
    M_r, _ = crossfit_resid(M, W, folds, gbr)
    Yr_r, _ = crossfit_resid(Yr, W, folds, gbr)
    Yv_r, _ = crossfit_resid(Yv, W, folds, gbr)
    C_r = np.column_stack([crossfit_resid(C[:, k], W, folds, gbr)[0] for k in range(C.shape[1])])

    grp = df["h"].values
    # a-vector (content->volume), cluster-robust p
    ra = sm.OLS(M_r, sm.add_constant(C_r)).fit(cov_type="cluster", cov_kwds={"groups": grp})
    a = np.asarray(ra.params[1:]); a_p = np.asarray(ra.pvalues[1:])
    # return / vol equations: Y_res ~ [M_res, C_res]  (b/bv on M; c_k/cv_k direct)
    Xm = sm.add_constant(np.column_stack([M_r, C_r]))
    rr = sm.OLS(Yr_r, Xm).fit(cov_type="cluster", cov_kwds={"groups": grp})
    b = float(rr.params[1]); c = np.asarray(rr.params[2:]); c_p = np.asarray(rr.pvalues[2:])
    rv = sm.OLS(Yv_r, Xm).fit(cov_type="cluster", cov_kwds={"groups": grp})
    bv = float(rv.params[1]); cvk = np.asarray(rv.params[2:]); cv_p = np.asarray(rv.pvalues[2:])
    a_fdr = bh_fdr(a_p); cr_fdr = bh_fdr(c_p); cv_fdr = bh_fdr(cv_p)
    tab = pd.DataFrame({
        "subtype": SUB30,
        "a (content->vol)": a, "a_fdr": a_fdr,
        "indirect_return (a*b)": a * b, "direct_return (c_k)": c, "direct_return_fdr": cr_fdr,
        "indirect_vol (a*bv)": a * bv, "direct_vol (cv_k)": cvk, "direct_vol_fdr": cv_fdr,
    })
    tab.to_csv(OUT / f"persubtype_mediation_table{sfx}.csv", index=False)

    # aggregate difference method
    def decomp(Y_r):
        total = r2(Y_r, C_r)                          # content alone
        full = r2(Y_r, np.column_stack([M_r, C_r]))   # content + volume
        m_only = r2(Y_r, M_r[:, None])                # volume alone
        direct = full - m_only                         # content's marginal over volume
        mediated = total - direct
        frac = mediated / total if total > 1e-12 else np.nan
        return total, direct, mediated, frac
    tot_r, dir_r, med_r, frac_r = decomp(Yr_r)
    tot_v, dir_v, med_v, frac_v = decomp(Yv_r)

    # cluster bootstrap (over h) for b, bv, frac_r, frac_v
    h = df["h"].values; cl = {}
    for i, hh in enumerate(h): cl.setdefault(hh, []).append(i)
    keys = list(cl.values()); rb = np.random.default_rng(args.seed + 1)
    bb, bbv, fr, fv = [], [], [], []
    for _ in range(args.n_boot):
        idx = np.concatenate([keys[j] for j in rb.integers(0, len(keys), len(keys))])
        Mr, Yrr, Yvr, Cr = M_r[idx], Yr_r[idx], Yv_r[idx], C_r[idx]
        Xb = np.column_stack([Mr, Cr])
        bb.append(np.linalg.lstsq(Xb, Yrr, rcond=None)[0][0])
        bbv.append(np.linalg.lstsq(Xb, Yvr, rcond=None)[0][0])
        tr = r2(Yrr, Cr); fu = r2(Yrr, Xb); mo = r2(Yrr, Mr[:, None]); fr.append((tr-(fu-mo))/tr if tr>1e-12 else np.nan)
        tv = r2(Yvr, Cr); fuv = r2(Yvr, Xb); mov = r2(Yvr, Mr[:, None]); fv.append((tv-(fuv-mov))/tv if tv>1e-12 else np.nan)
    def ci(x): x=np.array(x); return float(np.nanmean(x)), float(np.nanquantile(x,.025)), float(np.nanquantile(x,.975))
    bm = ci(bb); bvm = ci(bbv); frm = ci(fr); fvm = ci(fv)

    agg = pd.DataFrame([
        {"outcome": "return", "b_or_bv (vol->Y)": b, "boot_lo": bm[1], "boot_hi": bm[2],
         "total_R2_content": tot_r, "direct_R2": dir_r, "mediated_R2": med_r,
         "frac_mediated_by_volume": frac_r, "frac_lo": frm[1], "frac_hi": frm[2]},
        {"outcome": "volatility", "b_or_bv (vol->Y)": bv, "boot_lo": bvm[1], "boot_hi": bvm[2],
         "total_R2_content": tot_v, "direct_R2": dir_v, "mediated_R2": med_v,
         "frac_mediated_by_volume": frac_v, "frac_lo": fvm[1], "frac_hi": fvm[2]},
    ])
    agg.to_csv(OUT / f"persubtype_mediation_aggregate{sfx}.csv", index=False)
    log(f"\n=== [{spec}] volume's effect & mediation decomposition (v2) ===")
    log(agg.to_string(index=False))
    log(f"\nb (vol->return)  = {b:+.4f}  [{bm[1]:+.4f},{bm[2]:+.4f}]")
    log(f"bv (vol->vol)    = {bv:+.4f}  [{bvm[1]:+.4f},{bvm[2]:+.4f}]")
    log(f"return : total content R²={tot_r:.5f}  direct={dir_r:.5f}  mediated-by-vol={med_r:.5f}  frac={frac_r:.2%} [{frm[1]:.2%},{frm[2]:.2%}]")
    log(f"vol    : total content R²={tot_v:.5f}  direct={dir_v:.5f}  mediated-by-vol={med_v:.5f}  frac={frac_v:.2%} [{fvm[1]:.2%},{fvm[2]:.2%}]")
    # which content types matter (significant DIRECT effects, since mediation ~0)
    sig_v = tab[tab["direct_vol_fdr"] < 0.05].reindex(
        tab[tab["direct_vol_fdr"] < 0.05]["direct_vol (cv_k)"].abs().sort_values(ascending=False).index)
    sig_r = tab[tab["direct_return_fdr"] < 0.05].reindex(
        tab[tab["direct_return_fdr"] < 0.05]["direct_return (c_k)"].abs().sort_values(ascending=False).index)
    log(f"\nsubtypes with significant DIRECT volatility effect (FDR<5%): {len(sig_v)}/30")
    log(sig_v[["subtype", "direct_vol (cv_k)", "direct_vol_fdr"]].head(10).to_string(index=False))
    log(f"\nsubtypes with significant DIRECT return effect (FDR<5%): {len(sig_r)}/30")
    log(sig_r[["subtype", "direct_return (c_k)", "direct_return_fdr"]].head(10).to_string(index=False))

    md = ["# Per-subtype mediation via trading volume (v2)", "",
          f"Single continuous mediator = volume; content = 30 subtype shares. n={len(df):,}. "
          "Cross-fitted DML on controls (regime, vol, trailing return, Reddit volume, coin FE). "
          "Difference-method aggregate; cluster block-bootstrap over bar timestamps.", "",
          "## Volume's effect and the direct/indirect split", "", "```", agg.to_string(index=False), "```", "",
          f"- **Return:** volume→return b={b:+.3f}; content's predictive R² is **{frac_r:.0%}** mediated by "
          f"volume (rest direct). ",
          f"- **Volatility:** volume→vol bv={bv:+.3f}; content's predictive R² is **{frac_v:.0%}** mediated by "
          f"volume.", "",
          "Per-subtype indirect (a·b, a·bv) and direct (c_k, cv_k) in `persubtype_mediation_table.csv`."]
    md += ["", f"## Which content types matter ({spec})", "",
           f"Significant DIRECT volatility effect (FDR<5%): {len(sig_v)}/30 subtypes; "
           f"significant DIRECT return effect: {len(sig_r)}/30. Top in "
           f"`persubtype_mediation_table{sfx}.csv`."]
    (OUT / f"persubtype_mediation_summary{sfx}.md").write_text("\n".join(md))
    log(f"\n✓ -> {OUT/'persubtype_mediation_aggregate.csv'} , _table.csv , _summary.md")


if __name__ == "__main__":
    main()
