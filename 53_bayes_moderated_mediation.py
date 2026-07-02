#!/usr/bin/env python3
"""
53_bayes_moderated_mediation.py
===============================
STAGE 3.5 — hierarchical Bayesian MODERATED MEDIATION of the social-media → price chain.

Motivation (from Stage 1, script 50): narrative CONTENT predicts forward VOLUME / VOLATILITY
strongly and horizon-robustly, but predicts SIGNED RETURN ~zero unconditionally. Hypothesis
(user): content moves price only by AMPLIFYING the prevailing market direction — bullish
ambient outlook + narrative/activity shock → up; bearish outlook + same shock → down — so the
unconditional effect cancels to the null we observe. That is a *moderated mediation*:

    content X [t-W, t)  →  activity M (fwd volume) [t, t+m)  →  outcome Y [t+m, t+H)
    moderator Z (ambient direction, predetermined as-of t) gates the SIGN of the b-path.

Leak-free 3-segment timeline (X strictly before M strictly before Y; Z trailing as-of t).

Model (path analysis estimated as a hierarchical Bayesian system; NumPyro/NUTS):
    M_n ~ N( αM_j + a_j·Xs            + γ'Cm , σM )                       # a-path
    R_n ~ N( αR_j + b_j·M + d_j·(M·Z) + c·Xs + e·(Xs·Z) + θ'Cr , σR )    # b-path, MODERATED
    V_n ~ N( αV_j + bv_j·M            + cv·Xs + φ'Cv , σV )              # magnitude, UNcond.
  coin-varying slopes (a_j,b_j,d_j,bv_j) ~ MVN(μ, Σ), Σ via LKJ  (partial pooling, non-centred)
  Mundlak coin-mean covariates in Cm/Cr/Cv  → absorbs the fixed-effects (correlated-effects)
    concern while keeping the multilevel structure.

Estimands (full posterior, no bootstrap):
  IE_ret(Z) = a·(b + d·Z)         conditional indirect effect at Z ∈ {−1,0,+1} SD
  index of moderated mediation = a·d   (the single test that amplification is real)
  IE_vol    = a·bv                the confirmed magnitude channel

Identification: the headline assumes sequential ignorability (residuals independent across the
three equations). Because M is endogenous in the R-eq, freely estimating corr(ε_M,ε_R) is NOT
identified without an instrument — so instead we run an Imai-style SENSITIVITY ANALYSIS: the
b-path (and hence IE) is re-expressed under an assumed residual correlation ρ over a grid, and
we report how large |ρ| must be to overturn the conclusion. Honest, and standard.

Content scalar Xs ("narrative-implied activity"): a FROZEN linear map from the 34 codebook
shares + dynamics onto activity, fit ONLY on the pre-2023 train era (no returns involved, and
time-separated), then applied to all bars. This breaks the circularity that a supervised
within-sample projection would create in the a-path coefficient.

Companion (frequentist, printed for triangulation): the reduced-form interaction regression
and the bull/bear sign-split that the moderated mediation predicts.

Outputs (all under paper_narrative/, never touches production):
  outputs/mediation_bayes_effects.csv      posterior summaries of the path/effect quantities
  outputs/mediation_bayes_sensitivity.csv  IE_ret vs assumed confound ρ
  outputs/mediation_bayes_summary.md       human-readable write-up + diagnostics
  outputs/mediation_bayes_idata.nc         full ArviZ InferenceData (for figures later)
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

# ---- reuse script 50 (which itself reuses 46): build_bars, build_coin_caches, panel ----
def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m50 = _load("dec50", "50_horizon_scan_decomposition.py")
m46 = m50.m46
build_bars = m46.build_bars
build_coin_caches = m50.build_coin_caches
PRICES = m46.PRICES
PANEL = m46.PANEL
HOUR = m46.HOUR
CB_LABELS = m46.CB_LABELS
COINS = m46.COINS

CONTENT_COLS = [f"share_{l}" for l in CB_LABELS]
DYN_COLS = ["topic_hhi", "topic_shift", "topic_novelty"]
TRAIN_END = pd.Timestamp("2023-01-01", tz="UTC")   # production train/test boundary
WINSOR = 0.005


# ============================================================ panel assembly
def forward_blocks(coin: str, bars_h: np.ndarray, caches: dict, m: int, H: int):
    """Forward mediator + outcomes on the [t,t+m) / [t+m,t+H) segments from 1h price.
    M    = log1p Σ volume over [t, t+m)
    Yret = close[t+H]/close[t+m] − 1                 (return over [t+m, t+H))
    Yvol = sqrt Σ hourly-logret² over (t+m, t+H]      (realized vol over [t+m, t+H))
    """
    c = caches[coin]
    close, hmin, hmax = c["close"], c["hmin"], c["hmax"]
    cum_r2, cum_vol = c["cum_r2"], c["cum_vol"]
    i = bars_h - hmin
    span = hmax - hmin
    n = len(bars_h)

    # mediator: forward volume over dense indices i .. i+m-1
    M = np.full(n, np.nan)
    okm = (i >= 0) & (i + m <= span + 1)
    M[okm] = np.log1p(np.clip(cum_vol[i[okm] + m] - cum_vol[i[okm]], 0, None))

    # outcomes over [t+m, t+H)
    c_tm = close.reindex(bars_h + m).values
    c_tH = close.reindex(bars_h + H).values
    Yret = c_tH / c_tm - 1.0

    Yvol = np.full(n, np.nan)
    okv = (i + m + 1 >= 0) & (i + H + 1 <= span + 1)
    Yvol[okv] = np.sqrt(np.clip(cum_r2[i[okv] + H + 1] - cum_r2[i[okv] + m + 1], 0, None))
    return M, Yret, Yvol


def assemble(coins, W, m, H, min_items, min_posts, z_win_h):
    panel = pd.read_parquet(PANEL, columns=["ts", "coin", "ms_regime_prob_high", "vol_regime"])
    panel["ts"] = pd.to_datetime(panel["ts"], utc=True)
    panel["vol_ord"] = panel["vol_regime"].cat.codes.astype(float)

    caches = {c: build_coin_caches(c) for c in set(coins) | {"BTC"}}

    # ambient direction Z: trailing z_win_h-hour BTC ("market") return, as-of t — common to all coins
    btc = caches["BTC"]["close"]

    parts = []
    for coin in coins:
        d = build_bars(coin, W)
        pc = panel[panel["coin"] == coin].sort_values("ts")
        d = pd.merge_asof(d.sort_values("ts"),
                          pc[["ts", "ms_regime_prob_high", "vol_ord"]],
                          on="ts", direction="backward")
        bars_h = d["h"].values.astype(int)
        M, Yret, Yvol = forward_blocks(coin, bars_h, caches, m, H)
        d["M"] = M; d["Yret"] = Yret; d["Yvol"] = Yvol
        # Z = trailing market (BTC) return over [t-z_win_h, t)
        d["Z"] = btc.reindex(bars_h).values / btc.reindex(bars_h - z_win_h).values - 1.0
        # non-overlapping bars at step = H (clean SEs / independent rows for the Bayesian fit)
        d = d[(d["h"] - d["h"].min()) % H == 0]
        parts.append(d)

    df = pd.concat(parts, ignore_index=True)
    keep = (df["n_items"] >= min_items) & (df["n_posts"] >= min_posts) \
        & df["M"].notna() & df["Yret"].notna() & df["Yvol"].notna() \
        & df["Z"].notna() & df["trail_ret"].notna() & df["ms_regime_prob_high"].notna()
    df = df[keep].copy()
    # winsorise the heavy-tailed quantities
    for col in ["Yret", "Yvol", "M", "Z", "trail_ret"]:
        lo, hi = df[col].quantile([WINSOR, 1 - WINSOR])
        df[col] = df[col].clip(lo, hi)
    df["is_train"] = df["ts"] < TRAIN_END
    return df


def frozen_content_score(df: pd.DataFrame) -> np.ndarray:
    """Frozen 'narrative-implied activity' index: OLS map of (34 shares + dynamics) → M, fit on
    the pre-2023 train era only (no returns; time-separated), applied to all rows. Returns the
    content-only fitted component (controls/intercept excluded), standardised."""
    feat = CONTENT_COLS + DYN_COLS
    ctrl = ["ms_regime_prob_high", "vol_ord", "trail_ret"]
    tr = df[df["is_train"]]
    Xtr = np.column_stack([np.ones(len(tr)), tr[ctrl].values, tr[feat].values])
    beta, *_ = np.linalg.lstsq(Xtr, tr["M"].values, rcond=None)
    w_content = beta[1 + len(ctrl):]                       # coefficients on content features only
    score = df[feat].values @ w_content                    # apply to all rows, content part only
    return (score - score.mean()) / (score.std() + 1e-12)


# ============================================================ Bayesian model
def build_model():
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    def model(coin_idx, n_coins, Xs, Z, M, Cm, Cr, Cv, Yret=None, Yvol=None):
        kM, kR, kV = Cm.shape[1], Cr.shape[1], Cv.shape[1]
        # ---- population slope means [a, b, d, bv] and correlated coin-varying slopes ----
        mu_slope = numpyro.sample("mu_slope", dist.Normal(0.0, 1.0).expand([4]).to_event(1))
        tau_slope = numpyro.sample("tau_slope", dist.HalfNormal(1.0).expand([4]).to_event(1))
        Lcorr = numpyro.sample("L_corr", dist.LKJCholesky(4, concentration=2.0))
        z_slope = numpyro.sample("z_slope", dist.Normal(0, 1).expand([n_coins, 4]).to_event(2))
        L = tau_slope[:, None] * Lcorr                      # scale the cholesky factor
        slopes = mu_slope[None, :] + z_slope @ L.T          # (n_coins, 4)
        a, b, d, bv = slopes[:, 0], slopes[:, 1], slopes[:, 2], slopes[:, 3]

        # ---- varying intercepts (independent hierarchical normals) ----
        def varying_intercept(name):
            mu = numpyro.sample(f"mu_{name}", dist.Normal(0.0, 1.0))
            tau = numpyro.sample(f"tau_{name}", dist.HalfNormal(1.0))
            z = numpyro.sample(f"z_{name}", dist.Normal(0, 1).expand([n_coins]).to_event(1))
            return mu + tau * z
        aM, aR, aV = varying_intercept("aM"), varying_intercept("aR"), varying_intercept("aV")

        # ---- population control / direct-path coefficients ----
        gamma = numpyro.sample("gamma", dist.Normal(0, 1).expand([kM]).to_event(1))
        theta = numpyro.sample("theta", dist.Normal(0, 1).expand([kR]).to_event(1))
        phi = numpyro.sample("phi", dist.Normal(0, 1).expand([kV]).to_event(1))
        c_dir = numpyro.sample("c_dir", dist.Normal(0, 1))      # direct X→R
        e_dir = numpyro.sample("e_dir", dist.Normal(0, 1))      # X·Z → R (direct moderation)
        cv_dir = numpyro.sample("cv_dir", dist.Normal(0, 1))    # direct X→V
        sM = numpyro.sample("sigma_M", dist.HalfNormal(1.0))
        sR = numpyro.sample("sigma_R", dist.HalfNormal(1.0))
        sV = numpyro.sample("sigma_V", dist.HalfNormal(1.0))

        muM = aM[coin_idx] + a[coin_idx] * Xs + Cm @ gamma
        numpyro.sample("M_obs", dist.Normal(muM, sM), obs=M)
        muR = (aR[coin_idx] + b[coin_idx] * M + d[coin_idx] * (M * Z)
               + c_dir * Xs + e_dir * (Xs * Z) + Cr @ theta)
        numpyro.sample("R_obs", dist.Normal(muR, sR), obs=Yret)
        muV = aV[coin_idx] + bv[coin_idx] * M + cv_dir * Xs + Cv @ phi
        numpyro.sample("V_obs", dist.Normal(muV, sV), obs=Yvol)

        # ---- estimands (population level) ----
        A, B, D, BV = mu_slope[0], mu_slope[1], mu_slope[2], mu_slope[3]
        numpyro.deterministic("a_path", A)
        numpyro.deterministic("b_path", B)
        numpyro.deterministic("d_modz", D)
        numpyro.deterministic("bv_path", BV)
        numpyro.deterministic("index_modmed", A * D)            # a·d
        numpyro.deterministic("IE_ret_zneg1", A * (B - D))
        numpyro.deterministic("IE_ret_z0", A * B)
        numpyro.deterministic("IE_ret_zpos1", A * (B + D))
        numpyro.deterministic("IE_vol", A * BV)
    return model


# ============================================================ frequentist companion
def freq_companion(df, Xs, log):
    import statsmodels.api as sm
    z = df["Z"].values
    zc = (z - z.mean()) / (z.std() + 1e-12)
    M = (df["M"].values - df["M"].mean()) / (df["M"].std() + 1e-12)
    y = (df["Yret"].values - df["Yret"].mean()) / (df["Yret"].std() + 1e-12)
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float).values
    ctrl = np.column_stack([df[["ms_regime_prob_high", "vol_ord", "trail_ret"]].values, coin_d])
    # interaction model: y ~ M + M·Z + Xs + Xs·Z + Z + controls
    X = np.column_stack([M, M * zc, Xs, Xs * zc, zc, ctrl])
    X = sm.add_constant(X, has_constant="add")
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["h"].values})
    log(f"  [freq] M·Z coef = {res.params[2]:+.4f}  (p={res.pvalues[2]:.3g})  "
        f"| Xs·Z coef = {res.params[4]:+.4f} (p={res.pvalues[4]:.3g})")
    # sign-split on Z
    out = {}
    for name, mask in [("bull(Z>median)", zc > np.median(zc)), ("bear(Z<=median)", zc <= np.median(zc))]:
        Xs_s = sm.add_constant(np.column_stack([M[mask], Xs[mask], ctrl[mask]]), has_constant="add")
        r = sm.OLS(y[mask], Xs_s).fit(cov_type="cluster", cov_kwds={"groups": df["h"].values[mask]})
        log(f"  [freq] {name}: M→ret coef = {r.params[1]:+.4f} (p={r.pvalues[1]:.3g})  n={int(mask.sum())}")
        out[name] = (float(r.params[1]), float(r.pvalues[1]))
    return out


# ============================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=COINS)
    ap.add_argument("--W", type=int, default=24, help="content window (h)")
    ap.add_argument("--m", type=int, default=4, help="mediator/activity window (h)")
    ap.add_argument("--H", type=int, default=24, help="outcome horizon end (h); outcome over [t+m,t+H)")
    ap.add_argument("--z-win", type=int, default=168, help="ambient-direction trailing window (h)")
    ap.add_argument("--min-items", type=int, default=10)
    ap.add_argument("--min-posts", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--samples", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    log = print

    if args.smoke:
        args.coins = ["BTC", "ETH", "SOL"]
        args.warmup, args.samples, args.chains = 200, 200, 2

    log(f"assembling panel: coins={args.coins} W={args.W}h m={args.m}h H={args.H}h "
        f"Z=trailing{args.z_win}h BTC ret")
    df = assemble(args.coins, args.W, args.m, args.H, args.min_items, args.min_posts, args.z_win)
    log(f"  pooled bars: {len(df):,}  (train {int(df['is_train'].sum()):,} / "
        f"test {int((~df['is_train']).sum()):,})")

    Xs = frozen_content_score(df)
    log(f"  frozen content score: corr(Xs, M)={np.corrcoef(Xs, df['M'])[0,1]:+.3f}")

    # standardise for the path model (effects reported in SD units)
    def zsc(v):
        v = np.asarray(v, float); return (v - v.mean()) / (v.std() + 1e-12)
    Zc = zsc(df["Z"].values)
    Mc = zsc(df["M"].values)
    Yr = zsc(df["Yret"].values)
    Yv = zsc(df["Yvol"].values)
    reg = zsc(df["ms_regime_prob_high"].values)
    vol = zsc(df["vol_ord"].values)
    tr = zsc(df["trail_ret"].values)
    coins_u = sorted(df["coin"].unique())
    cidx = df["coin"].map({c: i for i, c in enumerate(coins_u)}).values.astype(int)

    # Mundlak coin-means (of the time-varying regressors that enter the structural paths)
    g = df.assign(_Xs=Xs, _M=Mc).groupby("coin")
    Xs_bar = df["coin"].map(g["_Xs"].mean()).values
    M_bar = df["coin"].map(g["_M"].mean()).values

    # control matrices
    Cm = np.column_stack([reg, vol, tr, Xs_bar])
    Cr = np.column_stack([reg, vol, tr, Zc, Xs_bar, M_bar])     # Z main effect lives here
    Cv = np.column_stack([reg, vol, tr, Xs_bar, M_bar])

    # ---- frequentist companion (triangulation) ----
    log("frequentist companion (reduced-form interaction + sign-split):")
    split = freq_companion(df, Xs, log)

    # ---- Bayesian NUTS ----
    import jax
    import numpyro
    from numpyro.infer import MCMC, NUTS
    import arviz as az
    numpyro.set_host_device_count(args.chains)

    model = build_model()
    kernel = NUTS(model, target_accept_prob=0.9)
    mcmc = MCMC(kernel, num_warmup=args.warmup, num_samples=args.samples,
                num_chains=args.chains, chain_method="parallel", progress_bar=True)
    mcmc.run(jax.random.PRNGKey(args.seed),
             coin_idx=cidx, n_coins=len(coins_u), Xs=Xs, Z=Zc, M=Mc,
             Cm=Cm, Cr=Cr, Cv=Cv, Yret=Yr, Yvol=Yv)

    idata = az.from_numpyro(mcmc)
    effect_vars = ["a_path", "b_path", "d_modz", "bv_path", "index_modmed",
                   "IE_ret_zneg1", "IE_ret_z0", "IE_ret_zpos1", "IE_vol"]
    summ = az.summary(idata, var_names=effect_vars, hdi_prob=0.95)
    post = idata.posterior
    # posterior P(effect > 0) for the directional quantities
    pgt0 = {v: float((post[v].values > 0).mean()) for v in effect_vars}
    summ["P(>0)"] = [pgt0[v] for v in summ.index]
    summ.to_csv(OUT / "mediation_bayes_effects.csv")
    log("\nposterior effect summary:\n" + summ.to_string())

    # divergences / convergence
    ndiv = int(idata.sample_stats["diverging"].values.sum()) if "diverging" in idata.sample_stats else -1
    maxrhat = float(az.rhat(idata, var_names=effect_vars).to_array().max())
    log(f"\ndiagnostics: divergences={ndiv}  max R-hat(effects)={maxrhat:.3f}")

    # ---- confound sensitivity (Imai-style): IE_ret(z0) under assumed corr(ε_M,ε_R)=ρ ----
    # bias in b ≈ ρ·σR/σM (standardised → σR≈σV≈1 ⇒ b(ρ)=b_naive−ρ); IE(ρ)=a·b(ρ)
    a_draws = post["a_path"].values.ravel()
    b_draws = post["b_path"].values.ravel()
    sM_draws = post["sigma_M"].values.ravel()
    sR_draws = post["sigma_R"].values.ravel()
    rows = []
    for rho in np.round(np.arange(-0.30, 0.301, 0.05), 2):
        b_adj = b_draws - rho * (sR_draws / sM_draws)
        ie = a_draws * b_adj
        rows.append({"rho": rho, "IE_ret_z0_mean": float(ie.mean()),
                     "hdi_lo": float(np.quantile(ie, 0.025)),
                     "hdi_hi": float(np.quantile(ie, 0.975)),
                     "P(>0)": float((ie > 0).mean())})
    sens = pd.DataFrame(rows)
    sens.to_csv(OUT / "mediation_bayes_sensitivity.csv", index=False)

    try:
        idata.to_netcdf(OUT / "mediation_bayes_idata.nc", engine="h5netcdf")
    except Exception as ex:
        log(f"(idata netcdf save failed: {ex}); falling back to pickle")
        import pickle
        with open(OUT / "mediation_bayes_idata.pkl", "wb") as fh:
            pickle.dump(idata, fh)

    # ---- write summary.md ----
    md = ["# Stage 3.5 — hierarchical Bayesian moderated mediation", "",
          f"Cell: content W={args.W}h → activity m={args.m}h → outcome [t+m,t+H), H={args.H}h; "
          f"moderator Z = standardised trailing {args.z_win}h BTC return (predetermined). "
          f"Pooled bars n={len(df):,} ({len(coins_u)} coins), non-overlapping at step={args.H}h. "
          "Effects in SD units. Content scalar Xs = frozen pre-2023 narrative→activity map.", "",
          f"NUTS: {args.chains}×{args.samples} draws ({args.warmup} warmup); "
          f"divergences={ndiv}; max R-hat(effects)={maxrhat:.3f}.", "",
          "## Posterior effect summary", "", "```", summ.to_string(), "```", "",
          "**Reading:** `index_modmed = a·d` is the test that amplification is real (≠0 ⇒ the "
          "content→activity→return effect flips sign with ambient direction). `IE_ret_z{neg1,0,"
          "pos1}` are the conditional indirect effects at Z = −1/0/+1 SD; the pooled null we saw "
          "in Stage 1 corresponds to IE_ret_z0 ≈ 0 while IE_ret_zpos1 and IE_ret_zneg1 carry "
          "opposite signs. `IE_vol = a·bv` is the confirmed magnitude channel.", "",
          "## Frequentist companion (triangulation)", "",
          "| quantity | coef | p |", "|---|---|---|"]
    for k, (cf, pv) in split.items():
        md.append(f"| M→ret slope, {k} | {cf:+.4f} | {pv:.3g} |")
    md += ["", "## Confound sensitivity (IE_ret at Z=0 vs assumed residual corr ρ)", "",
           "```", sens.to_string(index=False), "```", "",
           "How large the unmeasured common-shock correlation ρ(ε_M,ε_R) must be before the "
           "Z=0 indirect effect's sign/credibility changes — the robustness of the mediation "
           "claim to the one assumption (sequential ignorability) we cannot test directly."]
    (OUT / "mediation_bayes_summary.md").write_text("\n".join(md))

    log(f"\n✓ effects      -> {OUT/'mediation_bayes_effects.csv'}")
    log(f"✓ sensitivity  -> {OUT/'mediation_bayes_sensitivity.csv'}")
    log(f"✓ summary      -> {OUT/'mediation_bayes_summary.md'}")
    log(f"✓ idata        -> {OUT/'mediation_bayes_idata.nc'}")


if __name__ == "__main__":
    main()
