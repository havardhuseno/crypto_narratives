#!/usr/bin/env python3
"""
67_directional_robustness.py  (v2)
==================================
Stress-test the NEW directional signal (content improves OOS direction). Because the OOS Ridge is
seed-deterministic, the meaningful robustness is: (1) block-bootstrap the OOS dir-IC *improvement*
[content − controls] over time to get a CI / p-value; (2) per-coin dir-IC; (3) sub-period split.

OOS setup mirrors script 58: train pre-2023, test post-2023, standardisation frozen on train,
RidgeCV. Two nested models: controls (coin FE + trailing ret + regime + vol) vs +30 content shares.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

PAPER_ROOT = Path(__file__).resolve().parents[1]; HERE = Path(__file__).resolve().parent
OUT = PAPER_ROOT / "outputs"

def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m58 = _load("d58", "58_conditional_directional.py")
assemble = m58.m53.assemble; SUB30 = m58.SUB30; zsc = m58.zsc
ATTN = m58.ATTENTION; POL = m58.POLARITY

def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))

def main():
    import argparse
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=m58.COINS)
    ap.add_argument("--W", type=int, default=24); ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--H", type=int, default=24); ap.add_argument("--z-win", type=int, default=168)
    ap.add_argument("--n-boot", type=int, default=600); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True); log = print

    df = assemble(args.coins, args.W, args.m, args.H, m58.m53.m46.MIN_ITEMS, 3, args.z_win)
    tr = df["is_train"].values; te = ~tr
    y = df["Yret"].values
    coin_d = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float).values
    def sc(cols): raw = df[cols].values; return StandardScaler().fit(raw[tr]).transform(raw)
    ctrl = sc(["trail_ret", "ms_regime_prob_high", "vol_ord"])
    content = sc([f"share_{s}" for s in SUB30])
    Xc = np.column_stack([coin_d, ctrl])
    Xk = np.column_stack([coin_d, ctrl, content])
    alphas = np.logspace(-2, 4, 25)
    pc = RidgeCV(alphas=alphas).fit(Xc[tr], y[tr]).predict(Xc[te])
    pk = RidgeCV(alphas=alphas).fit(Xk[tr], y[tr]).predict(Xk[te])
    yte = y[te]; hte = df["h"].values[te]; coin_te = df["coin"].values[te]
    ts_te = pd.to_datetime(df["ts"].values[te], utc=True)

    ic_c = spearman(pc, yte); ic_k = spearman(pk, yte)
    log(f"pooled OOS dir-IC: controls={ic_c:+.4f}  +content={ic_k:+.4f}  diff={ic_k-ic_c:+.4f}  "
        f"hit(+content)={float(((pk>0)==(yte>0)).mean()):.4f}")

    # (1) block bootstrap over h-clusters: distribution of IC diff
    cl = {}
    for i, hh in enumerate(hte): cl.setdefault(hh, []).append(i)
    keys = list(cl.values()); rb = np.random.default_rng(args.seed)
    diffs = []
    for _ in range(args.n_boot):
        idx = np.concatenate([keys[j] for j in rb.integers(0, len(keys), len(keys))])
        diffs.append(spearman(pk[idx], yte[idx]) - spearman(pc[idx], yte[idx]))
    diffs = np.array(diffs)
    p_pos = float((diffs > 0).mean())
    log(f"bootstrap IC diff: mean={diffs.mean():+.4f}  CI[{np.quantile(diffs,.025):+.4f},"
        f"{np.quantile(diffs,.975):+.4f}]  P(diff>0)={p_pos:.3f}")

    # (2) per-coin
    rows = []
    for c in args.coins:
        sel = coin_te == c
        if sel.sum() < 50: continue
        rows.append({"coin": c, "n": int(sel.sum()), "IC_controls": spearman(pc[sel], yte[sel]),
                     "IC_content": spearman(pk[sel], yte[sel])})
    pcdf = pd.DataFrame(rows); pcdf["diff"] = pcdf["IC_content"] - pcdf["IC_controls"]
    log("\nper-coin OOS dir-IC:\n" + pcdf.to_string(index=False))

    # (3) sub-period
    for label, mask in [("2023", ts_te.year == 2023), ("2024+", ts_te.year >= 2024)]:
        if mask.sum() < 50: continue
        log(f"sub-period {label}: controls={spearman(pc[mask],yte[mask]):+.4f}  "
            f"+content={spearman(pk[mask],yte[mask]):+.4f}  n={int(mask.sum())}")

    pd.DataFrame({"ic_controls":[ic_c],"ic_content":[ic_k],"diff":[ic_k-ic_c],
                  "boot_diff_mean":[diffs.mean()],"boot_diff_lo":[np.quantile(diffs,.025)],
                  "boot_diff_hi":[np.quantile(diffs,.975)],"P_diff_gt0":[p_pos]}).to_csv(
                  OUT / "directional_robustness.csv", index=False)
    pcdf.to_csv(OUT / "directional_robustness_percoin.csv", index=False)
    log(f"\n✓ -> {OUT/'directional_robustness.csv'} , _percoin.csv")

if __name__ == "__main__":
    main()
