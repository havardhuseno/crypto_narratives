#!/usr/bin/env python3
"""
48_s1_groups_and_categories.py
==============================
Follow-on to script 47 (whole-S1 permutation said S1 ~ redundant for whole-sample IC).
Two questions the aggregate cannot answer:

  STEP 2 — WHICH S1 feature groups (if any) does the production model lean on?
    Permute one S1 sub-block at a time (sentiment / market-wide / velocity /
    narrative-taxonomy / signed-rollups), globally across samples, and measure ΔIC.

  STEP 3 — WHERE does S1 contribute? Strip / condition observations by category
    (coin, vol regime, HMM regime, BTC trend, coordination flag) and measure S1's
    contribution WITHIN each subset via within-subgroup permutation. A breakout- or
    regime-specific S1 effect is exactly what a whole-sample permutation washes out.

Two metrics per experiment, because the deployed model trades off BOTH:
  - reg IC   = Pearson(ret_pred, fwd_ret_4h)             (regression head, paper metric)
  - gate IC  = Pearson(prob_up - prob_dn, fwd_ret_4h)    (classification gate = what trades)

Fixed deployed model, no retraining (primary v16wg checkpoint). Reads the deployed
pipeline read-only; writes ONLY under paper_narrative/.

Writes:
  outputs/s1_feature_group_importance.{csv,md}     (step 2)
  outputs/s1_conditional_by_category.{csv,md}       (step 3)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

PAPER_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PAPER_ROOT / "scripts"
OUT_DIR = PAPER_ROOT / "outputs"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# reuse script-47 helpers (module-level, import-safe)
s47 = _load("s47", str(SCRIPTS / "47_s1_contribution_production.py"))

# S1 feature groups: contiguous index ranges into the 31-dim reddit feature axis
# (order fixed by the v16wg config s1_features list).
GROUPS = {
    "sentiment_percoin":  list(range(0, 12)),
    "sentiment_market":   list(range(12, 19)),
    "velocity_weighted":  list(range(19, 24)),
    "narrative_taxonomy": list(range(24, 27)),
    "signed_rollups":     list(range(27, 31)),
}


def ic(pred, true):
    m = ~(np.isnan(pred) | np.isnan(true))
    if m.sum() < 30:
        return float("nan")
    return float(pearsonr(pred[m], true[m])[0])


@torch.no_grad()
def run(model, make_loader, device, reddit_full=None):
    """Forward over a fresh deterministic loader. If reddit_full (N,14,31 CPU) is given,
    it replaces reddit_seq in loader order. Returns (ret_pred, gate_score) numpy arrays."""
    preds, gate, off = [], [], 0
    for b in make_loader():
        bsz = b["reddit_seq"].shape[0]
        bb = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        if reddit_full is not None:
            bb["reddit_seq"] = reddit_full[off:off + bsz].to(device)
        out = model(bb)
        preds.append(out["return_4h"].cpu().float().numpy())
        p = torch.softmax(out["logits_4h"], dim=-1).cpu().float().numpy()
        gate.append(p[:, 2] - p[:, 0])
        off += bsz
    return np.concatenate(preds), np.concatenate(gate)


def global_perm_cols(reddit_orig, cols, seed):
    """Copy of S1 with the given feature columns globally permuted across samples."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(reddit_orig.shape[0])
    out = reddit_orig.clone()
    out[:, :, cols] = reddit_orig[perm][:, :, cols]
    return out


def within_group_perm(reddit_orig, group_labels, seed):
    """Copy of S1 with ALL features permuted WITHIN each subgroup of group_labels."""
    rng = np.random.default_rng(seed)
    out = reddit_orig.clone()
    for g in pd.unique(group_labels):
        idx = np.where(group_labels == g)[0]
        if len(idx) < 2:
            continue
        out[idx] = reddit_orig[idx[rng.permutation(len(idx))]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["cpu", "mps"])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3, help="permutation repeats (step 2)")
    ap.add_argument("--checkpoint", default="primary")
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    log = logging.getLogger("s1g")
    device = torch.device(args.device if (args.device == "cpu" or torch.backends.mps.is_available())
                          else "cpu")
    print(f"device: {device}")

    cfg = json.load(open(s47.PRIMARY_CFG))
    s16 = s47.configure_module(s47.load_v16(), cfg)
    print("S1 groups (index -> feature):")
    for gname, cols in GROUPS.items():
        print(f"  {gname:20s} [{cols[0]:>2}..{cols[-1]:>2}]  "
              f"{[s16.S1_FEATURES[c] for c in cols]}")

    print("\nbuilding test dataset …")
    test_ds = s16.CryptoFusionDataset("test", log)
    n = len(test_ds)
    idxs = list(range(min(args.smoke, n))) if args.smoke else list(range(n))

    from torch.utils.data import DataLoader, Subset
    def make_loader():
        return DataLoader(Subset(test_ds, idxs), batch_size=args.batch_size,
                          shuffle=False, num_workers=0, collate_fn=s16.collate_fn)

    # pre-pass: targets + S1 stream + meta (loader order)
    fwd_true, reddit_all = [], []
    for b in make_loader():
        fwd_true.append(b["fwd_ret_4h"].cpu().float().numpy())
        reddit_all.append(b["reddit_seq"].clone())
    fwd_true = np.concatenate(fwd_true)
    reddit_orig = torch.cat(reddit_all, dim=0)
    del reddit_all
    meta = test_ds.panel.iloc[idxs][[
        "coin", "vol_regime", "ms_regime_prob_high", "btc_trend",
        "coordination_flag"]].reset_index(drop=True)
    print(f"  N={reddit_orig.shape[0]:,}")

    # model (deployed primary)
    stem = s47.CHECKPOINTS[args.checkpoint]
    model = s47.build_model(s16, cfg, s47.MODELS / f"{stem}.pt", device)
    base_reg, base_gate = run(model, make_loader, device)
    base_reg_ic = ic(base_reg, fwd_true)
    base_gate_ic = ic(base_gate, fwd_true)
    print(f"\nbaseline ({args.checkpoint}): reg IC={base_reg_ic:.4f}  gate IC={base_gate_ic:.4f}")

    # ========================= STEP 2: per-feature-group =========================
    print("\n=== STEP 2: per-S1-group permutation ΔIC ===")
    g_rows = []
    experiments = {**{g: cols for g, cols in GROUPS.items()},
                   "ALL_S1": list(range(31))}
    for gname, cols in experiments.items():
        dr, dg = [], []
        for r in range(args.repeats):
            rf = global_perm_cols(reddit_orig, cols, seed=100 + r)
            p_reg, p_gate = run(model, make_loader, device, reddit_full=rf)
            dr.append(base_reg_ic - ic(p_reg, fwd_true))
            dg.append(base_gate_ic - ic(p_gate, fwd_true))
        g_rows.append({"group": gname, "n_feats": len(cols),
                       "d_reg_ic_mean": float(np.mean(dr)), "d_reg_ic_std": float(np.std(dr)),
                       "d_gate_ic_mean": float(np.mean(dg)), "d_gate_ic_std": float(np.std(dg))})
        print(f"  {gname:20s} ΔregIC={np.mean(dr):+.4f}±{np.std(dr):.4f}   "
              f"ΔgateIC={np.mean(dg):+.4f}±{np.std(dg):.4f}")
    g_df = pd.DataFrame(g_rows)
    g_df.to_csv(OUT_DIR / "s1_feature_group_importance.csv", index=False)

    # ========================= STEP 3: strip-by-category =========================
    print("\n=== STEP 3: conditional S1 contribution by category (within-subgroup permute) ===")
    # build categorical partitions
    parts = {}
    parts["coin"] = meta["coin"].astype(str).to_numpy()
    parts["vol_regime"] = meta["vol_regime"].astype(str).to_numpy()
    msr = meta["ms_regime_prob_high"].to_numpy()
    med = np.nanmedian(msr)
    parts["hmm_regime"] = np.where(msr >= med, "high", "low")
    bt = meta["btc_trend"].to_numpy()
    parts["btc_trend"] = np.where(bt >= 0.5, "bull", "bear")
    parts["coordination_flag"] = np.where(meta["coordination_flag"].to_numpy() >= 0.5,
                                          "coord", "none")

    c_rows = []
    for pname, labels in parts.items():
        # one within-subgroup permuted pass (avg over a couple repeats for stability)
        reg_perm = np.zeros((args.repeats, len(fwd_true)))
        gate_perm = np.zeros((args.repeats, len(fwd_true)))
        for r in range(args.repeats):
            rf = within_group_perm(reddit_orig, labels, seed=200 + r)
            reg_perm[r], gate_perm[r] = run(model, make_loader, device, reddit_full=rf)
        for g in pd.unique(labels):
            m = labels == g
            if m.sum() < 100:
                continue
            b_reg = ic(base_reg[m], fwd_true[m])
            b_gate = ic(base_gate[m], fwd_true[m])
            dreg = np.mean([b_reg - ic(reg_perm[r][m], fwd_true[m]) for r in range(args.repeats)])
            dgate = np.mean([b_gate - ic(gate_perm[r][m], fwd_true[m]) for r in range(args.repeats)])
            c_rows.append({"partition": pname, "group": str(g), "n": int(m.sum()),
                           "base_reg_ic": b_reg, "d_reg_ic": float(dreg),
                           "base_gate_ic": b_gate, "d_gate_ic": float(dgate)})
            print(f"  {pname:18s} {str(g):10s} n={int(m.sum()):>6}  "
                  f"baseReg={b_reg:+.3f} ΔregIC={dreg:+.4f}  "
                  f"baseGate={b_gate:+.3f} ΔgateIC={dgate:+.4f}")
    c_df = pd.DataFrame(c_rows)
    c_df.to_csv(OUT_DIR / "s1_conditional_by_category.csv", index=False)

    # ---- summaries ----
    md2 = [
        f"# Step 2 — which S1 groups the production model leans on ({args.checkpoint})",
        "",
        f"Baseline: reg IC={base_reg_ic:.4f}, gate IC={base_gate_ic:.4f} (n={len(fwd_true):,}, "
        "2023+ test). ΔIC = baseline − IC after globally permuting that group's columns "
        f"(mean ± std over {args.repeats} shuffles). Positive ΔIC = the model relies on the group.",
        "",
        "| S1 group | #feat | Δ reg IC | Δ gate IC |",
        "|---|---|---|---|",
    ]
    for _, r in g_df.iterrows():
        md2.append(f"| {r.group} | {int(r.n_feats)} | {r.d_reg_ic_mean:+.4f} ± {r.d_reg_ic_std:.4f} | "
                   f"{r.d_gate_ic_mean:+.4f} ± {r.d_gate_ic_std:.4f} |")
    (OUT_DIR / "s1_feature_group_importance.md").write_text("\n".join(md2))

    md3 = [
        f"# Step 3 — where S1 contributes: conditional ΔIC by category ({args.checkpoint})",
        "",
        "Within each subset, S1's contribution = baseline subset IC − IC after permuting S1 "
        "WITHIN that subset (so a regime/breakout-specific effect is not diluted by the rest of "
        f"the sample). Averaged over {args.repeats} within-subgroup shuffles. Two metrics: reg "
        "(regression head) and gate (prob_up−prob_dn, what actually trades).",
        "",
        "| partition | group | n | base reg IC | Δ reg IC | base gate IC | Δ gate IC |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in c_df.iterrows():
        md3.append(f"| {r.partition} | {r.group} | {int(r.n)} | {r.base_reg_ic:+.3f} | "
                   f"{r.d_reg_ic:+.4f} | {r.base_gate_ic:+.3f} | {r.d_gate_ic:+.4f} |")
    md3 += ["", "Interpretation: look for subsets where Δ reg IC or Δ gate IC is materially "
            "positive (S1 helps there) even though the whole-sample permutation (~0 in step 1) "
            "hides it. That is the per-category heterogeneity / conditional-S1 hypothesis."]
    (OUT_DIR / "s1_conditional_by_category.md").write_text("\n".join(md3))

    print(f"\n✓ {OUT_DIR/'s1_feature_group_importance.md'}")
    print(f"✓ {OUT_DIR/'s1_conditional_by_category.md'}")


if __name__ == "__main__":
    main()
