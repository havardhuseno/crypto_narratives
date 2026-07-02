#!/usr/bin/env python3
"""
47_s1_contribution_production.py
================================
ENGINEERING track (not the QF-paper decomposition): does the Reddit **S1 modality as a
WHOLE** contribute to the deployed production fusion model (v16wg)? The earlier
"S1 adds nothing" was a *leave-one-out (LOO) ablation artifact* — LOO measures
incremental value GIVEN all other streams, so a stream correlated with price reads ~0
even with real standalone signal. Redundant != useless.

STEP 1 (this script): the **fixed-model** view. Take the deployed v16wg checkpoint(s)
AS-IS (no retraining) and measure how much the model leans on S1 via
**permutation importance**: shuffle the entire reddit_seq stream across samples (breaking
the S1<->target alignment) and measure the drop in test IC@4h. Per the method note,
permutation on the fixed model is the VALID fixed-model metric ("how much this model
leans on S1"); zeroing the input is reported too but only as an illustration (zeroing a
fixed model is not a clean attribution because the model never saw all-zero S1).
We also read the learned **stream-attention weight** on S1 (corroboration).

Multi-checkpoint: primary (seed 42) + seed1/2/3 so the reliance is shown to be
seed-robust, not a single-fit fluke (retrain-seed IC band is 0.219-0.271).

Leak / boundary: READS the deployed pipeline (../scripts, ../models, ../data) but writes
ONLY under paper_narrative/. Imports the v16 trainer module purely to reconstruct the
model + test dataset (read-only use); never edits it.

Reads (read-only, main repo):
  scripts/recency_17b_fusion_model_v16.py            (model + dataset classes)
  models/fusion_model_run_10p_v16wg_*.pt + _config.json
  data/panel/full_panel_4h_labeled_recency_v3w.parquet, full_panel_v12_signed_filtered.parquet
  data/processed/kronos/kronos_4h_panel.parquet, data/panel/onchain_*.parquet

Writes (paper_narrative/):
  outputs/s1_contribution_step1.csv     per-checkpoint baseline IC, ΔIC(perm/zero), w_S1
  outputs/s1_contribution_step1.md      human summary + interpretation
  data/s1_eval/baseline_preds.parquet   per-sample baseline ret_pred + meta (reuse 2/3)

Usage:
  python3 47_s1_contribution_production.py                       # all 4 ckpts, full test
  python3 47_s1_contribution_production.py --device mps
  python3 47_s1_contribution_production.py --smoke 3000          # quick wiring check
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

V16_SCRIPT = PROJECT_ROOT / "scripts" / "recency_17b_fusion_model_v16.py"
MODELS = PROJECT_ROOT / "models"

OUT_DIR = PAPER_ROOT / "outputs"
EVAL_DIR = PAPER_ROOT / "data" / "s1_eval"

RUN = "run_10p_v16wg"
CHECKPOINTS = {                       # tag -> checkpoint stem under models/
    "primary": "fusion_model_run_10p_v16wg_recency_volfamily_3head_hmm",
    "seed1":   "fusion_model_run_10p_v16wg_seed1",
    "seed2":   "fusion_model_run_10p_v16wg_seed2",
    "seed3":   "fusion_model_run_10p_v16wg_seed3",
}
PRIMARY_CFG = MODELS / "fusion_model_run_10p_v16wg_recency_volfamily_3head_hmm_config.json"


# ------------------------------------------------------------------ module loading
def load_v16():
    """Import the v16 trainer module without firing its __main__ guard."""
    spec = importlib.util.spec_from_file_location("s16_v16", V16_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["s16_v16"] = mod
    spec.loader.exec_module(mod)
    return mod


def configure_module(s16, cfg):
    """Set module-level flags + panel paths from the v16wg config BEFORE building the
    dataset/model (FusionModel reads ENABLE_RECENCY etc. from globals at construction)."""
    s16.ENABLE_2H_HEAD = bool(cfg.get("enable_2h_head", False))
    s16.ENABLE_S5_NER = bool(cfg.get("enable_s5_ner", False))
    s16.ENABLE_S1_SIGNED_ROLLUPS = bool(cfg.get("enable_s1_signed_rollups", False))
    s16.ENABLE_S1_CODEBOOK = bool(cfg.get("enable_s1_codebook", False))
    s16.ENABLE_S3_FEATURE_AGE = bool(cfg.get("enable_s3_feature_age", False))
    s16.ENABLE_RECENCY = bool(cfg.get("enable_recency", False))

    if s16.ENABLE_S1_SIGNED_ROLLUPS:
        s16.S1_FEATURES = list(s16.S1_FEATURES) + [
            c for c in s16.S1_SIGNED_ROLLUP_FEATURES if c not in s16.S1_FEATURES]
    if s16.ENABLE_S3_FEATURE_AGE:
        s16.S3_FEATURES = list(s16.S3_FEATURES) + [
            c for c in s16.S3_FEATURE_AGE_FEATURES if c not in s16.S3_FEATURES]

    # panel overrides (config points at the recency v3w 4h panel + v12 signed reddit panel)
    s16.PANEL_4H = PROJECT_ROOT / cfg["panel_4h"]
    s16.PANEL_REDDIT = PROJECT_ROOT / cfg["reddit_panel"]

    assert len(s16.S1_FEATURES) == cfg["s1_input_dim"], (
        f"S1 feature count {len(s16.S1_FEATURES)} != config s1_input_dim {cfg['s1_input_dim']}")
    return s16


def build_model(s16, cfg, ckpt_path, device):
    model = s16.FusionModel(
        n_coins=len(cfg.get("coins", s16.COINS)),
        hidden=cfg.get("hidden_dim", 128),
        coin_emb_dim=cfg.get("coin_emb_dim", 8),
        dropout=0.0,
        s1_input_dim=cfg.get("s1_input_dim"),
        s3_input_dim=cfg.get("s3_input_dim"),
    )
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model


# ------------------------------------------------------------------ eval helpers
def ic(pred, true):
    m = ~(np.isnan(pred) | np.isnan(true))
    if m.sum() < 10:
        return float("nan")
    return float(pearsonr(pred[m], true[m])[0])


@torch.no_grad()
def forward_collect(model, make_loader, device, reddit_perm=None, zero_s1=False,
                    want_attn=False):
    """Run model over a FRESH deterministic loader (memory-frugal: hourly_seq is never
    held in RAM across passes). Optional S1 manipulation:
      reddit_perm: a global permutation array; batch b's reddit_seq is replaced by the
                   permuted rows (requires `reddit_flat` in the enclosing scope).
      zero_s1: zero reddit_seq.
    Returns ret_pred (np), and optionally the S1 attention weight (np)."""
    preds, store = [], []
    handle = None
    if want_attn:
        import torch.nn.functional as F
        def hook(mod, inp, out):
            ctx = inp[1]
            store.append(F.softmax(mod.weight_net(ctx), dim=-1).detach().cpu().float().numpy())
        handle = model.attention.register_forward_hook(hook)

    off = 0
    for b in make_loader():
        bsz = b["reddit_seq"].shape[0]
        bb = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
        if zero_s1:
            bb["reddit_seq"] = torch.zeros_like(bb["reddit_seq"])
        elif reddit_perm is not None:
            sel = reddit_perm[off:off + bsz]
            bb["reddit_seq"] = forward_collect.reddit_flat[sel].to(device)
        out = model(bb)
        preds.append(out["return_4h"].cpu().float().numpy())
        off += bsz
    if handle is not None:
        handle.remove()
    attn = np.concatenate(store)[:, 0] if want_attn else None   # stream 0 = S1/Reddit
    return np.concatenate(preds), attn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--perm-repeats", type=int, default=3,
                    help="permutation-importance repeats (different shuffle seeds)")
    ap.add_argument("--checkpoints", nargs="+", default=list(CHECKPOINTS.keys()))
    ap.add_argument("--smoke", type=int, default=0,
                    help="if >0, use only the first N test samples (wiring check)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    log = logging.getLogger("s1")

    device = torch.device(args.device if (args.device == "cpu" or torch.backends.mps.is_available())
                          else "cpu")
    print(f"device: {device}")

    cfg = json.load(open(PRIMARY_CFG))
    s16 = configure_module(load_v16(), cfg)
    print(f"v16 module configured: recency={s16.ENABLE_RECENCY} 2h={s16.ENABLE_2H_HEAD} "
          f"signed_rollups={s16.ENABLE_S1_SIGNED_ROLLUPS}  |S1|={len(s16.S1_FEATURES)}")
    print(f"PANEL_4H={s16.PANEL_4H.name}  PANEL_REDDIT={s16.PANEL_REDDIT.name}")

    # ---- build test dataset once ----
    print("\nbuilding test dataset …")
    t0 = time.time()
    test_ds = s16.CryptoFusionDataset("test", log)
    n = len(test_ds)
    idxs = list(range(min(args.smoke, n))) if args.smoke else list(range(n))
    print(f"  n_test={n:,}  (using {len(idxs):,})  built in {time.time()-t0:.1f}s")

    # per-sample meta for steps 2/3 (and conditional reporting)
    meta = test_ds.panel.iloc[idxs][[
        "coin", "ts", "vol_regime", "ms_regime_prob_high", "btc_trend",
        "coordination_flag"]].reset_index(drop=True)

    # fresh deterministic loader factory (frugal: rebuild per pass, never hold hourly_seq)
    from torch.utils.data import DataLoader, Subset
    def make_loader():
        return DataLoader(Subset(test_ds, idxs), batch_size=args.batch_size,
                          shuffle=False, num_workers=0, collate_fn=s16.collate_fn)

    # one pre-pass: collect targets + the (small) S1 stream for permutation
    print("pre-pass: collecting targets + S1 stream …")
    t0 = time.time()
    fwd_true, reddit_all = [], []
    for b in make_loader():
        fwd_true.append(b["fwd_ret_4h"].cpu().float().numpy())
        reddit_all.append(b["reddit_seq"].clone())     # CPU; (b,14,31)
    fwd_true = np.concatenate(fwd_true)
    forward_collect.reddit_flat = torch.cat(reddit_all, dim=0)   # (N,14,31) CPU, ~0.1 GB
    del reddit_all
    print(f"  {forward_collect.reddit_flat.shape[0]:,} samples in {time.time()-t0:.1f}s")

    rows = []
    baseline_saved = False
    for tag in args.checkpoints:
        stem = CHECKPOINTS[tag]
        ckpt_path = MODELS / f"{stem}.pt"
        if not ckpt_path.exists():
            print(f"  [skip] {tag}: {ckpt_path.name} missing")
            continue
        print(f"\n=== checkpoint: {tag} ({stem}) ===")
        model = build_model(s16, cfg, ckpt_path, device)

        # baseline + attention weight
        t0 = time.time()
        base_pred, w_s1 = forward_collect(model, make_loader, device, want_attn=True)
        base_ic = ic(base_pred, fwd_true)
        print(f"  baseline IC@4h = {base_ic:.4f}   mean w_S1(attn) = {w_s1.mean():.4f}   "
              f"({time.time()-t0:.0f}s)")

        # zero-S1 (illustration only)
        zero_pred, _ = forward_collect(model, make_loader, device, zero_s1=True)
        zero_ic = ic(zero_pred, fwd_true)

        # permutation importance (valid fixed-model metric)
        N = forward_collect.reddit_flat.shape[0]
        drops = []
        for r in range(args.perm_repeats):
            rng = np.random.default_rng(1000 + r)
            perm = rng.permutation(N)
            p_pred, _ = forward_collect(model, make_loader, device, reddit_perm=perm)
            drops.append(base_ic - ic(p_pred, fwd_true))
        perm_drop_mean = float(np.mean(drops))
        perm_drop_std = float(np.std(drops))
        print(f"  ΔIC permute-S1 = {perm_drop_mean:+.4f} ± {perm_drop_std:.4f}   "
              f"ΔIC zero-S1 = {base_ic - zero_ic:+.4f}")

        rows.append({
            "checkpoint": tag, "n": int(len(fwd_true)),
            "baseline_ic": base_ic, "w_s1_attn_mean": float(w_s1.mean()),
            "ic_zero_s1": zero_ic, "delta_ic_zero_s1": base_ic - zero_ic,
            "delta_ic_perm_s1_mean": perm_drop_mean,
            "delta_ic_perm_s1_std": perm_drop_std,
        })

        if not baseline_saved:
            out = meta.copy()
            out["ret_pred_baseline"] = base_pred
            out["fwd_ret_4h"] = fwd_true
            out["w_s1_attn"] = w_s1
            out.to_parquet(EVAL_DIR / "baseline_preds.parquet", index=False)
            baseline_saved = True
        del model

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "s1_contribution_step1.csv", index=False)

    # ---- summary md ----
    lines = [
        "# S1 (Reddit) contribution to the production fusion model — Step 1 (fixed model)",
        "",
        "Deployed **v16wg** checkpoint(s), no retraining. IC@4h = Pearson(ret_pred, fwd_ret_4h) "
        f"on the 2024+ test split (n={int(df['n'].iloc[0]):,}).",
        "",
        "- **ΔIC permute-S1** = baseline IC − IC after globally shuffling the *entire* reddit_seq "
        "stream across samples. This is the valid fixed-model reliance metric: how much the "
        "model's return ranking degrades when S1 is destroyed but all other streams are intact.",
        "- **ΔIC zero-S1** = baseline IC − IC with reddit_seq set to 0 (illustration only; zeroing "
        "a fixed model is off-distribution, so permutation is preferred).",
        "- **w_S1(attn)** = mean learned stream-attention weight on S1 (of 3 streams).",
        "",
        "| ckpt | baseline IC | w_S1 attn | ΔIC permute-S1 | ΔIC zero-S1 |",
        "|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {r.checkpoint} | {r.baseline_ic:.4f} | {r.w_s1_attn_mean:.4f} | "
                     f"{r.delta_ic_perm_s1_mean:+.4f} ± {r.delta_ic_perm_s1_std:.4f} | "
                     f"{r.delta_ic_zero_s1:+.4f} |")
    lines += [
        "",
        f"Across {len(df)} checkpoint(s): mean ΔIC permute-S1 = "
        f"**{df['delta_ic_perm_s1_mean'].mean():+.4f}**, mean w_S1 attn = "
        f"{df['w_s1_attn_mean'].mean():.4f}.",
        "",
        "Interpretation: a materially positive ΔIC permute-S1 (relative to its repeat-std and "
        "seed spread) means the deployed model *does* lean on S1 — i.e. S1 contributes in the "
        "fixed-model sense, even though LOO-retrain called it redundant. Next: step 2 isolates "
        "*which* S1 feature groups carry the reliance; step 3 strips observations by category "
        "(regime / vol / breakout / coin) to find *where* S1 helps.",
    ]
    (OUT_DIR / "s1_contribution_step1.md").write_text("\n".join(lines))
    print(f"\n✓ {OUT_DIR/'s1_contribution_step1.csv'}")
    print(f"✓ {OUT_DIR/'s1_contribution_step1.md'}")
    print(f"✓ {EVAL_DIR/'baseline_preds.parquet'}")


if __name__ == "__main__":
    main()
