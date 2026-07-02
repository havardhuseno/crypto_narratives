#!/usr/bin/env python3
"""
49_s1_standalone_and_loo_retrain.py
===================================
ENGINEERING track, the OTHER bracket. Steps 1-3 (scripts 47/48) used *permutation
importance on the fixed deployed model* — that measures how much the AS-TRAINED v16wg
LEANS on S1. It cannot detect signal that S1 carries but which is REDUNDANT with the
price/volume family the model already saturates on (the classic LOO artifact:
redundant != useless). This script supplies the complementary, retrain-based bracket:

  (A) full        — retrain the v16wg architecture from scratch under THIS harness/seed.
                     Control: confirms the harness reproduces ~0.28 and gives the matched
                     baseline that (B) is compared against (NOT the deployed ckpt, so the
                     LOO delta is not confounded by harness/seed differences).
  (B) without_s1  — retrain with the S1 (reddit_seq) input ZEROED during BOTH train and
                     eval. The model adapts (it never relies on S1), so test-IC(full) -
                     test-IC(without_s1) = the value LOST if S1 had never existed
                     = S1's true incremental contribution. (Valid LOO: zero-at-train,
                     not zero-at-inference-on-a-fixed-model.)
  (C) s1_only     — retrain with EVERYTHING zeroed except S1 (reddit_seq) and the coin
                     embedding (= coin fixed effects). test-IC here = S1's STANDALONE
                     predictive value, controlling only for coin FE. This is the number
                     the "I think S1 still contributes" hypothesis predicts is > 0.

(B) and (C) bracket the truth: (C) is S1's own signal floor; full-(B) is its incremental
ceiling given the rest of the model. If both are ~0, S1 genuinely carries nothing the
model can use at 4h; if (C) > 0 but full-(B) ~0, S1 has standalone signal that is fully
redundant with price (publishable nuance, not "useless").

Test setup (verified against paper1_methodology.md §16-17 + CLAUDE.md): leak-free panels
full_panel_4h_labeled_recency_v3w + full_panel_v12_signed_filtered, authoritative panel
`regime` split (train 50,539 / val 17,520 / test 58,875 = ts>=2023). IC@4h = pooled
Pearson(return_4h, fwd_ret_4h) over valid test rows — the SAME metric the production
trainer reports (recency_17b_fusion_model_v16.py:1174), which reproduces 0.282. This is
the corrected (post-leak-fix) setup, NOT the inflated v14 (0.296) or the 2024+ split.

Boundary: READS ../scripts + ../data + ../models (read-only); imports the v16 trainer
module to reuse FusionModel / CryptoFusionDataset / train_epoch / evaluate. WRITES ONLY
under paper_narrative/ (checkpoints -> paper_narrative/models/, results -> outputs/).
Never saves kronos stats to ../models (that production-only side effect is skipped).

Usage:
  python3 49_s1_standalone_and_loo_retrain.py --device mps                 # all 3 modes
  python3 49_s1_standalone_and_loo_retrain.py --device mps --modes full without_s1
  python3 49_s1_standalone_and_loo_retrain.py --smoke 3000 --epochs 2      # wiring check
"""
from __future__ import annotations

import argparse
import importlib.util
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PAPER_ROOT / "scripts"

OUT_DIR = PAPER_ROOT / "outputs"
PAPER_MODELS = PAPER_ROOT / "models"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# reuse module loader / configurator from script 47
s47 = _load("s47", str(SCRIPTS / "47_s1_contribution_production.py"))

# ------------------------------------------------------------------ masking specs
# Keys to ZERO (during train AND eval) for each ablation mode. Inputs are standardized,
# so a zero column == the neutral/mean value == "no information from this stream".
#   full        : nothing zeroed (control)
#   without_s1  : zero S1 stream only (LOO retrain)
#   s1_only     : zero everything EXCEPT reddit_seq + coin_idx (coin FE). This isolates
#                 S1's standalone signal controlling for coin fixed effects.
ALL_NON_S1 = [
    "hourly_seq",   # S2 market-structure + S4 Kronos (TCN)
    "onchain_seq",  # S3 on-chain (LSTM)
    "ner_seq",      # S5 (off in v16wg, but zero for safety)
    "recency_vec",  # S6 recency side-channel
    "vol_ohe", "ms_regime", "btc_trend", "macro_ctx",  # attention/fusion context
    "coord_flag",   # coordination flag
]
MASK_SPECS = {
    "full":       [],
    "without_s1": ["reddit_seq"],
    "s1_only":    list(ALL_NON_S1),   # keep reddit_seq + coin_idx
}


def wrap_masked_forward(model, mask_keys):
    """Monkeypatch model.forward to zero `mask_keys` before the real forward. train_epoch
    and evaluate both call model(batch) AFTER moving the batch to device, so zeros_like
    keeps device/dtype. No-op when mask_keys is empty."""
    if not mask_keys:
        return
    orig = model.forward

    def fwd(batch):
        if mask_keys:
            b = dict(batch)
            for k in mask_keys:
                if k in b and torch.is_tensor(b[k]):
                    b[k] = torch.zeros_like(b[k])
            batch = b
        return orig(batch)

    model.forward = fwd


# ------------------------------------------------------------------ training harness
def build_datasets(s16, log, smoke=0):
    train_ds = s16.CryptoFusionDataset("train", log)
    val_ds   = s16.CryptoFusionDataset("val",   log)
    test_ds  = s16.CryptoFusionDataset("test",  log)
    if smoke:
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(min(smoke, len(train_ds)))))
        val_ds   = Subset(val_ds,   list(range(min(smoke, len(val_ds)))))
        test_ds  = Subset(test_ds,  list(range(min(smoke, len(test_ds)))))
    return train_ds, val_ds, test_ds


def class_weights(s16, train_ds, device):
    n = min(len(train_ds), 5000)
    labels = [int(train_ds[i]["label_4h"].item()) for i in range(n)]
    counts = np.bincount(labels, minlength=3).astype(float)
    total = counts.sum()
    return torch.tensor([total / (3 * c) if c > 0 else 1.0 for c in counts],
                        dtype=torch.float32, device=device)


def train_mode(s16, mode, datasets, device, args, log):
    """Train one ablation mode under the production harness; return final test/val IC."""
    train_ds, val_ds, test_ds = datasets

    # deterministic per-mode seed (same seed across modes for a clean comparison)
    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    from torch.utils.data import DataLoader
    kw = dict(num_workers=0, collate_fn=s16.collate_fn)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, **kw)

    model = s16.FusionModel().to(device)
    wrap_masked_forward(model, MASK_SPECS[mode])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    cw = class_weights(s16, train_ds, device)
    ce_crit = nn.CrossEntropyLoss(weight=cw, label_smoothing=s16.LABEL_SMOOTHING)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=s16.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, eta_min=1e-5)

    ckpt_path = PAPER_MODELS / f"v16wg_retrain_{mode}.pt"
    best_val_ic, best_epoch, patience = -math.inf, 0, 0
    log.info(f"  [{mode}] params={n_params:,}  mask={MASK_SPECS[mode] or 'none'}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        s16.train_epoch(model, train_loader, optim, ce_crit, device)
        val_m, _ = s16.evaluate(model, val_loader, ce_crit, device)
        sched.step()
        val_ic = val_m.get("ic_4h", -math.inf)
        if math.isnan(val_ic):
            val_ic = -math.inf
        flag = ""
        if val_ic > best_val_ic:
            best_val_ic, best_epoch, patience = val_ic, epoch, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_ic": best_val_ic, "mode": mode}, ckpt_path)
            flag = " *"
        else:
            patience += 1
        log.info(f"    e{epoch:02d} val_ic={val_ic:+.4f} (best={best_val_ic:+.4f}"
                 f"@{best_epoch}) {time.time()-t0:.0f}s{flag}")
        if patience >= args.patience:
            log.info(f"    early stop @ e{epoch}")
            break

    # final eval from best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state"])
    wrap_masked_forward(model, MASK_SPECS[mode])  # re-wrap (load_state_dict doesn't touch .forward)
    val_m,  _ = s16.evaluate(model, val_loader,  ce_crit, device)
    test_m, _ = s16.evaluate(model, test_loader, ce_crit, device)
    return {
        "mode": mode, "best_epoch": best_epoch,
        "val_ic_4h": float(val_m.get("ic_4h", float("nan"))),
        "val_rank_ic_4h": float(val_m.get("rank_ic_4h", float("nan"))),
        "test_ic_4h": float(test_m.get("ic_4h", float("nan"))),
        "test_rank_ic_4h": float(test_m.get("rank_ic_4h", float("nan"))),
        "test_ic_2h": float(test_m.get("ic_2h", float("nan"))),
        "n_params": n_params,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--modes", nargs="+", default=["full", "without_s1", "s1_only"],
                    choices=list(MASK_SPECS))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("s49")

    device = torch.device(args.device)
    print(f"device: {device}")
    PAPER_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load + configure the production module exactly as deployed (flags, panels, S1 feats)
    import json
    cfg = json.loads(s47.PRIMARY_CFG.read_text())
    s16 = s47.load_v16()
    s47.configure_module(s16, cfg)
    print(f"v16wg configured: |S1|={len(s16.S1_FEATURES)}  recency={s16.ENABLE_RECENCY}  "
          f"2h={s16.ENABLE_2H_HEAD}  rollups={s16.ENABLE_S1_SIGNED_ROLLUPS}")
    print(f"PANEL_4H={s16.PANEL_4H.name}  PANEL_REDDIT={s16.PANEL_REDDIT.name}")

    print("\nbuilding datasets …")
    datasets = build_datasets(s16, log, smoke=args.smoke)
    print(f"  train={len(datasets[0]):,}  val={len(datasets[1]):,}  test={len(datasets[2]):,}\n")

    rows = []
    for mode in args.modes:
        print(f"=== mode: {mode} ===")
        r = train_mode(s16, mode, datasets, device, args, log)
        rows.append(r)
        print(f"  -> {mode}: test_ic_4h={r['test_ic_4h']:+.4f}  "
              f"val_ic_4h={r['val_ic_4h']:+.4f}  best_epoch={r['best_epoch']}\n")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "s1_retrain_ablation.csv", index=False)

    # brackets (only if the relevant modes were run)
    by = {r["mode"]: r for r in rows}
    lines = ["# S1 retrain brackets — standalone & leave-one-out (v16wg architecture)\n",
             "Retrained under the production harness (same seed/hyperparams), leak-free "
             "panels, authoritative regime split. IC@4h = pooled Pearson(return_4h, "
             "fwd_ret_4h), the production metric.\n",
             "| mode | test IC@4h | test RankIC@4h | val IC@4h | best epoch |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['mode']} | {r['test_ic_4h']:+.4f} | {r['test_rank_ic_4h']:+.4f} "
                     f"| {r['val_ic_4h']:+.4f} | {r['best_epoch']} |")
    lines.append("")
    if "full" in by and "without_s1" in by:
        loo = by["full"]["test_ic_4h"] - by["without_s1"]["test_ic_4h"]
        lines.append(f"**LOO contribution of S1** (full − without_s1) = "
                     f"**{loo:+.4f}** IC@4h — value lost if S1 never existed (incremental).")
    if "s1_only" in by:
        lines.append(f"**S1 standalone IC@4h** (S1 + coin FE only) = "
                     f"**{by['s1_only']['test_ic_4h']:+.4f}** — S1's own predictive floor.")
    lines.append("")
    lines.append("Interpretation: both ~0 -> S1 carries nothing usable at 4h. standalone>0 "
                 "but LOO~0 -> S1 has real signal that is fully redundant with the "
                 "price/volume family (redundant != useless). LOO>0 -> S1 adds incremental value.")
    (OUT_DIR / "s1_retrain_ablation.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\n✓ {OUT_DIR/'s1_retrain_ablation.csv'}")
    print(f"✓ {OUT_DIR/'s1_retrain_ablation.md'}")


if __name__ == "__main__":
    main()
