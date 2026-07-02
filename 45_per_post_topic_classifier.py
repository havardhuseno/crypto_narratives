#!/usr/bin/env python3
"""
45_per_post_topic_classifier.py
===============================
Design B: a per-post topic classifier against the FROZEN 34-label codebook, needing
NO clustering at inference. This is the leak-free replacement for the production panel's
monthly `narr_cb_*` shares (which carry a monthly-clustering look-ahead leak: a day-d
post's cluster membership is computed from the whole month incl. future days).

Pipeline
--------
  (1) LABEL TRANSFER. The Phase-C open-coding labels live at CLUSTER granularity
      (coin, month, cluster_id) -> primary_label in {3.1..5.20} (34 classes). Each
      cluster has a saved unit-normalised centroid. We embed every post (MiniLM, the
      same all-MiniLM-L6-v2 / normalize_embeddings=True the pipeline used) and assign it
      the label of its NEAREST centroid *within its own coin-month* (cosine = dot product).
      This reconstructs the historical cluster assignment as per-post pseudo-labels.

      NOTE on leakage: the per-month centroid uses within-month info, so the pseudo-LABELS
      are "monthly". That is fine — labels are only used to TRAIN. The deliverable feature
      is the fitted classifier applied to a post's OWN embedding, which uses no clustering,
      no month aggregation, no future info => leak-free at inference. Returns never enter
      this script at all, so the classifier cannot be a return-leak.

  (2) CLASSIFIER. Multinomial logistic regression on the 384-d embedding -> 34 classes
      (recommended start; swap the head via --head if held-out macro-F1 is weak).
      Validity is reported on a TEMPORAL hold-out (train early months -> test late months)
      so referees see the codebook semantics generalise across time.

  (3) DELIVERABLES (all under paper_narrative/, never touching the pipeline):
        data/embeddings/{coin}.npy           cached unit embeddings (post order)
        data/embeddings/{coin}_index.parquet id, created_utc, has_text (post order)
        data/post_topics/{coin}.parquet      id, created_utc, pred_subtype, pred_proba
        models/topic_classifier_{head}.joblib  fitted clf + label list + meta

Reads  (read-only, main repo):
  data/processed/reddit_clean/{coin}/posts.parquet            (id, created_utc, text_for_labeling)
  data/processed/narratives/{coin}/cluster_samples/cluster_centroids_{coin}_{YYYY}_{MM}.npz
  outputs/open_coding/phase_c_llm_labels.parquet              (coin, month, cluster_id, primary_label)

Usage:
  python 45_per_post_topic_classifier.py            # logreg, all 6 coins
  python 45_per_post_topic_classifier.py --head lgbm
  python 45_per_post_topic_classifier.py --coins ETH BTC --no-refit
"""
from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd

# paper_narrative/scripts/45_*.py -> parents[1]=paper_narrative (write), parents[2]=repo (read-only)
PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

REDDIT = PROJECT_ROOT / "data" / "processed" / "reddit_clean"
NARR = PROJECT_ROOT / "data" / "processed" / "narratives"
PHASE_C = PROJECT_ROOT / "outputs" / "open_coding" / "phase_c_llm_labels.parquet"

EMB_DIR = PAPER_ROOT / "data" / "embeddings"
TOPIC_DIR = PAPER_ROOT / "data" / "post_topics"
MODEL_DIR = PAPER_ROOT / "models"
OUT_DIR = PAPER_ROOT / "outputs"

PANEL_COINS = ["BTC", "ETH", "DOGE", "XRP", "ADA", "SOL"]
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # matches scripts/07_extract_narratives.py:81
TEXT_COL = "text_for_labeling"
TEMPORAL_SPLIT = "2023-07"   # train months < cutoff, test months >= cutoff (validity check)

# Memory-frugal embedding defaults (this machine: 16 GB RAM, heavy swap pressure).
# The killer is peak activation memory, not throughput: long sequences * big batch on a
# full machine thrash swap (observed ~1.5 min/batch). Cap sequence length, keep batches
# small, encode in disk-backed shards so a kill/rerun resumes instead of restarting.
EMBED_BATCH = 64
EMBED_MAX_SEQ = 128      # MiniLM default is 256; topic is evident early -> half the memory
SHARD_POSTS = 20_000     # encode + flush this many text-posts at a time

# 34 codebook subtypes (3.x noise, 4.x operational, 5.x narrative)
CB_LABELS = [f"3.{i}" for i in range(1, 5)] + [f"4.{i}" for i in range(1, 11)] + \
            [f"5.{i}" for i in range(1, 21)]


# ----------------------------------------------------------------------------- embeddings
def _get_model(device: str):
    from sentence_transformers import SentenceTransformer
    import torch
    if device == "auto":
        device = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu")
    print(f"  embedding device: {device}  (batch={EMBED_BATCH}, max_seq={EMBED_MAX_SEQ})")
    m = SentenceTransformer(EMBED_MODEL_NAME, device=device)
    m.max_seq_length = EMBED_MAX_SEQ
    return m


def _cache_exists(coin: str) -> bool:
    return (EMB_DIR / f"{coin}.npy").exists() and (EMB_DIR / f"{coin}_index.parquet").exists()


def embed_coin(coin: str, get_model) -> tuple[np.ndarray, pd.DataFrame]:
    """Embed all text-bearing posts of `coin` (unit vectors), in disk-backed shards so a
    kill/rerun resumes instead of restarting. Returns
    (emb[n_text,384], index_df: id/created_utc/has_text/emb_row).
    Posts with null/empty text get has_text=False and emb_row=-1 (never embedded).
    `get_model` is a 0-arg callable that lazily loads the embedding model only when a
    cache miss forces a (re)compute — so a fully-cached run never touches torch/MiniLM."""
    emb_path = EMB_DIR / f"{coin}.npy"
    idx_path = EMB_DIR / f"{coin}_index.parquet"
    shard_dir = EMB_DIR / f"{coin}_shards"
    if emb_path.exists() and idx_path.exists():
        emb = np.load(emb_path)
        idx = pd.read_parquet(idx_path)
        print(f"  [{coin}] cached embeddings: {emb.shape[0]:,} text posts / {len(idx):,} total")
        return emb, idx
    model = get_model()

    posts = pd.read_parquet(REDDIT / coin / "posts.parquet",
                            columns=["id", "created_utc", TEXT_COL])
    posts = posts.sort_values("created_utc").reset_index(drop=True)
    has_text = (posts[TEXT_COL].notna()
                & (posts[TEXT_COL].astype(str).str.strip().str.len() > 0)).to_numpy()
    txt = posts.loc[has_text, TEXT_COL].astype(str).tolist()
    n = len(txt)
    n_shards = (n + SHARD_POSTS - 1) // SHARD_POSTS
    print(f"  [{coin}] embedding {n:,} text posts of {len(posts):,} in {n_shards} shard(s) …")
    shard_dir.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    for s in range(n_shards):
        sp = shard_dir / f"{s:04d}.npy"
        if sp.exists():
            print(f"    shard {s+1}/{n_shards}: cached, skip")
            continue
        lo, hi = s * SHARD_POSTS, min((s + 1) * SHARD_POSTS, n)
        t0 = time.time()
        e = model.encode(txt[lo:hi], batch_size=EMBED_BATCH, normalize_embeddings=True,
                         show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        np.save(sp, e)
        rate = (hi - lo) / max(time.time() - t0, 1e-6)
        done = hi
        eta = (n - done) / max(rate, 1e-6)
        print(f"    shard {s+1}/{n_shards}: {hi-lo:,} posts @ {rate:,.0f}/s  "
              f"[{done:,}/{n:,}]  ETA {eta/60:.1f} min", flush=True)

    emb = np.concatenate([np.load(shard_dir / f"{s:04d}.npy") for s in range(n_shards)]) \
        if n_shards else np.zeros((0, 384), np.float32)
    print(f"    [{coin}] embedded {emb.shape[0]:,} in {(time.time()-t_all)/60:.1f} min")

    idx = posts[["id", "created_utc"]].copy()
    idx["has_text"] = has_text
    idx["emb_row"] = -1
    idx.loc[has_text, "emb_row"] = np.arange(emb.shape[0])
    np.save(emb_path, emb)
    idx.to_parquet(idx_path, index=False)
    # clean shards once the consolidated cache is written
    for s in range(n_shards):
        (shard_dir / f"{s:04d}.npy").unlink(missing_ok=True)
    shard_dir.rmdir()
    return emb, idx


# ----------------------------------------------------------------------------- label transfer
def load_phase_c() -> pd.DataFrame:
    pc = pd.read_parquet(PHASE_C)
    pc = pc[pc["primary_label"].isin(CB_LABELS)].copy()
    pc["cluster_id"] = pc["cluster_id"].astype(int)
    pc["month"] = pc["month"].astype(str)
    return pc[["coin", "month", "cluster_id", "primary_label"]]


def pseudolabel_coin(coin: str, emb: np.ndarray, idx: pd.DataFrame,
                     pc: pd.DataFrame) -> pd.DataFrame:
    """Assign each text-bearing post the primary_label of its nearest labelled centroid
    within its own coin-month. Returns df: emb_row, label, month (only assigned posts)."""
    txt = idx[idx["has_text"]].copy()
    txt["month"] = pd.to_datetime(txt["created_utc"], unit="s", utc=True).dt.strftime("%Y-%m")
    pcc = pc[pc["coin"] == coin]
    label_by_month = {m: g.set_index("cluster_id")["primary_label"].to_dict()
                      for m, g in pcc.groupby("month")}

    rows_emb, rows_lab = [], []
    n_skip_month = 0
    for month, grp in txt.groupby("month"):
        cfile = NARR / coin / "cluster_samples" / f"cluster_centroids_{coin}_{month.replace('-', '_')}.npz"
        lbls = label_by_month.get(month)
        if not cfile.exists() or not lbls:
            n_skip_month += len(grp)
            continue
        z = np.load(cfile)
        cids = [c for c in z.files if int(c) in lbls]   # only labelled clusters
        if not cids:
            n_skip_month += len(grp)
            continue
        C = np.stack([z[c] for c in cids]).astype(np.float32)        # (k,384) unit
        clab = np.array([lbls[int(c)] for c in cids])
        rows = grp["emb_row"].to_numpy()
        sims = emb[rows] @ C.T                                       # cosine
        rows_emb.append(rows)
        rows_lab.append(clab[sims.argmax(1)])
    if not rows_emb:
        return pd.DataFrame(columns=["emb_row", "label", "month"])
    out = pd.DataFrame({"emb_row": np.concatenate(rows_emb),
                        "label": np.concatenate(rows_lab)})
    out = out.merge(txt[["emb_row", "month"]], on="emb_row", how="left")
    print(f"  [{coin}] pseudo-labelled {len(out):,} posts; "
          f"{n_skip_month:,} skipped (no labelled centroids that month)")
    return out


# ----------------------------------------------------------------------------- classifier
def build_head(head: str):
    if head == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                                  multi_class="multinomial", n_jobs=-1)
    if head == "mlp":
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(256,), max_iter=40, early_stopping=False,
                             random_state=0)  # early_stopping=False: sklearn bug w/ string labels
    if head == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63,
                              subsample=0.8, colsample_bytree=0.8, n_jobs=-1)
    raise ValueError(f"unknown head: {head}")


def _stratified_subsample(y: np.ndarray, max_n: int, seed: int = 0) -> np.ndarray:
    """Indices of a label-stratified subsample of size ~max_n (memory cap for lbfgs,
    which copies X to float64). A linear head needs nowhere near the full 1.5M rows."""
    n = len(y)
    if n <= max_n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    keep = []
    for lab in np.unique(y):
        li = np.where(y == lab)[0]
        take = min(len(li), max(1, int(round(len(li) * max_n / n))))
        keep.append(rng.choice(li, take, replace=False))
    return np.concatenate(keep)


def evaluate(head: str, X: np.ndarray, y: np.ndarray, months: np.ndarray,
             max_train: int) -> dict:
    """Temporal hold-out validity: train months < cutoff, test months >= cutoff."""
    from sklearn.metrics import f1_score, accuracy_score, classification_report
    tr = months < TEMPORAL_SPLIT
    te = ~tr
    if tr.sum() == 0 or te.sum() == 0:
        raise RuntimeError(f"temporal split {TEMPORAL_SPLIT} left an empty side "
                           f"(train={tr.sum()}, test={te.sum()})")
    tr_idx = np.where(tr)[0]
    sub = tr_idx[_stratified_subsample(y[tr_idx], max_train)]
    clf = build_head(head)
    print(f"  fit {head} on {len(sub):,} (capped from {tr.sum():,}, train < {TEMPORAL_SPLIT}); "
          f"test {te.sum():,} (>= {TEMPORAL_SPLIT}) …")
    clf.fit(X[sub], y[sub])
    pred = clf.predict(X[te])
    macro_f1 = f1_score(y[te], pred, average="macro")
    weighted_f1 = f1_score(y[te], pred, average="weighted")
    acc = accuracy_score(y[te], pred)
    rep = classification_report(y[te], pred, zero_division=0, output_dict=True)
    print(f"  TEMPORAL hold-out: macro-F1={macro_f1:.3f}  weighted-F1={weighted_f1:.3f}  acc={acc:.3f}")
    return {"macro_f1": macro_f1, "weighted_f1": weighted_f1, "acc": acc,
            "n_train": int(tr.sum()), "n_test": int(te.sum()), "report": rep}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coins", nargs="+", default=PANEL_COINS)
    ap.add_argument("--head", default="logreg", choices=["logreg", "mlp", "lgbm"])
    ap.add_argument("--no-refit", action="store_true",
                    help="skip refit-on-all + per-post topic table (validity report only)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"],
                    help="embedding device; default cpu (avoids MPS unified-memory thrash "
                         "on a memory-pressured machine)")
    ap.add_argument("--max-train", type=int, default=500_000,
                    help="cap on rows fed to the (linear) head; lbfgs copies X to float64 "
                         "so the full 1.5M rows blow up RAM. Stratified subsample.")
    ap.add_argument("--predict-chunk", type=int, default=100_000,
                    help="predict_proba batch size (DOGE ~690k posts -> float64 blowup)")
    ap.add_argument("--drop-short", action="store_true",
                    help="exclude is_short posts before train/eval (classifier bake-off)")
    ap.add_argument("--min-len", type=int, default=0,
                    help="also drop posts whose text length <= this (0 = off)")
    ap.add_argument("--out-tag", default="",
                    help="suffix for the validity-report filename so bake-off variants don't clobber")
    args = ap.parse_args()

    for d in (EMB_DIR, TOPIC_DIR, MODEL_DIR, OUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print("Loading Phase-C labels …")
    pc = load_phase_c()
    print(f"  {len(pc):,} labelled clusters across {pc['coin'].nunique()} coins")

    # lazy embedding model: only constructed on a cache miss (fully-cached run skips torch)
    _model_box = {}
    def get_model():
        if "m" not in _model_box:
            _model_box["m"] = _get_model(args.device)
        return _model_box["m"]

    # 1) embed + 2) pseudo-label, per coin. Build the training set incrementally;
    # do NOT hold every coin's full embedding matrix (that + X + lbfgs float64 copy is
    # what thrashed swap). Embeddings are reloaded from cache per coin in the predict loop.
    X_parts, y_parts, m_parts = [], [], []
    for coin in args.coins:
        print(f"\n=== {coin} ===")
        emb, idx = embed_coin(coin, get_model)
        pl = pseudolabel_coin(coin, emb, idx, pc)
        if len(pl) and (args.drop_short or args.min_len > 0):     # bake-off filter
            posts = pd.read_parquet(REDDIT / coin / "posts.parquet")
            mask = pd.Series(False, index=posts.index)
            if args.drop_short and "is_short" in posts.columns:
                mask |= posts["is_short"].astype(bool)
            if args.min_len > 0 and TEXT_COL in posts.columns:
                mask |= posts[TEXT_COL].astype(str).str.len() <= args.min_len
            drop_ids = set(posts.loc[mask, "id"])
            txt = idx[idx["has_text"]]
            row_id = dict(zip(txt["emb_row"].to_numpy(), txt["id"].to_numpy()))
            keep = np.fromiter((row_id.get(r) not in drop_ids
                                for r in pl["emb_row"].to_numpy()), dtype=bool, count=len(pl))
            n0 = len(pl); pl = pl[keep]
            print(f"  [{coin}] drop-short filter: kept {len(pl):,}/{n0:,}")
        if len(pl):
            X_parts.append(emb[pl["emb_row"].to_numpy()].copy())   # only the labelled rows
            y_parts.append(pl["label"].to_numpy())
            m_parts.append(pl["month"].to_numpy())
        del emb, idx, pl

    X = np.concatenate(X_parts); y = np.concatenate(y_parts); months = np.concatenate(m_parts)
    del X_parts, y_parts, m_parts
    print(f"\nTraining set: {len(y):,} pseudo-labelled posts, {len(np.unique(y))} classes present")

    # 3) validity (temporal hold-out)
    metrics = evaluate(args.head, X, y, months, args.max_train)

    # write validity report
    tag = f"_{args.out_tag}" if args.out_tag else ""
    rep = pd.DataFrame(metrics["report"]).T
    rep.to_csv(OUT_DIR / f"topic_classifier_report_{args.head}{tag}.csv")
    summ = [
        f"# Per-post topic classifier — validity ({args.head})", "",
        f"Label transfer: nearest per-month centroid -> Phase-C primary_label (34 classes).",
        f"Temporal hold-out split at **{TEMPORAL_SPLIT}** (train earlier, test later).", "",
        f"- training posts (all): **{len(y):,}**",
        f"- temporal train / test: {metrics['n_train']:,} / {metrics['n_test']:,}",
        f"- **macro-F1 = {metrics['macro_f1']:.3f}**, weighted-F1 = {metrics['weighted_f1']:.3f}, "
        f"accuracy = {metrics['acc']:.3f}", "",
        "Per-class precision/recall/F1/support in "
        f"`topic_classifier_report_{args.head}.csv`.", "",
        "If macro-F1 is weak (esp. on the substantive 5.x narrative classes), re-run with "
        "`--head mlp` or `--head lgbm` before building the intraday panel (script 46).",
    ]
    (OUT_DIR / f"topic_classifier_report_{args.head}{tag}.md").write_text("\n".join(summ))
    print(f"  ✓ validity report -> {OUT_DIR / f'topic_classifier_report_{args.head}{tag}.md'}")

    if args.no_refit:
        print("\n--no-refit set: stopping after validity report.")
        return

    # 4) refit for production labelling. Subsample (stratified) to the same memory cap —
    # a linear head saturates well below 1.5M rows, and we never need the full float64 copy.
    import joblib
    sub = _stratified_subsample(y, args.max_train, seed=1)
    print(f"\nRefit {args.head} on {len(sub):,} posts (capped from {len(y):,}; "
          f"returns never seen) for production labelling …")
    clf = build_head(args.head)
    clf.fit(X[sub], y[sub])
    classes = list(clf.classes_)
    joblib.dump({"clf": clf, "classes": classes, "embed_model": EMBED_MODEL_NAME,
                 "text_col": TEXT_COL, "metrics": {k: metrics[k] for k in
                 ("macro_f1", "weighted_f1", "acc", "n_train", "n_test")}},
                MODEL_DIR / f"topic_classifier_{args.head}.joblib")
    del X, y, months   # free the training matrix before the per-coin predict pass

    classes_arr = np.array(classes)
    for coin in args.coins:
        emb = np.load(EMB_DIR / f"{coin}.npy")                      # reload from cache
        idx = pd.read_parquet(EMB_DIR / f"{coin}_index.parquet")
        txt = idx[idx["has_text"]]
        rows = txt["emb_row"].to_numpy()
        pred_sub = np.empty(len(rows), dtype=object)
        pred_pr = np.empty(len(rows), dtype=np.float32)
        for lo in range(0, len(rows), args.predict_chunk):         # chunk: avoid float64 blowup
            hi = min(lo + args.predict_chunk, len(rows))
            proba = clf.predict_proba(emb[rows[lo:hi]])
            ci = proba.argmax(1)
            pred_sub[lo:hi] = classes_arr[ci]
            pred_pr[lo:hi] = proba[np.arange(len(ci)), ci].astype(np.float32)
        topic = pd.DataFrame({
            "id": txt["id"].to_numpy(),
            "created_utc": txt["created_utc"].to_numpy(),
            "pred_subtype": pred_sub,
            "pred_proba": pred_pr,
        })
        topic.to_parquet(TOPIC_DIR / f"{coin}.parquet", index=False)
        print(f"  [{coin}] topic table -> {len(topic):,} posts "
              f"(mean max-proba {topic['pred_proba'].mean():.2f})")
        del emb, idx, txt

    print(f"\n✓ model -> {MODEL_DIR / f'topic_classifier_{args.head}.joblib'}")
    print(f"✓ topic tables -> {TOPIC_DIR}")


if __name__ == "__main__":
    main()
