#!/usr/bin/env python3
"""
56_classify_general_stream.py
=============================
Topic-classify the GENERAL-MARKET Reddit stream (r/CryptoCurrency + r/CryptoMarkets) against the
frozen 34-label codebook, to enable the market-augmented mediation (script 57). Unlike script 45
this does NOT train anything — it reuses the FROZEN classifier `models/topic_classifier_logreg.joblib`
fit on the 6 coin streams (time-separated, returns never seen) and merely applies it to general posts'
own embeddings. So it inherits 45's leak-free property: per-post topic from its own text, no clustering,
no future info, no returns.

The general stream is the market-wide narrative signal (common to all coins), to be added in script 57
as SHARED, non-coin-specific content + attention/volume inputs.

Reuses script 45's embedding machinery (same MiniLM, batch/seq caps, disk-backed shards so a
wall-clock-cap kill RESUMES rather than restarts).

Reads (read-only, main repo):
  data/processed/reddit_clean/general/{CryptoCurrency,CryptoMarkets}/posts.parquet  (id, created_utc, text_for_labeling)
  paper_narrative/models/topic_classifier_logreg.joblib                              (frozen clf + classes)

Writes (under paper_narrative/):
  data/embeddings/general_{sub}.npy + _index.parquet     cached unit embeddings (resumable)
  data/post_topics/general_{sub}.parquet                 id, created_utc, pred_subtype, pred_proba
  data/post_topics/general.parquet                       combined (+ subreddit col) for script 57

Usage:
  python 56_classify_general_stream.py                       # both subs, cpu
  python 56_classify_general_stream.py --subs CryptoMarkets  # one sub (validate fast)
"""
from __future__ import annotations

import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np
import pandas as pd

PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def _load(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m45 = _load("clf45", "45_per_post_topic_classifier.py")
REDDIT = PROJECT_ROOT / "data" / "processed" / "reddit_clean"
EMB_DIR = m45.EMB_DIR
TOPIC_DIR = m45.TOPIC_DIR
MODEL_DIR = m45.MODEL_DIR
TEXT_COL = m45.TEXT_COL
SHARD_POSTS = m45.SHARD_POSTS
EMBED_BATCH = m45.EMBED_BATCH

GEN_SUBS = ["CryptoCurrency", "CryptoMarkets"]


def embed_general(sub: str, get_model) -> tuple[np.ndarray, pd.DataFrame]:
    """Embed text-bearing general-sub posts (unit vectors), disk-backed shards (resumable).
    Mirrors m45.embed_coin but for the general/{sub}/posts.parquet layout."""
    key = f"general_{sub}"
    emb_path = EMB_DIR / f"{key}.npy"
    idx_path = EMB_DIR / f"{key}_index.parquet"
    shard_dir = EMB_DIR / f"{key}_shards"
    if emb_path.exists() and idx_path.exists():
        emb = np.load(emb_path); idx = pd.read_parquet(idx_path)
        print(f"  [{key}] cached embeddings: {emb.shape[0]:,} text posts / {len(idx):,} total")
        return emb, idx

    posts = pd.read_parquet(REDDIT / "general" / sub / "posts.parquet",
                            columns=["id", "created_utc", TEXT_COL])
    posts = posts.sort_values("created_utc").reset_index(drop=True)
    has_text = (posts[TEXT_COL].notna()
                & (posts[TEXT_COL].astype(str).str.strip().str.len() > 0)).to_numpy()
    txt = posts.loc[has_text, TEXT_COL].astype(str).tolist()
    n = len(txt)
    n_shards = (n + SHARD_POSTS - 1) // SHARD_POSTS
    print(f"  [{key}] embedding {n:,} text posts of {len(posts):,} in {n_shards} shard(s) …")
    shard_dir.mkdir(parents=True, exist_ok=True)
    model = get_model()

    t_all = time.time()
    for s in range(n_shards):
        sp = shard_dir / f"{s:04d}.npy"
        if sp.exists():
            print(f"    shard {s+1}/{n_shards}: cached, skip", flush=True)
            continue
        lo, hi = s * SHARD_POSTS, min((s + 1) * SHARD_POSTS, n)
        t0 = time.time()
        e = model.encode(txt[lo:hi], batch_size=EMBED_BATCH, normalize_embeddings=True,
                         show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        np.save(sp, e)
        rate = (hi - lo) / max(time.time() - t0, 1e-6)
        eta = (n - hi) / max(rate, 1e-6)
        print(f"    shard {s+1}/{n_shards}: {hi-lo:,} @ {rate:,.0f}/s  [{hi:,}/{n:,}]  "
              f"ETA {eta/60:.1f} min", flush=True)

    emb = np.concatenate([np.load(shard_dir / f"{s:04d}.npy") for s in range(n_shards)]) \
        if n_shards else np.zeros((0, 384), np.float32)
    print(f"    [{key}] embedded {emb.shape[0]:,} in {(time.time()-t_all)/60:.1f} min")
    idx = posts[["id", "created_utc"]].copy()
    idx["has_text"] = has_text
    idx["emb_row"] = -1
    idx.loc[has_text, "emb_row"] = np.arange(emb.shape[0])
    np.save(emb_path, emb)
    idx.to_parquet(idx_path, index=False)
    for s in range(n_shards):
        (shard_dir / f"{s:04d}.npy").unlink(missing_ok=True)
    shard_dir.rmdir()
    return emb, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", nargs="+", default=GEN_SUBS, choices=GEN_SUBS)
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda", "auto"])
    ap.add_argument("--predict-chunk", type=int, default=100_000)
    args = ap.parse_args()
    for d in (EMB_DIR, TOPIC_DIR, MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    import joblib
    bundle = joblib.load(MODEL_DIR / "topic_classifier_mlp.joblib")  # v2: MLP (matches coin classifier)
    clf = bundle["clf"]; classes_arr = np.array(bundle["classes"])
    print(f"frozen classifier: {len(classes_arr)} classes, "
          f"macro-F1={bundle['metrics']['macro_f1']:.3f} (from script 45)")

    _model_box = {}
    def get_model():
        if "m" not in _model_box:
            _model_box["m"] = m45._get_model(args.device)
        return _model_box["m"]

    combined = []
    for sub in args.subs:
        print(f"\n=== general/{sub} ===")
        emb, idx = embed_general(sub, get_model)
        txt = idx[idx["has_text"]]
        rows = txt["emb_row"].to_numpy()
        pred_sub = np.empty(len(rows), dtype=object)
        pred_pr = np.empty(len(rows), dtype=np.float32)
        for lo in range(0, len(rows), args.predict_chunk):
            hi = min(lo + args.predict_chunk, len(rows))
            proba = clf.predict_proba(emb[rows[lo:hi]])
            ci = proba.argmax(1)
            pred_sub[lo:hi] = classes_arr[ci]
            pred_pr[lo:hi] = proba[np.arange(len(ci)), ci].astype(np.float32)
        topic = pd.DataFrame({"id": txt["id"].to_numpy(),
                              "created_utc": txt["created_utc"].to_numpy(),
                              "pred_subtype": pred_sub, "pred_proba": pred_pr})
        topic.to_parquet(TOPIC_DIR / f"general_{sub}.parquet", index=False)
        t = topic.copy(); t["subreddit"] = sub
        combined.append(t)
        print(f"  [general/{sub}] topic table -> {len(topic):,} posts "
              f"(mean max-proba {topic['pred_proba'].mean():.2f})")
        del emb, idx, txt

    allc = pd.concat(combined, ignore_index=True)
    allc.to_parquet(TOPIC_DIR / "general.parquet", index=False)
    print(f"\n✓ combined general topic table -> {TOPIC_DIR/'general.parquet'}  ({len(allc):,} posts)")
    print("  subtype distribution (top 10):")
    print(allc["pred_subtype"].value_counts(normalize=True).head(10).to_string())


if __name__ == "__main__":
    main()
