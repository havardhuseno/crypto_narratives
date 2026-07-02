#!/usr/bin/env python3
"""
56b_market_aggregate.py  (v2 only)
==================================
Rebuild the 4h market-sentiment aggregate FRESH from the updated general-stream labels, so the
market-augmented analyses (scripts 57/62) are consistent with the post-fix coin data. We do NOT use
the parent's prebuilt market_4h_sentiment.parquet (it is stale: 06-04 02:48, predating the
general posts/comments fix at 07:52-10:34).

Mirrors parent scripts/08b_aggregate_reddit_4h.py for the market panel:
  bar_start_utc = 4h floor (UTC) of created_utc; window [B, B+4h).
  market_n_total      = posts+comments across r/CryptoCurrency + r/CryptoMarkets in the bar
  market_net_sentiment = bullish_share - bearish_share over labelled rows
(scripts 57/62 consume only these two columns.)

Reads (parent, read-only): data/processed/labeled_full/general/stance/{CryptoCurrency,CryptoMarkets}_{posts,comments}_b*.parquet
Writes (v2): paper_narrative_v2/data/market_4h_sentiment.parquet
"""
from __future__ import annotations
import glob
from pathlib import Path
import numpy as np
import pandas as pd

PAPER_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEN = PROJECT_ROOT / "data" / "processed" / "labeled_full" / "general" / "stance"
OUT = PAPER_ROOT / "data" / "market_4h_sentiment.parquet"
HOUR4 = 4 * 3600


def main():
    files = []
    for sub in ["CryptoCurrency", "CryptoMarkets"]:
        for ctype in ["posts", "comments"]:
            files += sorted(glob.glob(str(GEN / f"{sub}_{ctype}_b*.parquet")))
    print(f"reading {len(files)} general stance batches …")
    parts = []
    for f in files:
        try:
            parts.append(pd.read_parquet(f, columns=["created_utc", "stance"]))
        except Exception:
            parts.append(pd.read_parquet(f, columns=["created_utc"]).assign(stance=np.nan))
    df = pd.concat(parts, ignore_index=True)
    df = df.dropna(subset=["created_utc"])
    df["bar"] = (df["created_utc"].astype("int64") // HOUR4) * HOUR4
    g = df.groupby("bar")
    out = pd.DataFrame({
        "market_n_total": g.size(),
        "bull": g["stance"].apply(lambda s: (s == "bullish").sum()),
        "bear": g["stance"].apply(lambda s: (s == "bearish").sum()),
        "n_lab": g["stance"].apply(lambda s: s.notna().sum()),
    })
    den = out["n_lab"].replace(0, np.nan)
    out["market_net_sentiment"] = (out["bull"] - out["bear"]) / den
    out["market_net_sentiment"] = out["market_net_sentiment"].fillna(0.0)
    out = out.reset_index()
    out["bar_start_utc"] = pd.to_datetime(out["bar"], unit="s", utc=True)
    out = out[["bar_start_utc", "market_n_total", "market_net_sentiment"]].sort_values("bar_start_utc")
    out.to_parquet(OUT, index=False)
    print(f"✓ {len(out):,} 4h bars -> {OUT}  "
          f"(span {out.bar_start_utc.min().date()}..{out.bar_start_utc.max().date()})")


if __name__ == "__main__":
    main()
