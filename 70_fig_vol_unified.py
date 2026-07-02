#!/usr/bin/env python3
"""
70_fig_vol_unified.py — unify the two volatility-mechanism figures (old figB + figC) into a
single 2x2 exhibit, so the paper makes the "gain lives in turbulent/surprise states" point once.

  (a) QLIKE reduction by realized-volatility tercile          [was figB; reads volecon_conditional.csv]
  (b) QLIKE reduction by predicted x realized cell (heatmap)  [was figC-left; surprise = pred-LO n real-HI]
  (c) narrative improvement vs HAR forecast miss (continuous) [was figC-right]
  (d) surprise-cell QLIKE reduction by year                   [new: the effect replicates every year]

Frozen (pre-2023) M0 (HAR+vol) vs M2 (+attention/sentiment+narrative), same pipeline as script 69.
Outputs: drafts/figures/figD_vol_mechanism.{png,pdf}
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PAPER = Path(__file__).resolve().parents[1]; HERE = Path(__file__).resolve().parent
OUT = PAPER / "outputs"; FIGS = PAPER / "drafts" / "figures"; FIGS.mkdir(parents=True, exist_ok=True)
def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m61 = _load("e61", "61_volforecast_economic.py")
m57 = _load("m57", "57_market_augmented.py")
qlike, SUB30, REDDIT_VOL, POLARITY = m61.qlike, m61.SUB30, m61.REDDIT_VOL, m61.POLARITY

from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
alphas = np.logspace(-3, 4, 30)   # match the tables' Ridge grid (scripts 61/vol_surprise_by_year)
df = m61.assemble(m61.COINS, 24, 24, [24, 72, 168], m61.m46.MIN_ITEMS, 3)
harrv, harvol = df.attrs["harrv"], df.attrs["harvol"]   # capture before merge (drops attrs)
df, _ = m57.merge_market(df, m57.build_market_features(24))   # canonical test sample (n=7480), matches tables
df = df.reset_index(drop=True); df["ts"] = pd.to_datetime(df["ts"], utc=True)
tr = df["is_train"].values; te = ~tr; y = df["y_logrv"].values
coind = pd.get_dummies(df["coin"], prefix="coin", drop_first=True).astype(float).values
def sc(c): raw = df[c].values; return StandardScaler().fit(raw[tr]).transform(raw)
PRICE = sc(harrv + harvol); SOC = sc(REDDIT_VOL + [c for c in POLARITY if c in df.columns]); CON = sc([f"share_{s}" for s in SUB30])
M0 = np.column_stack([coind, PRICE]); M2 = np.column_stack([coind, PRICE, SOC, CON])
p0 = RidgeCV(alphas=alphas).fit(M0[tr], y[tr]).predict(M0[te])
p2 = RidgeCV(alphas=alphas).fit(M2[tr], y[tr]).predict(M2[te])
yt = y[te]; qb = qlike(yt, p0); qn = qlike(yt, p2); improve = qb - qn; miss = yt - p0
yr = df["ts"].dt.year.values[te]

plt.rcParams.update({"font.family": "Arial", "font.size": 10.5, "axes.titlesize": 11,
                     "axes.titleweight": "bold", "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
                     "savefig.bbox": "tight", "figure.dpi": 120,
                     "ps.fonttype": 42, "pdf.fonttype": 42})  # embed TrueType (T&F artwork guide)
fig, ax = plt.subplots(2, 2, figsize=(11.5, 9.2))
for a, L in zip(ax.ravel(), "abcd"):
    a.text(-0.10, 1.06, f"({L})", transform=a.transAxes, fontsize=13, fontweight="bold", va="bottom")

# (a) tercile bars ----------------------------------------------------------------------
t = pd.read_csv(OUT / "volecon_conditional.csv").set_index("vol_tercile").reindex(["low", "mid", "high"])
red = t["QLIKE_reduction_%"].values
colors = ["#b2182b" if v < 0 else "#1b7837" for v in red]
b = ax[0, 0].bar(["Low", "Mid", "High"], red, color=colors, width=0.62, edgecolor="white")
ax[0, 0].axhline(0, color="#444", lw=0.9)
for bb, v in zip(b, red):
    ax[0, 0].annotate(f"{v:+.1f}%", (bb.get_x()+bb.get_width()/2, v), ha="center",
                      va="bottom" if v >= 0 else "top", xytext=(0, 4 if v >= 0 else -4),
                      textcoords="offset points", fontsize=10.5, fontweight="bold")
ax[0, 0].set_ylabel("QLIKE reduction vs.\nprice baseline (%)"); ax[0, 0].set_xlabel("Realized-volatility tercile")
ax[0, 0].set_title("Gain concentrates in turbulent regimes", loc="left")
ax[0, 0].spines[["top", "right"]].set_visible(False); ax[0, 0].margins(y=0.18)

# (b) predicted x realized heatmap ------------------------------------------------------
pmed, rmed = np.median(p0), np.median(yt); plo = p0 <= pmed; rlo = yt <= rmed
G = np.full((2, 2), np.nan)
for i, pm in enumerate([plo, ~plo]):
    for j, rm in enumerate([rlo, ~rlo]):
        m = pm & rm
        if m.sum(): G[i, j] = 100*(qb[m].mean()-qn[m].mean())/qb[m].mean()
im = ax[0, 1].imshow(G, cmap="RdYlGn", vmin=-13, vmax=13, aspect="auto")
ax[0, 1].set_xticks([0, 1], ["realized\nLOW", "realized\nHIGH"]); ax[0, 1].set_yticks([0, 1], ["predicted\nLOW", "predicted\nHIGH"])
for i in range(2):
    for j in range(2):
        if not np.isnan(G[i, j]): ax[0, 1].text(j, i, f"{G[i,j]:+.1f}%", ha="center", va="center", fontsize=12, fontweight="bold")
ax[0, 1].add_patch(plt.Rectangle((0.5, -0.5), 1, 1, fill=False, edgecolor="black", lw=2.5))
ax[0, 1].set_title("Largest in the surprise cell (calm→spike)", loc="left")
fig.colorbar(im, ax=ax[0, 1], fraction=0.046, label="QLIKE reduction (%)")

# (c) continuous view -------------------------------------------------------------------
bins = np.unique(np.quantile(miss, np.linspace(0, 1, 13))); idx = np.digitize(miss, bins[1:-1])
bx = [miss[idx == k].mean() for k in range(len(bins)-1) if (idx == k).any()]
by = [improve[idx == k].mean() for k in range(len(bins)-1) if (idx == k).any()]
ax[1, 0].axhline(0, color="#888", lw=0.8); ax[1, 0].axvline(0, color="#888", lw=0.8, ls=":")
ax[1, 0].plot(bx, by, "o-", color="#1b7837", lw=1.6)
ax[1, 0].set_xlabel("HAR forecast miss (realized − predicted log-RV)\n← over-predicts   |   under-predicts (surprise) →")
ax[1, 0].set_ylabel("Narrative QLIKE improvement\n(baseline − narrative)")
ax[1, 0].set_title("Helps exactly where HAR under-predicts", loc="left")
ax[1, 0].spines[["top", "right"]].set_visible(False)

# (d) surprise-cell reduction by year ---------------------------------------------------
# NB: sharper TERCILE surprise cell (bottom-tercile predicted n top-tercile realized), matching
# Table~tab:volsurpriseyr and its stable ~2.6% surprise frequency — not the median 2x2 of panel (b).
pcut = np.quantile(p0, [1/3, 2/3]); ycut = np.quantile(yt, [1/3, 2/3])
surprise = (p0 <= pcut[0]) & (yt >= ycut[1])
periods = [("2023", yr == 2023), ("2024", yr == 2024), ("2025–26", yr >= 2025)]
sred, ored = [], []
for _, mk in periods:
    s = mk & surprise; o = mk & ~surprise
    sred.append(100*(qb[s].mean()-qn[s].mean())/qb[s].mean() if s.sum() else np.nan)
    ored.append(100*(qb[o].mean()-qn[o].mean())/qb[o].mean() if o.sum() else np.nan)
x = np.arange(len(periods)); w = 0.38
ax[1, 1].bar(x-w/2, sred, w, color="#1b7837", label="Surprise cell (calm→spike)", edgecolor="white")
ax[1, 1].bar(x+w/2, ored, w, color="#bdbdbd", label="All other states", edgecolor="white")
ax[1, 1].axhline(0, color="#444", lw=0.9)
for xi, v in zip(x-w/2, sred): ax[1, 1].annotate(f"{v:+.0f}%", (xi, v), ha="center", va="bottom", xytext=(0, 3), textcoords="offset points", fontsize=9.5, fontweight="bold")
ax[1, 1].set_xticks(x, [p for p, _ in periods]); ax[1, 1].set_ylabel("QLIKE reduction (%)")
ax[1, 1].set_title("Surprise gain recurs every year", loc="left")
ax[1, 1].legend(fontsize=8.5, frameon=False, loc="upper left"); ax[1, 1].spines[["top", "right"]].set_visible(False); ax[1, 1].margins(y=0.20)

fig.tight_layout(w_pad=2.5, h_pad=3.0)
fig.savefig(FIGS / "figD_vol_mechanism.png", dpi=300); fig.savefig(FIGS / "figD_vol_mechanism.pdf")
fig.savefig(FIGS / "figD_vol_mechanism.eps", dpi=600)   # panel (b) imshow rasterizes at 600dpi in EPS
plt.close(fig)
print("terciles:", dict(zip(["low", "mid", "high"], np.round(red, 1))))
print("heatmap G:\n", np.round(G, 1))
print(f"n_test={te.sum()}  continuous-view peak improvement={max(by):+.3f}")
print("by-year surprise:", [f"{p}={s:+.1f}%" for (p, _), s in zip(periods, sred)])
print("by-year other:  ", [f"{p}={o:+.1f}%" for (p, _), o in zip(periods, ored)])
print("✓ figD ->", FIGS / "figD_vol_mechanism.pdf")
