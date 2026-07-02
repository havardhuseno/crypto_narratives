#!/usr/bin/env python3
"""
65_figures.py — publication figures for the QF paper.

Fig A  narrative_volatility_timeseries: for an illustrative coin, realized volatility, Reddit
       activity, and narrative composition over time co-move (the intuition).

The volatility-mechanism exhibit (old figB/figC) is now the unified 2x2 built by
70_fig_vol_unified.py (figD_vol_mechanism).

Outputs: drafts/figures/figA_narrative_volatility.{png,pdf}
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

PAPER = Path(__file__).resolve().parents[1]
FIGS = PAPER / "drafts" / "figures"; FIGS.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).resolve().parent

def _load(n, f):
    s = importlib.util.spec_from_file_location(n, HERE / f); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
m46 = _load("m46", "46_intraday_decomposition.py")
m50 = _load("m50", "50_horizon_scan_decomposition.py")
HOUR = m46.HOUR; CB = m46.CB_LABELS

plt.rcParams.update({
    "font.family": "Arial", "font.size": 11, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.labelsize": 11, "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8, "figure.dpi": 120, "savefig.bbox": "tight",
    "ps.fonttype": 42, "pdf.fonttype": 42,   # embed TrueType (T&F artwork guide)
})
C_VOL = "#b2182b"; C_ACT = "#2166ac"; C_F5 = "#762a83"; C_F4 = "#1b7837"; C_F3 = "#bdbdbd"


def tint(c, a):
    """Solid color equal to `c` at alpha `a` over white — EPS has no transparency."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(c)
    return (1 - a * (1 - r), 1 - a * (1 - g), 1 - a * (1 - b))


def fig_timeseries(coin="DOGE"):
    d = m46.build_bars(coin, 24).sort_values("ts").copy()
    c = m50.build_coin_caches(coin)
    # hourly realized variance -> daily RV (annualized %)
    r2 = np.diff(c["cum_r2"]); hours = np.arange(c["hmin"], c["hmin"] + len(r2))
    rv = pd.Series(r2, index=pd.to_datetime(hours * HOUR, unit="s", utc=True))
    rv_d = np.sqrt(rv.resample("1D").sum()) * np.sqrt(365) * 100.0
    # daily Reddit activity (24h-window post+comment counts, sampled end-of-day)
    act = d.set_index("ts")["n_items"].resample("1D").max()
    # daily narrative family composition (share of posts)
    for fam in ["3", "4", "5"]:
        d[f"fam{fam}"] = d[[f"share_{l}" for l in CB if l.startswith(fam + ".")]].sum(axis=1)
    fam = d.set_index("ts")[["fam3", "fam4", "fam5"]].resample("1D").mean().dropna()

    fig, ax = plt.subplots(3, 1, figsize=(10, 8.2), sharex=True,
                           gridspec_kw={"height_ratios": [1.1, 1, 1.1], "hspace": 0.12})
    ax[0].fill_between(rv_d.index, rv_d.values, color=tint(C_VOL, 0.20))
    ax[0].plot(rv_d.index, rv_d.values, color=C_VOL, lw=1.1)
    ax[0].set_ylabel("Realized vol.\n(annualized %)")
    ax[0].set_title(f"({coin}) Narrative, attention, and volatility co-move", loc="left")

    ax[1].fill_between(act.index, act.values, color=tint(C_ACT, 0.25))
    ax[1].plot(act.index, act.values, color=C_ACT, lw=0.9)
    ax[1].set_ylabel("Reddit activity\n(posts+comments / day)")
    ax[1].set_yscale("log")

    ax[2].stackplot(fam.index, fam["fam5"], fam["fam4"], fam["fam3"],
                    colors=[tint(C_F5, 0.9), tint(C_F4, 0.9), tint(C_F3, 0.9)],
                    labels=["Family 5 (speculation / price / market)",
                            "Family 4 (technology / events / adoption)",
                            "Family 3 (noise)"])
    ax[2].set_ylabel("Narrative\ncomposition (share)")
    ax[2].set_ylim(0, 1); ax[2].yaxis.set_major_formatter(PercentFormatter(1.0))
    ax[2].legend(loc="upper center", ncol=3, fontsize=8, frameon=False,
                 bbox_to_anchor=(0.5, -0.18))
    ax[2].set_xlabel("Date")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False); a.margins(x=0)
    stem = f"figA_narrative_volatility_{coin}"
    fig.savefig(FIGS / f"{stem}.png", dpi=300)
    fig.savefig(FIGS / f"{stem}.pdf")
    fig.savefig(FIGS / f"{stem}.eps")
    plt.close(fig)
    print("✓ figA ->", FIGS / f"{stem}.png")


if __name__ == "__main__":
    import sys
    coins = sys.argv[1:] if len(sys.argv) > 1 else ["DOGE", "BTC"]
    for c in coins:
        fig_timeseries(c)
