# Attention, Sentiment, or Narrative? — analysis code

Analysis code for the paper *"Attention, sentiment, or narrative? Decomposing the social-media
signal in cryptocurrency markets"* (submitted for review).

This repository contains the **code only**, for browsing. The full replication package — these
scripts *plus* all input data (per-post narrative labels and identifiers, the frozen classifier,
stance/intent labels, feature panel, hourly prices) — is permanently archived at Zenodo:

**DOI: [10.5281/zenodo.21139695](https://doi.org/10.5281/zenodo.21139695)**

To actually run anything, download the Zenodo package and run the scripts from its
`paper_narrative_v2/scripts/` directory — paths resolve relative to the package layout.
The raw Reddit text is not distributable under Reddit's terms of service; the deposited post
identifiers allow rehydration directly from Reddit.

## Map from script to exhibit

| Script | Produces |
|---|---|
| `45_per_post_topic_classifier.py` | Frozen per-post narrative classifier + temporal-holdout validity |
| `46_intraday_decomposition.py` | Intraday bars and window-level features |
| `50_horizon_scan_decomposition.py` | Horizon scan (Table 2); `--substantive-only` for the 30-subtype version |
| `53_bayes_moderated_mediation.py`, `54_ml_mediation_triangulation.py`, `55_multivariate_apath.py` | Mediation estimands (Table 4) |
| `58_conditional_directional.py`, `67_directional_robustness.py` | Directional blip + robustness battery |
| `61_volforecast_economic.py`, `62_volecon_finish.py` | Volatility horse race (Table 6), VaR economic value |
| `63_regime_rolling.py`, `68_regime_conditional.py` | Regime and rolling robustness |
| `64_replication_ladder.py` | Rigour ladder (Table 8) |
| `65_figures.py` | Figure 1 (time-series intuition) |
| `70_fig_vol_unified.py` | Figure 2 (2×2 volatility-mechanism exhibit) |

Remaining scripts are supporting stages; each file's docstring states its role.

## Requirements

Python ≥ 3.9: `numpy pandas scikit-learn scipy statsmodels matplotlib pyarrow`
(plus `sentence-transformers` only if re-embedding rehydrated text from scratch).

## License

MIT (see `LICENSE`). The data files in the Zenodo deposit are under CC-BY 4.0.
