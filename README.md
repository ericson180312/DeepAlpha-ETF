# DeepAlpha-ETF: Does an Ensemble LSTM Beat Momentum at Monthly ETF Rotation?

![Python](https://img.shields.io/badge/python-3.13-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)

A research pipeline that asks one question: **can an ensemble LSTM trained on price and macro features rank a
13-ETF universe by next-month excess return over SPY better than a plain momentum rule?** The pipeline is built so
that the answer is decided by pre-registered criteria computed by code, not by reading an equity curve.

**Answer (walk-forward, 2022-02 to 2026-08, 55 months): no.** The LSTM has no detectable out-of-sample ranking
ability (monthly rank-IC −0.01, t = −0.16), is not better than a ridge regression on the same features, and its
net-of-cost Sharpe sits at the 10th percentile of random 5-of-13 selection. Simple 3/6/12-month momentum has a
rank-IC of +0.11 (t = 2.1) and a Sharpe at the 98th percentile of the same null.

Three pre-registered rounds have now been run, each committed and pushed before it was executed:
**round 1** rejected the ensemble LSTM, **round 2** confirmed the momentum baseline is a plateau rather
than a lucky parameter, and **round 3** rejected a pooled cross-sectional LightGBM that was built
specifically to fix the LSTM's structural flaw. Details below.

## What the pipeline does

| Step | Script | What it does |
| :--- | :--- | :--- |
| 1 | `_1_fetch_and_clean_data.py` | Downloads adjusted closes for a pinned window (`START_DATE`/`END_DATE` in `_config.py`); forward-fills only, errors on leading NaN |
| 2 | `_2_baseline_performance.py` | Momentum baseline (mean of 3/6/12-month returns, top-5, absolute-momentum filter to cash). Also holds the **shared backtest accounting** used by every strategy |
| 3 | `_3_ml_data_pipeline.py` | Features (SPY 200-day trend and 20-day vol, per-ETF daily return / vol / 20-60-120-day momentum), 20-day forward excess-return targets, and **walk-forward folds with purging** |
| 4 | `_4_lstm_model.py` | Trains one 20-seed LSTM ensemble per fold, early-stopped on that fold's validation tail |
| 4b | `_4b_ridge_baseline.py` | Ridge regression on the same features, targets and folds; alpha picked by validation MSE |
| 5 | `_5_strategy_backtest.py` | Stitches each fold's out-of-sample scores into one series and backtests LSTM / ridge / momentum / SPY under identical accounting; per-fold tables |
| 6 | `_6_diagnostics.py` | Rank-IC by strategy and fold, in-sample vs OOS IC, Sharpe SE and bootstrap CI, random-selection null, seed dispersion, cost sensitivity, and the **mechanical verdict** against `_config.PRE_REGISTRATION` |
| 7 | `_7_momentum_robustness.py` | Neighbourhood check on the momentum baseline — plateau or lone peak? — judged against `_config.PRE_REGISTRATION_MOMENTUM` |
| 4c | `_4c_pooled_gbdt.py` | Pooled (date, asset) representation with no asset identity; LightGBM and a pooled ridge on the same rows |
| 8 | `_8_pooled_diagnostics.py` | Verdict for the pooled round against `_config.PRE_REGISTRATION_POOLED`; the bar is plain momentum |

`main.py` runs them in order (about 6 minutes on a GPU). All parameters live in `_config.py`.
`pytest tests/` runs 37 tests covering the accounting and the walk-forward split; every expected value
in them is hand-computed rather than copied from the implementation.

### Validation design

* **Walk-forward, expanding window.** Test years 2022, 2023, 2024, 2025 and 2026-to-date; each fold trains only on
  data before its test start. Everything from 2022-02 on is out of sample.
* **Purging.** The target spans 20 trading days, so the last 20 samples before each train/val and val/test boundary
  are dropped. Early stopping never sees a return that overlaps the test period.
* **Same accounting for every line.** Equal weight per slot, cash months count as 0% (not deleted), incomplete
  final month excluded, 10 bps per unit traded, Sharpe on excess return over BIL with Lo (2002) standard error.
* **Consistent fallback.** The ML target is alpha *relative to SPY*, so a slot whose predicted alpha is ≤ 0 holds SPY.
  The momentum baseline uses absolute momentum, so its filtered slots hold cash.
* **Baseline-first.** The burden of proof is on the LSTM: it must beat a ridge on the same inputs and the momentum rule.
* **Pre-registration.** The pass/fail criteria are text in `_config.PRE_REGISTRATION`, printed verbatim at the top
  of the diagnostics report and evaluated by code at the bottom. They were committed and pushed before the ensembles
  were trained (commit `1a462a2`; research log in the author's `claude-notes` repo).

## Results

### Walk-forward out-of-sample, 2022-02 to 2026-08 (55 months, net of 10 bps, Sharpe ex-BIL)

Every line is net of 10 bps on traded value and carries its average exposure, so that a return
comparison is not simply a comparison of risk budgets.

| Strategy | Rank-IC (hit rate) | CAGR | Max DD | Sharpe ± SE | Exposure | Percentile vs random 5-of-13 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Ensemble LSTM | −0.01 (53%) | 8.7% | −23.4% | 0.37 ± 0.47 | 1.00 | 10% |
| Ridge (same features) | +0.09 (64%) | 9.3% | −18.9% | 0.41 ± 0.47 | 1.00 | 14% |
| Momentum (3/6/12m, top 5) | +0.11 (67%) | 17.7% | −8.9% | 1.06 ± 0.48 | 0.96 | 98% |
| SPY | — | 13.9% | −20.3% | 0.66 ± 0.47 | 1.00 | — |
| Equal-weight 13 ETFs | — | 12.6% | −14.2% | 0.83 ± 0.47 | 1.00 | 84% |

### Per fold: the LSTM memorises its training period and forgets it out of sample

| Fold (test year) | Train samples | In-sample rank-IC | OOS rank-IC | OOS Sharpe LSTM / Ridge / Momentum / SPY |
| :--- | :--- | :--- | :--- | :--- |
| 2022 | 166 | +0.90 | −0.07 | −0.18 / 0.08 / 0.02 / −0.32 |
| 2023 | 418 | +0.82 | −0.06 | −0.18 / −0.16 / 0.10 / 1.06 |
| 2024 | 669 | +0.86 | +0.18 | 1.40 / 1.22 / 1.78 / 1.78 |
| 2025 | 919 | +0.84 | −0.04 | 0.85 / 0.85 / 3.28 / 1.08 |
| 2026 (Jan–Aug) | 1169 | +0.77 | −0.09 | 0.69 / 0.71 / 0.47 / 0.94 |

### Pre-registered verdict (computed by `_6_diagnostics.py`)

| Criterion | Rule | Result |
| :--- | :--- | :--- |
| A — predictive power | OOS rank-IC mean > 0 and t > 2 | **Fails** (mean −0.009, t −0.16) |
| B — vs linear baseline | paired t of (LSTM − ridge) monthly IC > 2 | **Fails** (t −1.95; ridge is better) |
| C — strategy level | Sharpe(LSTM) − Sharpe(momentum) ≥ 1 SE and null percentile ≥ 95% | **Fails** (−0.69 vs SE 0.47; 10th percentile) |
| D — consistency | per-fold results reported alongside the aggregate | 1 of 5 folds has positive OOS IC (2024); reported above |

![diagnostics](diagnostics_report.png)

### Is the momentum baseline itself just a lucky parameter?

Once the LSTM is ruled out, momentum is the only signal holding up any conclusion here — and its
63/126/252 windows and top-5 were inherited, never checked. `_7_momentum_robustness.py` rescales the
window triple by 0.5×–2× and varies TOP_N over 3–7, giving a 30-cell grid evaluated on the same 55
months under the same accounting.

| | Top 3 | Top 4 | Top 5 | Top 6 | Top 7 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ×0.50 | 0.58 | 0.78 | 1.05 | 1.11 | 1.07 |
| ×0.75 | 0.89 | 0.95 | 0.88 | 1.14 | 1.16 |
| **×1.00** | 0.95 | 0.90 | **1.06** | 0.94 | 0.97 |
| ×1.25 | 0.98 | 0.90 | 0.91 | 1.19 | 1.13 |
| ×1.50 | 0.97 | 0.90 | 0.80 | 0.79 | 1.00 |
| ×2.00 | 0.91 | 0.66 | 0.59 | 0.50 | 0.55 |

Grid Sharpe spans 0.50–1.19, a range of 1.5 SE. The current setting is **not a lone peak**: it sits
0.33 SE above the mean of its four neighbours, well inside the 1 SE spike-veto threshold, so the
"momentum works" conclusion is not an artefact of parameter choice. Criterion C (inferior) is not
triggered either, which under the pre-registration **forbids** proposing any replacement.

Criterion A and criterion E disagreed, and the pre-registration had already decided how to resolve it.
The base cell's quantised percentile is 77%, two points outside the 25–75% plateau band — but the grid
has 30 cells, so one cell is worth 3.3% and the threshold falls between steps. The unquantised
cross-check, (base − grid median) ÷ SE = **+0.29**, is well inside the 0.5 SE materiality threshold.
Per the pre-registered tie-break the unquantised reading wins: **no material difference, keep
63/126/252 × top 5.** Reporting the contradiction is part of the result.

Two caveats that the aggregate hides. Per fold, the base cell is inside the plateau band in only 3 of
5 folds (87th percentile in 2023, 80th in 2024). And there is one genuine gradient rather than noise:
the ×2.00 row (126/252/504 days) is uniformly the worst, so very long lookbacks do degrade — the
plateau has an edge, and the current setting is not on it.

![momentum robustness](momentum_robustness.png)

### Round 3: does a pooled cross-sectional representation rescue ML?

The LSTM's output layer has one head per ETF, so it carries per-asset parameters and can memorise asset
identity — which is what in-sample IC of 0.77–0.90 against ~zero OOS IC looks like. Round 3 tests a
structural fix rather than another model: stack the data as (date, asset) rows whose features are only
the asset's own momentum/vol, its cross-sectional z-scores that day, and the shared macro pair. **No
asset identity, no one-hot, no embedding.** 20,228 rows instead of 1,556, and the model can only learn
a general rule. LightGBM, shallow and heavily regularised, hyper-parameters fixed in advance; a pooled
ridge on identical rows is the linear baseline.

The pre-registration set the bar at **plain momentum**, not the already-rejected LSTM, and recorded
beforehand that the ranking component of momentum is only worth +0.19 Sharpe (0.40 SE) — the ceiling
this round was competing for.

| | OOS rank-IC | t | Hit | Sharpe ± SE | Null percentile |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Pooled GBDT | +0.063 | 1.22 | 56% | 0.46 ± 0.47 | 19% |
| Pooled ridge (baseline) | +0.130 | 2.55 | 65% | 0.78 ± 0.47 | 78% |
| Momentum | +0.108 | 2.12 | 67% | 1.06 ± 0.48 | 98% |
| LSTM (round 1) | −0.009 | −0.16 | 53% | 0.37 ± 0.47 | 10% |

| Criterion | Rule | Result |
| :--- | :--- | :--- |
| A — predictive power | OOS rank-IC mean > 0 and t > 2 | **Fails** (+0.063, t 1.22) |
| B — vs pooled ridge | paired t of (GBDT − ridge) IC > 2 | **Fails** (t −1.89; the linear model is better) |
| C — vs plain momentum | paired t of (GBDT − momentum) IC > 2 | **Fails** (t −0.93) |
| D — strategy level | ΔSharpe ≥ 1 SE and null percentile ≥ 95% | **Fails** (−0.61 vs SE 0.47; 19th percentile) |
| E — consistency | per-fold reported | GBDT beats momentum on IC in 2 of 5 folds |
| F — materiality | improvement ≥ 0.5 SE | 1.30 SE — material, but in the **wrong direction** |

The gradient boosting failed in a specific and legible way: in 2 of the 5 folds early stopping halted
at iteration 1, meaning the first tree already failed to improve validation error, and across the
round it scored *below its own linear baseline*. That is the textbook small-sample signature — the
model added variance, not signal.

The representation change itself was not useless: the same ridge went from IC +0.095 non-pooled to
+0.130 pooled, and pooled ridge is the only model in this repo whose IC t-statistic clears 2. But two
things stop that from being a finding. Its paired t against non-pooled ridge is only +0.62, and its
paired t against plain momentum is +0.52 — both noise. And its monthly IC correlates 0.67 with
momentum's, i.e. **it largely rediscovered momentum**, at a strictly worse portfolio Sharpe (0.78 vs
1.06). Adopting it on the strength of one favourable number would be exactly the post-hoc selection
the protocol exists to prevent; it would need its own pre-registered round.

![pooled](pooled_report.png)

### What this means

* The model has 36,685 parameters and, in the first fold, 166 overlapping daily samples (about 8 independent months).
  In-sample IC of 0.8–0.9 in every fold with OOS IC around zero is memorisation, not prediction.
* The ridge regression on the same features chooses the strongest regularisation available in every fold and still
  barely beats predicting zero on validation. There is little linearly recoverable signal in these features at a
  20-day horizon; a deeper model does not create signal that is not there.
* The momentum baseline is the only signal with a positive, borderline-significant OOS rank-IC. Its Sharpe advantage
  over SPY is one standard error and should be read as "consistent with a modest momentum premium", not as proof.
* The momentum edge survives its own robustness check, but that check is weak by construction: the 30
  cells share the same 55 months, so their spread understates the true uncertainty. It can say the
  setting is not anomalous within the grid; it cannot say the Sharpe is reliable. That question is
  answered by the ±0.48 standard error and the 98th null percentile above, and the honest reading of
  those is "consistent with a modest momentum premium", not proof.
* The single-split result reported in earlier versions of this README (12-month OOS, Sharpe 2.93, MDD −3.6%) came
  from a strong bull market in which random 5-of-13 portfolios had a median Sharpe of 2.8, from a strategy whose
  Sharpe SE was ±1.2, and from a pipeline that deleted cash months, used no transaction costs, and had no purge
  between validation and test. It should not be cited.

### Honest declarations

* 55 monthly observations give limited power: a true IC of 0.05 would likely go undetected. "No evidence" is not
  "proof of absence".
* The 13 ETFs were chosen with hindsight and the sample ends at the data-fetch date; both flatter every line.
* The first fold's LSTM is trained on 166 samples and is close to a random initialisation; dropping it after the fact
  would be selection, so it stays.
* yfinance adjusted closes are revised with each distribution, so re-runs reproduce the numbers only to a few
  decimal places.
* The 13 ETFs, the 20-day target horizon, the 10 bps cost and the monthly rebalance are all fixed
  assumptions that have not been varied. Only the momentum windows and TOP_N have been stress-tested.
* Three model families have now been rejected on the same 55 months. That is evidence about this
  sample, not a general claim about machine learning on ETF rotation. With SE 0.48, a true edge of
  0.3 Sharpe would very likely go undetected here.

## Getting started

```bash
pip install -r requirements.txt
python main.py
```

```bash
pytest tests/ -q
```

Model weights are written to `saved_models/<fold>/` and are not tracked in git. Outputs in the repo root:
`baseline_vs_spy_performance.png`, `ml_vs_baseline_performance.png`, `diagnostics_report.png`,
`momentum_robustness.png`, `oos_monthly_returns.csv`, `diagnostics_summary.csv`,
`diagnostics_ic_monthly.csv`, `diagnostics_ic_by_fold.csv`, `momentum_grid_*.csv`.

To change the experiment, edit `_config.py`. If you change anything a pre-registration forbids (sequence length,
hidden size, epochs, learning rate, fold boundaries, universe, momentum windows), also change the corresponding
`PRE_REGISTRATION*` string so the diff records it.

### Accounting notes

Two things that are easy to get wrong and silently flatter the results, both covered by tests:
turnover is measured against the **drifted** weights, not by diffing target weights (otherwise a
constant-weight benchmark appears to trade for free), and max drawdown clips the running peak at the
initial capital (otherwise a loss in the opening month is invisible).

## Model

`MomentumLSTM`: single-layer LSTM (hidden 64) over a 60-day window of 67 standardised features → LayerNorm →
Linear(64→32) + GELU → Dropout(0.3) → Linear(32→13). Huber loss with δ = 0.05 (about 1.2σ of the target; the
earlier δ = 1.0 never entered the linear region and was MSE in effect). Adam, lr 0.002, weight decay 1e-4,
ReduceLROnPlateau, 60 epochs, best-validation checkpoint. 20 seeds per fold, predictions averaged.

## Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice. Historical
performance is not indicative of future results. Trading in financial markets involves significant risk of loss.
