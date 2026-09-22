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
rank-IC of +0.11 (t = 2.1) and a Sharpe at the 98th percentile of the same null. Details below.

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

`main.py` runs them in order (about 6 minutes on a GPU). All parameters live in `_config.py`.

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

| Strategy | Rank-IC (hit rate) | CAGR | Max DD | Sharpe ± SE | Percentile vs random 5-of-13 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Ensemble LSTM | −0.01 (53%) | 8.7% | −23.4% | 0.37 ± 0.47 | 10% |
| Ridge (same features) | +0.09 (64%) | 9.4% | −18.9% | 0.42 ± 0.47 | 14% |
| Momentum (3/6/12m, top 5) | +0.11 (67%) | 17.7% | −8.9% | 1.06 ± 0.48 | 98% |
| SPY | — | 13.9% | −20.3% | 0.66 ± 0.47 | — |
| Equal-weight 13 ETFs | — | 12.6% | −14.2% | 0.83 ± 0.47 | 84% |

### Per fold: the LSTM memorises its training period and forgets it out of sample

| Fold (test year) | Train samples | In-sample rank-IC | OOS rank-IC | OOS Sharpe LSTM / Ridge / Momentum / SPY |
| :--- | :--- | :--- | :--- | :--- |
| 2022 | 166 | +0.90 | −0.07 | −0.18 / 0.08 / 0.02 / −0.32 |
| 2023 | 418 | +0.82 | −0.06 | −0.18 / −0.16 / 0.10 / 1.06 |
| 2024 | 669 | +0.86 | +0.18 | 1.40 / 1.22 / 1.78 / 1.78 |
| 2025 | 919 | +0.84 | −0.04 | 0.85 / 0.85 / 3.29 / 1.08 |
| 2026 (Jan–Aug) | 1169 | +0.77 | −0.09 | 0.69 / 0.71 / 0.47 / 0.94 |

### Pre-registered verdict (computed by `_6_diagnostics.py`)

| Criterion | Rule | Result |
| :--- | :--- | :--- |
| A — predictive power | OOS rank-IC mean > 0 and t > 2 | **Fails** (mean −0.009, t −0.16) |
| B — vs linear baseline | paired t of (LSTM − ridge) monthly IC > 2 | **Fails** (t −1.95; ridge is better) |
| C — strategy level | Sharpe(LSTM) − Sharpe(momentum) ≥ 1 SE and null percentile ≥ 95% | **Fails** (−0.69 vs SE 0.47; 10th percentile) |
| D — consistency | per-fold results reported alongside the aggregate | 1 of 5 folds has positive OOS IC (2024); reported above |

![diagnostics](diagnostics_report.png)

### What this means

* The model has 36,685 parameters and, in the first fold, 166 overlapping daily samples (about 8 independent months).
  In-sample IC of 0.8–0.9 in every fold with OOS IC around zero is memorisation, not prediction.
* The ridge regression on the same features chooses the strongest regularisation available in every fold and still
  barely beats predicting zero on validation. There is little linearly recoverable signal in these features at a
  20-day horizon; a deeper model does not create signal that is not there.
* The momentum baseline is the only signal with a positive, borderline-significant OOS rank-IC. Its Sharpe advantage
  over SPY is one standard error and should be read as "consistent with a modest momentum premium", not as proof.
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

## Getting started

```bash
pip install -r requirements.txt
python main.py
```

Model weights are written to `saved_models/<fold>/` and are not tracked in git. Outputs in the repo root:
`baseline_vs_spy_performance.png`, `ml_vs_baseline_performance.png`, `diagnostics_report.png`,
`oos_monthly_returns.csv`, `diagnostics_summary.csv`, `diagnostics_ic_monthly.csv`, `diagnostics_ic_by_fold.csv`.

To change the experiment, edit `_config.py`. If you change anything the pre-registration forbids (sequence length,
hidden size, epochs, learning rate, fold boundaries, universe), also change `PRE_REGISTRATION` so the diff records it.

## Model

`MomentumLSTM`: single-layer LSTM (hidden 64) over a 60-day window of 67 standardised features → LayerNorm →
Linear(64→32) + GELU → Dropout(0.3) → Linear(32→13). Huber loss with δ = 0.05 (about 1.2σ of the target; the
earlier δ = 1.0 never entered the linear region and was MSE in effect). Adam, lr 0.002, weight decay 1e-4,
ReduceLROnPlateau, 60 epochs, best-validation checkpoint. 20 seeds per fold, predictions averaged.

## Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice. Historical
performance is not indicative of future results. Trading in financial markets involves significant risk of loss.
