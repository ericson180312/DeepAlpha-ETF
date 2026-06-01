# 📈 DeepAlpha-ETF: Ensemble LSTM & Macro-Aware Momentum Strategy

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)

DeepAlpha-ETF is an end-to-end quantitative trading pipeline that applies **Deep Learning (Ensemble LSTM)** and **Macroeconomic Feature Engineering** to optimize ETF rotation strategies. It aims to outperform traditional momentum strategies by dynamically adjusting risk exposure and predicting excess returns (Alpha) based on market volatility and trend conditions.

## ✨ Key Features

* **Macro-Aware Feature Engineering:** Integrates global market features (SPY 200-day MA deviation, rolling volatility) with local ETF features (momentum, volatility) to give the model a holistic view of the market regime.
* **Targeting Alpha:** Instead of predicting absolute returns, the model is trained to predict **Excess Return (Alpha)** against the SPY benchmark, naturally forcing it to seek outperforming assets.
* **Ensemble Learning (AI Committee):** Trains multiple LSTM models with different random seeds and averages their predictions. This significantly reduces random variance and prevents overfitting, ensuring highly stable out-of-sample (OOS) performance.
* **Signal-Weighted Allocation:** Automatically drops assets with negative expected Alpha and dynamically allocates capital proportionally to positive Alpha scores, effectively acting as an automated de-leveraging mechanism during market crashes.
* **Strict Out-of-Sample (OOS) Backtesting:** Completely isolates the test set from the training and validation sets, eliminating Look-Ahead Bias.

## 🗂️ Project Structure

The project is modularized for easy maintenance and execution. All hyper-parameters are centrally managed.

```text
DeepAlpha-ETF/
│
├── _config.py                     # Centralized global hyper-parameters (Train/Test split, epochs, etc.)
├── main.py                        # Master script to execute the entire pipeline sequentially
│
├── _1_fetch_and_clean_data.py     # Data ingestion and cleaning (using yfinance)
├── _2_baseline_performance.py     # Traditional multi-timeframe momentum baseline
├── _3_ml_data_pipeline.py         # Feature engineering and PyTorch DataLoader creation
├── _4_lstm_model.py               # LSTM architecture and Ensemble training loop
└── _5_strategy_backtest.py        # OOS backtesting, performance metrics (Sharpe, Drawdown), and plotting

```

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ installed. Install the required libraries:

```bash
pip install yfinance pandas numpy torch matplotlib

```

### 2. Configuration

You can tweak the parameters (e.g., train/test ratios, lookback windows, hidden size) in `_config.py` before running the pipeline.

### 3. Run the Pipeline

Execute the master script to run the entire process from data downloading to backtest visualization:

```bash
python main.py

```

## 📊 Performance & Visualization

The pipeline will generate two main charts in the root directory:

* **`baseline_vs_spy_performance.png`**: Traditional Momentum vs. SPY.
* **`ml_vs_baseline_performance.png`**: The Ultimate Showdown (Ensemble LSTM vs. Traditional Baseline vs. SPY).

> **Note:** The Ensemble LSTM strategy typically demonstrates superior downside protection (drastically lower Max Drawdown) and a higher Sharpe Ratio during out-of-sample testing compared to traditional unhedged momentum.

## 🧠 Model Architecture (MomentumLSTM)

* **Input Layer:** Processes time-series sequences (default: 60 days lookback).
* **LSTM Layer:** Captures temporal dependencies in price and macro data.
* **Layer Normalization:** Stabilizes distribution shifts common in financial time-series.
* **Non-Linear Fusion:** Uses a fully connected layer with GELU activation for feature crossing (Macro AND Micro conditions).
* **Dropout:** Applied at 30% to prevent overfitting.
* **Loss Function:** HuberLoss to provide robustness against extreme market outliers (fat tails).

## ⚠️ Disclaimer

This project is for educational and research purposes only. It does not constitute financial advice. Historical performance is not indicative of future results. Trading in financial markets involves significant risk of loss. Always test algorithms thoroughly in a paper-trading environment before deploying real capital.
