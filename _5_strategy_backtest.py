"""
策略回測模組 (walk-forward)：用每個 fold 自己的 ensemble 對該 fold 的 test 期推論，把各 fold 的樣本外分數
接起來，餵給 _2 的回測框架，與 ridge 線性基準、傳統動能、SPY 比較。
- 樣本外的定義：每個月底決策日的分數，都來自「只看過該日之前資料」的模型；第一個 fold 的 test 起點之後全部是 OOS。
- 四條線用同一套會計 (現金月計 0%、扣成本、扣 rf 的 Sharpe)，見 _2_baseline_performance.py。
- 逐 fold (逐年) 的結果與匯總並列，不以單一 fold 下結論。
"""

import os
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

import _config
from _2_baseline_performance import (calculate_momentum_features, backtest_momentum_strategy,
                                     performance_metrics, month_end_returns)
from _3_ml_data_pipeline import generate_ml_features, fold_specs, make_fold, month_end_positions
from _4_lstm_model import MomentumLSTM, model_paths
from _4b_ridge_baseline import ridge_scores

PALETTE = {'lstm': '#2a78d6', 'ridge': '#eb6834', 'mom': '#1baf7a', 'spy': '#52514e'}


# ==========================================
# 1. Ensemble 推論
# ==========================================
def usable_folds(X, Y):
    """展開設定檔的 fold，略過 train 樣本不足者。回傳 [(name, fold_dict), ...]。"""
    folds = [(name, make_fold(X, Y, ts, te)) for name, ts, te in fold_specs(X)]
    return [(n, f) for n, f in folds if len(f['train_pos']) >= _config.WF_MIN_TRAIN_SAMPLES]


def load_fold_models(fold_name, input_size, output_size, device):
    models = []
    for path in model_paths(fold_name):
        if not os.path.exists(path):
            raise FileNotFoundError(f"找不到模型權重檔 {path}，請先執行 _4_lstm_model.py 訓練。")
        model = MomentumLSTM(input_size, _config.HIDDEN_SIZE, output_size)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        models.append(model)
    return models


def score_positions(models, X_scaled, positions, seq_length=_config.SEQ_LENGTH, device='cpu'):
    """對一組視窗結束位置推論，回傳 (n_models, n_positions, n_assets)。"""
    windows = np.stack([X_scaled.values[p - seq_length + 1: p + 1] for p in positions])
    tensor_in = torch.tensor(windows, dtype=torch.float32).to(device)
    with torch.no_grad():
        return np.stack([m(tensor_in).cpu().numpy() for m in models])


def generate_ml_scores(X, Y, folds, assets, where='test'):
    """
    每個 fold 用自己的 ensemble 對 where 段 ('test' = 樣本外, 'train' = 樣本內診斷用) 的月底決策日推論。
    回傳 (ensemble 平均分數 [決策日 x 標的], {成員序號: 該成員分數})。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    ens_pieces, member_pieces = [], defaultdict(list)
    for name, fold in folds:
        pos = month_end_positions(X, fold[f'{where}_pos'])
        if len(pos) == 0:
            continue
        models = load_fold_models(name, X.shape[1], len(assets), device)
        preds = score_positions(models, fold['X_scaled'], pos, device=device)
        idx = X.index[pos]
        ens_pieces.append(pd.DataFrame(preds.mean(axis=0), index=idx, columns=assets))
        for i in range(preds.shape[0]):
            member_pieces[i].append(pd.DataFrame(preds[i], index=idx, columns=assets))
    ensemble = pd.concat(ens_pieces).sort_index()
    members = {i: pd.concat(v).sort_index() for i, v in member_pieces.items()}
    return ensemble, members


# ==========================================
# 2. 逐 fold 績效
# ==========================================
def fold_of_holding_month(dates, folds):
    """持有月 -> 所屬 fold 名稱 (依決策日 = 前一個月底)。"""
    decision = pd.DatetimeIndex(dates) - pd.offsets.MonthEnd(1)
    out = pd.Series(None, index=dates, dtype=object)
    for name, fold in folds:
        mask = (decision >= fold['test_start']) & (decision < fold['test_end'])
        out[mask] = name
    return out


def per_fold_table(series_dict, fold_labels, rf, metric):
    """列 = fold，欄 = 策略，值 = 指定指標。"""
    rows = {}
    for name in fold_labels.dropna().unique():
        idx = fold_labels.index[fold_labels == name]
        rows[name] = {s: performance_metrics(r.reindex(idx).dropna(), rf)[metric] for s, r in series_dict.items()}
    return pd.DataFrame(rows).T


if __name__ == "__main__":
    df = pd.read_csv("etf_adj_close_clean.csv", index_col="Date", parse_dates=True)
    for col in ('SPY', _config.RF_TICKER):
        if col not in df.columns:
            print(f"❌ 資料中找不到 '{col}'，請確認資料管線。")
            exit()

    X, Y = generate_ml_features(df)
    assets = [c.replace('_Target', '') for c in Y.columns]
    folds = usable_folds(X, Y)
    top_n = _config.TOP_N

    # --- 三種分數 ---
    print(f"\n🧠 Walk-forward 推論：{len(folds)} 個 fold × {_config.N_MODELS} 個成員")
    ml_scores, _ = generate_ml_scores(X, Y, folds, assets)
    ridge_sc, ridge_info = ridge_scores(X, Y, folds, assets)
    mom_scores = calculate_momentum_features(df)
    print(f"✅ LSTM 分數 {len(ml_scores)} 個決策日 ({ml_scores.index[0].date()} ~ {ml_scores.index[-1].date()})")

    # --- 回測 (同一套會計) ---
    ml_ret, ml_w = backtest_momentum_strategy(df, ml_scores, top_n, _config.ML_MIN_SCORE, _config.ML_FALLBACK, _config.COST_BPS)
    ridge_ret, _ = backtest_momentum_strategy(df, ridge_sc, top_n, _config.ML_MIN_SCORE, _config.ML_FALLBACK, _config.COST_BPS)
    base_ret, _ = backtest_momentum_strategy(df, mom_scores, top_n, _config.BASE_MIN_SCORE, _config.BASE_FALLBACK, _config.COST_BPS)
    monthly_returns, last_month_complete = month_end_returns(df)
    if not last_month_complete:
        print(f"⚠️ 最後一個月不完整 (資料止於 {df.index[-1].date()})，該月不納入回測。")
    rf = monthly_returns[_config.RF_TICKER]

    oos_start = folds[0][1]['test_start']
    oos_index = ml_ret.index[(ml_ret.index - pd.offsets.MonthEnd(1)) >= oos_start]
    oos_index = oos_index.intersection(ridge_ret.index).intersection(base_ret.index)
    if len(oos_index) == 0:
        print("❌ 沒有任何樣本外持有月，請檢查 fold 設定與資料長度。")
        exit()

    series = {
        'LSTM ensemble': ml_ret.reindex(oos_index),
        'Ridge (linear)': ridge_ret.reindex(oos_index),
        'Momentum': base_ret.reindex(oos_index),
        'SPY': monthly_returns['SPY'].reindex(oos_index),
    }

    # --- 各月 AI 選股 (一行一個月) ---
    print("\n" + "🌟" * 25)
    print(f"🤖 LSTM 各月份 Top {top_n} 標的 (樣本外，決策日 -> 持有下月)")
    print("🌟" * 25)
    for date, row in ml_scores.iterrows():
        top = row.nlargest(top_n)
        held = [f"{k}({v:+.3f})" for k, v in top.items()]
        n_pos = int((top > _config.ML_MIN_SCORE).sum())
        tail = "" if n_pos == top_n else f"  ← {top_n - n_pos} 槽改持 {_config.ML_FALLBACK}"
        print(f"  {date.strftime('%Y-%m-%d')}: " + ", ".join(held) + tail)

    # --- 匯總績效 ---
    print("\n" + "=" * 60)
    print(f"🤖 樣本外匯總 ({len(oos_index)} 個月, {oos_index[0].strftime('%Y-%m')} ~ {oos_index[-1].strftime('%Y-%m')}, "
          f"成本 {_config.COST_BPS} bps, Sharpe 扣 {_config.RF_TICKER})")
    print("=" * 60)
    summary = pd.DataFrame({name: performance_metrics(r, rf) for name, r in series.items()}).T
    print(summary.to_string(formatters={'CAGR': '{:.2%}'.format, 'MDD': '{:.2%}'.format, 'n': '{:.0f}'.format},
                            float_format=lambda v: f"{v:.2f}"))

    # --- 逐 fold ---
    fold_labels = fold_of_holding_month(oos_index, folds)
    print("\n📆 逐 fold Sharpe (扣 rf)：")
    print(per_fold_table(series, fold_labels, rf, 'Sharpe').to_string(float_format=lambda v: f"{v:6.2f}"))
    print("\n📆 逐 fold CAGR：")
    print(per_fold_table(series, fold_labels, rf, 'CAGR').to_string(float_format=lambda v: f"{v:7.1%}"))
    print("\n📐 Ridge 每個 fold 選到的 alpha 與 val MSE (zero_pred = 永遠預測 0)：")
    print(ridge_info[['alpha', 'val_mse', 'zero_pred_val_mse']].to_string(float_format=lambda v: f"{v:.6f}"))

    # --- 繪圖 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), facecolor='#fcfcfb')
    styles = {'LSTM ensemble': (PALETTE['lstm'], '-', 2.5), 'Ridge (linear)': (PALETTE['ridge'], '-', 1.8),
              'Momentum': (PALETTE['mom'], '-', 1.8), 'SPY': (PALETTE['spy'], '--', 1.5)}
    for name, r in series.items():
        color, ls, lw = styles[name]
        cum = (1 + r).cumprod()
        dd = cum / cum.cummax().clip(lower=1.0) - 1
        ax1.plot(cum.index, cum.values, label=name, color=color, linestyle=ls, linewidth=lw)
        ax2.plot(dd.index, dd.values, label=name, color=color, linestyle=ls, linewidth=lw)
    ax1.set_title(f'Walk-forward out-of-sample equity ({oos_index[0].year}-{oos_index[-1].year}, net of {_config.COST_BPS} bps)', loc='left')
    ax1.set_ylabel('Cumulative Return'); ax1.legend(frameon=False); ax1.grid(True, alpha=0.3)
    ax2.set_title('Drawdown', loc='left'); ax2.set_ylabel('Drawdown'); ax2.legend(frameon=False); ax2.grid(True, alpha=0.3)
    for ax in (ax1, ax2):
        ax.set_facecolor('#fcfcfb')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    plt.tight_layout()
    output_image = "ml_vs_baseline_performance.png"
    plt.savefig(output_image, dpi=110)
    print(f"\n📈 比較圖表已儲存至: {output_image}")

    pd.DataFrame(series).to_csv("oos_monthly_returns.csv")
    print("💾 樣本外月報酬已儲存至: oos_monthly_returns.csv")
