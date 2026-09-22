"""
策略回測模組：將訓練好的 LSTM 模型應用於歷史資料，生成 AI 預測分數，並與傳統基準策略進行績效比較。
- AI 預測分數生成：使用訓練好的 LSTM 模型對每個月底的資料進行推論，生成一個分數矩陣，代表每個 ETF 在下個月的預期表現。
- 回測邏輯：將 AI 預測分數餵給原本的傳統動能回測框架，選取分數最高的前 N 強 ETF，計算投資組合的月報酬率。
- 嚴格樣本外測試：將資料切分為訓練集、驗證集和樣本外測試集，確保模型在未見過的資料上進行回測，避免未來函數 (Look-Ahead Bias)。
- 績效比較：計算年化報酬率、最大回撤、夏普值，並繪製資金曲線與回撤曲線對比圖表，展示 AI 策略相對於傳統基準和大盤的優劣。
"""

import pandas as pd
import numpy as np
import torch
import _config
import matplotlib.pyplot as plt
from _3_ml_data_pipeline import generate_ml_features  # 引入寫好的特徵工程
from _4_lstm_model import MomentumLSTM                 # 引入模型架構
from _2_baseline_performance import backtest_momentum_strategy, calculate_performance_metrics, month_end_returns

# ==========================================
# 1. 執行 LSTM 集成模型推論 (Ensemble Inference)
# ==========================================
def generate_ml_scores(df, model_paths, seq_length=60):
    print(f"🧠 啟動 AI 投資委員會進行回測 (共 {len(model_paths)} 個模型)...")
    
    X, _ = generate_ml_features(df)
    
    # 標準化參數
    train_end = int(len(X) * 0.6) # 依照你的 60% 訓練集比例
    train_mean = X.iloc[:train_end].mean()
    train_std = X.iloc[:train_end].std() + 1e-8
    X_scaled = (X - train_mean) / train_std
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    input_size = X.shape[1]
    output_size = len([c for c in df.columns if c != 'SPY'])
    
    # 🌟 載入所有的模型到清單中
    models = []
    for path in model_paths:
        model = MomentumLSTM(input_size=input_size, hidden_size=_config.HIDDEN_SIZE, output_size=output_size, num_layers=1)
        try:
            model.load_state_dict(torch.load(path, map_location=device))
            model.to(device)
            model.eval()
            models.append(model)
        except FileNotFoundError:
            print(f"❌ 找不到模型權重檔: {path}，請先執行訓練！")
            exit()
            
    print(f"✅ 成功載入 {len(models)} 個 LSTM 模型權重！")
    
    # 找出每個月最後一個實際交易日
    monthly_dates = df.groupby(df.index.strftime('%Y-%m')).tail(1).index
    ml_scores = pd.DataFrame(index=monthly_dates, columns=[c for c in df.columns if c != 'SPY'])
    
    with torch.no_grad():
        for date in monthly_dates:
            if date in X_scaled.index:
                idx = X_scaled.index.get_loc(date)
                if idx >= seq_length - 1:
                    window = X_scaled.iloc[idx - seq_length + 1 : idx + 1]
                    tensor_in = torch.tensor(window.values, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    # 🌟 讓 20 個模型各自預測，然後平均
                    ensemble_preds = []
                    for model in models:
                        pred = model(tensor_in).cpu().numpy()[0]
                        ensemble_preds.append(pred)
                        
                    # 取平均分數 (axis=0 代表對各個標的的分數分別取平均)
                    avg_pred = np.mean(ensemble_preds, axis=0)
                    
                    ml_scores.loc[date] = avg_pred
                    
    ml_scores = ml_scores.dropna(how='all').astype(float)
    print("✅ AI 委員會評分完畢！")
    return ml_scores

if __name__ == "__main__":
    # ==========================================
    # 2. 讀取資料與基準策略績效
    # ==========================================
    df = pd.read_csv("etf_adj_close_clean.csv", index_col="Date", parse_dates=True)
    
    if 'SPY' not in df.columns:
        print("❌ 資料中找不到 'SPY'，請確認資料管線。")
        exit()

    # 讀取上一階段存下來的 Baseline 與 SPY 報酬率
    try:
        baseline_df = pd.read_csv("baseline_monthly_returns.csv", index_col="Date", parse_dates=True)
        baseline_returns = baseline_df['Strategy_Return']
        spy_returns = baseline_df['SPY_Return']
    except FileNotFoundError:
        print("❌ 找不到 baseline_monthly_returns.csv，請先執行基準策略回測。")
        exit()

    # ==========================================
    # 3. 執行 AI 策略回測
    # ==========================================
    top_n = _config.TOP_N # 一樣選取前 5 強
    
    # 定義要載入的 Ensemble 模型清單
    ensemble_paths = [f'saved_models/ensemble_lstm_seed_{seed}.pth' for seed in range(42, 62)]
    
    # 取得 AI 委員會的平均預測分數
    ml_scores = generate_ml_scores(df, model_paths=ensemble_paths)
    
    # 將 ml_scores 餵給原本的 traditional 回測框架
    ml_port_returns, ml_weights = backtest_momentum_strategy(
        df, ml_scores, top_n=top_n,
        min_score=_config.ML_MIN_SCORE, fallback=_config.ML_FALLBACK, cost_bps=_config.COST_BPS
    )
    rf_returns = month_end_returns(df)[0][_config.RF_TICKER]
    
    # 對齊所有策略的時間軸 (取交集，確保起跑點一致)
    common_index = ml_port_returns.index.intersection(baseline_returns.index)

    # 🚨🚨🚨 新增：嚴格切分「樣本外測試期 (Out-of-Sample)」 🚨🚨🚨
    # 調整測試集比例：前 60% 訓練，20% 驗證，最後 20% 留給 OOS 測試
    X, _ = generate_ml_features(df)
    # 計算 Train + Val 的比例總和，也就是 Test Set 的起點
    train_val_ratio = _config.TRAIN_RATIO + _config.VAL_RATIO 
    test_start_idx = int(len(X) * train_val_ratio) 
    test_start_date = X.index[test_start_idx]
    
    print(f"\n🔍 嚴格檢驗：模型未看過的 Test Set 起始日期為 {test_start_date.strftime('%Y-%m-%d')}")
    
    # 僅保留「決策日」落在 Test Set 起始日之後的持有月 (持有月索引減一個月 = 決策日)
    oos_index = common_index[(common_index - pd.offsets.MonthEnd(1)) >= test_start_date]

    # 更新回測報酬率變數為純樣本外資料
    ml_port_returns = ml_port_returns.loc[oos_index]
    baseline_returns = baseline_returns.loc[oos_index]
    spy_returns = spy_returns.loc[oos_index]

    if len(oos_index) == 0:
        print("❌ 錯誤：樣本外期間太短，沒有足夠的月底交易日可以回測！建議增加資料總長度。")
        exit()
        
    # ==========================================
    # 📈 額外功能：印出樣本外 (OOS) 每個月的 AI 選股名單
    # ==========================================
    print("\n" + "🌟"*25)
    print(f"🤖 AI 策略各月份 Top {top_n} 標的與評分 (樣本外期間)")
    print("🌟"*25)
    
# 🌟 關鍵修正：將分數矩陣也對齊到「日曆月底」，解決週末日期對不上的問題
    ml_scores_aligned = ml_scores.resample('ME').last()
    
    for date in oos_index:
        # 改從對齊後的矩陣尋找
        if date in ml_scores_aligned.index:
            valid_scores = ml_scores_aligned.loc[date].dropna()
            
            # 使用 nlargest 抓出分數最高的 Top N 檔
            top_tickers = valid_scores.nlargest(top_n)
            
            print(f"📅 換股日: {date.strftime('%Y-%m-%d')}")
            for rank, (ticker, score) in enumerate(top_tickers.items(), 1):
                # {ticker:<5} 代表向左對齊並保留 5 個字元寬度，讓排版更整齊
                # {score:+.4f} 代表強制顯示正負號，並取到小數點後 4 位
                print(f"   第 {rank} 名: {ticker:<5} | 預期 Alpha 分數: {score:+.4f}")
            print("-" * 40)
    
    # ==========================================
    # 4. 終極績效比較與視覺化
    # ==========================================
    print("\n" + "="*50)
    print(f"🤖 AI 策略 vs 傳統基準 vs 大盤 (交易月數: {len(oos_index)}, 成本 {_config.COST_BPS} bps)")
    print("="*50)
    ml_cum, ml_dd = calculate_performance_metrics(ml_port_returns, name=f"LSTM AI 策略 (Top {top_n})", rf=rf_returns)
    print("-" * 50)
    base_cum, base_dd = calculate_performance_metrics(baseline_returns, name=f"傳統基準動能 (Top {top_n})", rf=rf_returns)
    print("-" * 50)
    spy_cum, spy_dd = calculate_performance_metrics(spy_returns, name="大盤基準 (SPY)", rf=rf_returns)
    print("="*50)
    
    # 繪圖
    plt.figure(figsize=(16, 10))
    
    # 資金曲線
    plt.subplot(2, 1, 1)
    plt.plot(ml_cum.index, ml_cum.values, label='LSTM AI Strategy', color='purple', linewidth=2.5)
    plt.plot(base_cum.index, base_cum.values, label='Baseline Momentum', color='blue', linewidth=1.5, alpha=0.7)
    plt.plot(spy_cum.index, spy_cum.values, label='SPY (Benchmark)', color='gray', linestyle='--', linewidth=1.5)
    plt.title('Ultimate Showdown: AI Strategy vs Baseline vs SPY')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 回撤曲線
    plt.subplot(2, 1, 2)
    plt.fill_between(ml_dd.index, ml_dd.values, 0, color='purple', alpha=0.2, label='AI Drawdown')
    plt.plot(base_dd.index, base_dd.values, color='blue', linewidth=1, alpha=0.5, label='Baseline Drawdown')
    plt.plot(spy_dd.index, spy_dd.values, color='gray', linestyle='--', linewidth=1, label='SPY Drawdown')
    plt.title('Drawdown Comparison')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_image = "ml_vs_baseline_performance.png"
    plt.savefig(output_image)
    print(f"\n📈 終極比較圖表已儲存至: {output_image}")