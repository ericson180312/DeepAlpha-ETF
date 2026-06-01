"""
基準動能策略回測模組：計算動能特徵、執行傳統動能排名邏輯、向量化回測，並與 SPY 基準進行績效比較。

- 動能特徵計算：3個月、6個月、12個月的滾動報酬率，並綜合成一個動能分數。
- 回測邏輯：每月底選取動能分數最高的前 N 強 ETF，並在下個月持有，計算投資組合的月報酬率。
- 績效指標：年化報酬率、最大回撤、夏普值，並繪製資金曲線與回撤曲線對比圖表。
- 最終將策略與 SPY 的月報酬率合併存檔，供未來 ML 模型對比使用。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_momentum_features(df):
    """
    計算絕對動能指標：3個月(63天)、6個月(126天)、12個月(252天)滾動報酬率
    """
    print("⚙️ 計算動能特徵中...")
    mom_3m = df.pct_change(63)
    mom_6m = df.pct_change(126)
    mom_12m = df.pct_change(252)
    
    # 綜合動能分數：這裡採用 3M, 6M, 12M 的平均，平滑單一週期的極端值
    mom_score = (mom_3m*1 + mom_6m*1 + mom_12m*1) / 3
    return mom_score

def backtest_momentum_strategy(df, mom_score, top_n):
    """
    實作傳統動能排名邏輯與向量化回測
    """
    print(f"🔄 執行月底換股回測 (選取前 {top_n} 強標的)...")
    
    # 1. 將日資料重採樣 (Resample) 為「月底最後一個交易日」
    monthly_prices = df.resample('ME').last()
    monthly_mom = mom_score.resample('ME').last()
    
    # 2. 計算每個月的真實報酬率
    monthly_returns = monthly_prices.pct_change()
    
    # 3. 建立權重矩陣 (Ranking Logic)
    # 在建立 df 時預設 float64 以避免型態警告
    weights = pd.DataFrame(0.0, index=monthly_mom.index, columns=monthly_mom.columns)
    
    # 暫時移除 SPY，不讓它參與動能排名（SPY 僅作對比，不作為選股標的）
    if 'SPY' in monthly_mom.columns:
        ranking_mom = monthly_mom.drop(columns=['SPY'])
    else:
        ranking_mom = monthly_mom

    for date, row in ranking_mom.iterrows():
        valid_scores = row.dropna()
        if len(valid_scores) >= top_n:
            # 1. 抓出分數最高的前 N 檔 (包含標的名稱與分數)
            top_n_scores = valid_scores.nlargest(top_n)
            
            # 2. 核心邏輯：剔除預期 Alpha <= 0 的標的
            positive_scores = top_n_scores[top_n_scores > 0]
            
            # 3. 依照 Alpha 分數比例分配權重
            if not positive_scores.empty:
                # 計算剩餘標的之 Alpha 總和
                total_alpha = positive_scores.sum() + positive_scores.count()
                # 算出每檔標的應分配的權重 (如 0.0182 / 0.0253 = 71.9%)
                proportional_weights = (positive_scores + 1) / total_alpha
                
                # 將權重寫入矩陣
                weights.loc[date, positive_scores.index] = proportional_weights
            else:
                # ⚠️ 邊界情況：如果前 5 名的預期 Alpha 全部 <= 0 怎麼辦？
                # 代表 AI 認為下個月「所有資產都會輸給大盤」。
                # 因為 weights 矩陣預設是 0.0，這裡什麼都不做，
                # 就等同於該月「100% 滿手現金 (持有 0 股)」，完美避險！
                pass

    # 4. 計算投資組合報酬 (⚠️ 避免未來函數的關鍵)
    # 本月底算出的權重，會乘上「下個月」的報酬率，所以 weights 必須 shift(1)
    shifted_weights = weights.shift(1)
    portfolio_returns = (shifted_weights * monthly_returns).sum(axis=1)
    
    # 捨棄初期因動能計算與平移產生的無效資料 (0 或 NaN)
    portfolio_returns = portfolio_returns[portfolio_returns != 0].dropna()
    
    return portfolio_returns, weights

def calculate_performance_metrics(returns, name="Strategy"):
    """
    計算核心績效指標：年化報酬率、最大回撤、夏普值 (模組化以支援重複呼叫)
    """
    # 1. 累積報酬率 (Equity Curve)
    cum_returns = (1 + returns).cumprod()
    
    # 2. 年化報酬率 (CAGR)
    years = len(returns) / 12
    cagr = (cum_returns.iloc[-1] ** (1 / years)) - 1
    
    # 3. 最大回撤 (Maximum Drawdown)
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    # 4. 夏普值 (Sharpe Ratio)
    sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(12)
    
    print(f"--- {name} ---")
    print(f"年化報酬率 (CAGR): {cagr:.2%}")
    print(f"最大回撤 (Max Drawdown): {max_drawdown:.2%}")
    print(f"夏普值 (Sharpe Ratio): {sharpe_ratio:.2f}")
    
    return cum_returns, drawdown

if __name__ == "__main__":
    # 1. 讀取資料
    try:
        df = pd.read_csv("etf_adj_close_clean.csv", index_col="Date", parse_dates=True)
    except FileNotFoundError:
        print("❌ 找不到 etf_adj_close_clean.csv，請先執行資料下載模組。")
        exit()

    if 'SPY' not in df.columns:
        print("❌ 資料中找不到 'SPY' 欄位！請確認已將 SPY 加入下載標的池中。")
        exit()

    # 2. 執行核心邏輯
    mom_scores = calculate_momentum_features(df)
    
    top_n = 5  # 選取前 5 強
    # 取得策略報酬率
    port_returns, weight_matrix = backtest_momentum_strategy(df, mom_scores, top_n)
    
    # 取得 SPY 基準報酬率，並嚴格對齊策略的時間軸
    monthly_spy = df['SPY'].resample('ME').last().pct_change()
    spy_returns = monthly_spy.reindex(port_returns.index)

    # 3. 計算並印出績效比較
    print("\n" + "="*40)
    print(f"🏆 績效報告比較 (總交易月數: {len(port_returns)} 個月)")
    print("="*40)
    strat_cum, strat_dd = calculate_performance_metrics(port_returns, name=f"基準動能策略 (Top {top_n})")
    print("-" * 40)
    spy_cum, spy_dd = calculate_performance_metrics(spy_returns, name="大盤基準 (SPY)")
    print("="*40)
    
    # 4. 繪製並儲存績效圖表
    plt.figure(figsize=(14, 8))
    
    # 繪製資金曲線對比
    plt.subplot(2, 1, 1)
    plt.plot(strat_cum.index, strat_cum.values, label=f'Strategy (Top {top_n} Momentum)', color='blue', linewidth=2)
    plt.plot(spy_cum.index, spy_cum.values, label='SPY (Benchmark)', color='gray', linestyle='--', linewidth=2)
    plt.title('Portfolio Equity Curve: Strategy vs SPY')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 繪製回撤曲線對比
    plt.subplot(2, 1, 2)
    plt.fill_between(strat_dd.index, strat_dd.values, 0, color='red', alpha=0.3, label='Strategy Drawdown')
    plt.plot(spy_dd.index, spy_dd.values, color='gray', linestyle='--', linewidth=1.5, label='SPY Drawdown')
    plt.title('Drawdown Curve: Strategy vs SPY')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_image = "baseline_vs_spy_performance.png"
    plt.savefig(output_image)
    print(f"\n📈 績效比較圖表已儲存至: {output_image}")
    
    # 將兩者的月報酬率合併存檔，供未來 ML 模型對比
    comparison_df = pd.DataFrame({
        'Strategy_Return': port_returns,
        'SPY_Return': spy_returns
    })
    comparison_df.to_csv("baseline_monthly_returns.csv")
    print("💾 策略與大盤報酬率已合併儲存至: baseline_monthly_returns.csv")