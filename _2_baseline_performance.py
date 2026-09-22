"""
基準動能策略回測模組：計算動能特徵、執行傳統動能排名邏輯、向量化回測，並與 SPY 基準進行績效比較。

- 動能特徵計算：_config.MOM_WINDOWS 各窗口滾動報酬率的等權平均。
- 回測邏輯：每月底選取分數最高的前 N 強，分數 <= min_score 的槽位改持 fallback（預設現金），其餘等權。
- 會計口徑（_5 與 _6 共用同一套函式，確保三條線可比）：
    * 現金月（權重全 0）計 0% 報酬，不從序列中刪除；「尚無有效訊號」的月份才剔除。
    * 若資料最後一天不是該月最後一個營業日，該月視為不完整並剔除。
    * 交易成本 = 每個持有月的成交金額比例 Σ|目標權重 − 漂移後權重| × COST_BPS（首月建倉算全額）。
      跟漂移比而非直接 diff 目標權重，否則會低估「目標權重穩定」那幾條線（等權、buy & hold）的成本。
    * Sharpe 以扣除 RF_TICKER 月報酬後的超額報酬計算，並附 Lo (2002) 的 iid 標準誤。
- 最終將策略與 SPY 的月報酬率合併存檔，供 ML 模型對比使用。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import _config


# ==========================================
# 1. 時間軸
# ==========================================
def month_end_returns(df):
    """
    日資料 -> 以日曆月底為索引的月報酬。
    若資料最後一天不是該月最後一個營業日，視為不完整月份並剔除。回傳 (月報酬, 最後一月是否完整)。
    """
    last = df.index[-1]
    complete = pd.offsets.BMonthEnd().rollforward(last) == last
    px = df.resample('ME').last()
    if not complete:
        px = px.iloc[:-1]
    return px.pct_change(), complete


# ==========================================
# 2. 動能特徵
# ==========================================
def calculate_momentum_features(df, windows=None, verbose=True):
    """
    絕對動能分數 = 各回看窗口滾動報酬率的等權平均 (預設 _config.MOM_WINDOWS = 63/126/252 天)。
    多個窗口平均是為了平滑單一週期的極端值；窗口本身是否坐在高原上由 _7_momentum_robustness.py 檢驗。
    """
    windows = list(_config.MOM_WINDOWS if windows is None else windows)
    if verbose:
        print(f"⚙️ 計算動能特徵中 (窗口 {windows} 天)...")
    return sum(df.pct_change(w) for w in windows) / len(windows)


# ==========================================
# 3. 權重與回測
# ==========================================
def build_weights(score_df, top_n, min_score=0.0, fallback=None):
    """
    分數矩陣 (日或月頻) -> 以決策日 (日曆月底) 為索引的權重矩陣。

    - 每個決策日取分數最高的 top_n 檔，每檔佔 1/top_n。
    - 分數 <= min_score 的槽位不持有該標的：fallback=None 時留作現金 (權重 0)，
      否則把該槽位配置到 fallback 標的 (例如預測 alpha <= 0 就改持 SPY，與「alpha 相對 SPY」的目標定義一致)。
    - SPY 只作為 fallback，不參與排名。
    - 有效分數不足 top_n 的決策日整列為 NaN，代表「尚無訊號」——與「有訊號但選擇空手」(整列 0) 區分開。
    """
    monthly_scores = score_df.resample('ME').last()
    ranking = monthly_scores.drop(columns=['SPY'], errors='ignore')
    columns = list(ranking.columns) + ([fallback] if fallback is not None and fallback not in ranking.columns else [])
    weights = pd.DataFrame(np.nan, index=ranking.index, columns=columns)

    for date, row in ranking.iterrows():
        valid = row.dropna()
        if len(valid) < top_n:
            continue
        weights.loc[date] = 0.0
        selected = valid.nlargest(top_n)
        kept = selected[selected > min_score]
        if len(kept):
            weights.loc[date, kept.index] = 1.0 / top_n
        residual = (top_n - len(kept)) / top_n
        if fallback is not None and residual > 0:
            weights.loc[date, fallback] += residual
    return weights


def drifted_weights(prev_weights, month_return):
    """
    上個決策日的目標權重，經過一個月的報酬漂移後的實際權重。
    未投資的部分視為現金 (0% 報酬)，仍佔總市值，所以分母含它。
    """
    grown = prev_weights * (1 + month_return)
    total = grown.sum() + (1 - prev_weights.sum())     # 資產市值 + 現金
    return grown / total if total else grown


def traded_fraction(weights, monthly_returns):
    """
    每個決策日的成交金額比例 Σ|目標權重 − 漂移後權重|，首月建倉算全額。

    ⚠️ 必須跟漂移比，不能直接 diff 目標權重：等權組合的目標權重每月相同，
    diff 會得到 0，但實際上每月再平衡都要把漲多的賣掉、跌多的買回。
    用 diff 會系統性低估「目標權重穩定」那幾條線的成本 (等權、buy & hold)。
    """
    wv = weights.dropna(how='all')
    cols = wv.columns.intersection(monthly_returns.columns)
    traded = pd.Series(np.nan, index=wv.index)
    traded.iloc[0] = wv.iloc[0].abs().sum()

    for prev_date, date in zip(wv.index[:-1], wv.index[1:]):
        # 決策日 prev_date 的權重，持有到決策日 date 為止 (即 date 當月的報酬)
        r = monthly_returns.loc[date, cols] if date in monthly_returns.index else None
        if r is None:
            traded.loc[date] = wv.loc[date, cols].sub(wv.loc[prev_date, cols]).abs().sum()
            continue
        drift = drifted_weights(wv.loc[prev_date, cols], r)
        traded.loc[date] = wv.loc[date, cols].sub(drift).abs().sum()
    return traded


def constant_weights(assets, index, weight=None):
    """
    固定目標權重的 benchmark (等權組合、buy & hold) 的權重矩陣，
    讓它們跟策略走同一套會計 (含再平衡換手成本)。weight 省略時為等權。
    """
    w = 1.0 / len(assets) if weight is None else weight
    return pd.DataFrame(w, index=index, columns=list(assets))


def backtest_momentum_strategy(df, score_df, top_n, min_score=0.0, fallback=None, cost_bps=0.0):
    """
    月底換股回測。回傳 (以持有月為索引的淨報酬, 以決策日為索引的權重)。
    本月底的權重乘上「下個月」的報酬 (shift(1))，避免未來函數。
    """
    print(f"🔄 執行月底換股回測 (Top {top_n}, min_score={min_score}, fallback={fallback or '現金'}, cost={cost_bps} bps)...")
    monthly_returns, _ = month_end_returns(df)
    weights = build_weights(score_df, top_n, min_score, fallback)

    w = weights.reindex(monthly_returns.index).shift(1)      # 決策日 -> 持有月
    cols = w.columns.intersection(monthly_returns.columns)
    gross = (w[cols] * monthly_returns[cols]).sum(axis=1, min_count=1)   # 現金月 = 0, 無決策 = NaN
    traded = traded_fraction(weights, monthly_returns).reindex(monthly_returns.index).shift(1)
    net = gross - traded * cost_bps / 1e4
    return net.dropna(), weights


# ==========================================
# 4. 績效指標
# ==========================================
def performance_metrics(returns, rf=None, weights=None):
    """
    CAGR、最大回撤、年化 Sharpe 及其標準誤 (Lo 2002 的 iid 近似)。
    rf 給定時 (月頻序列)，Sharpe 以超額報酬計算。
    weights 給定時附上「平均曝險」與「每單位曝險的 CAGR」——報酬比較沒有並列曝險是套套邏輯
    (曝險高賺得多不是發現)。
    """
    r = returns.dropna()
    cum = (1 + r).cumprod()
    cagr = cum.iloc[-1] ** (12 / len(r)) - 1
    # ⚠️ cummax() 從第一個月的淨值起算，若第一個月就虧損，那段回撤會看不見。
    #    期初本金 1.0 本身就是一個峰值，所以下限要夾在 1.0。
    mdd = (cum / cum.cummax().clip(lower=1.0) - 1).min()
    ex = r if rf is None else (r - rf.reindex(r.index)).dropna()
    sr_m = ex.mean() / ex.std(ddof=1)
    out = {'n': len(r), 'CAGR': cagr, 'MDD': mdd,
           'Sharpe': sr_m * np.sqrt(12),
           'Sharpe_SE': np.sqrt((1 + 0.5 * sr_m ** 2) / len(ex)) * np.sqrt(12)}
    if weights is not None:
        # 持有月 t 的曝險由決策日 t-1 的權重決定
        exposure = weights.sum(axis=1).reindex(r.index - pd.offsets.MonthEnd(1))
        out['Exposure'] = float(exposure.mean())
        out['CAGR_per_exposure'] = cagr / out['Exposure'] if out['Exposure'] else np.nan
    return out


def calculate_performance_metrics(returns, name="Strategy", rf=None):
    """印出核心績效指標並回傳 (資金曲線, 回撤曲線)。"""
    m = performance_metrics(returns, rf)
    cum_returns = (1 + returns).cumprod()
    drawdown = cum_returns / cum_returns.cummax().clip(lower=1.0) - 1

    print(f"--- {name} ({m['n']} 個月) ---")
    print(f"年化報酬率 (CAGR): {m['CAGR']:.2%}")
    print(f"最大回撤 (Max Drawdown): {m['MDD']:.2%}")
    rf_note = "未扣 rf" if rf is None else f"扣 {_config.RF_TICKER}"
    print(f"夏普值 (Sharpe, {rf_note}): {m['Sharpe']:.2f} ± {m['Sharpe_SE']:.2f}")
    return cum_returns, drawdown


if __name__ == "__main__":
    # 1. 讀取資料
    try:
        df = pd.read_csv("etf_adj_close_clean.csv", index_col="Date", parse_dates=True)
    except FileNotFoundError:
        print("❌ 找不到 etf_adj_close_clean.csv，請先執行資料下載模組。")
        exit()

    for col in ('SPY', _config.RF_TICKER):
        if col not in df.columns:
            print(f"❌ 資料中找不到 '{col}' 欄位！請確認已將其加入下載標的池中。")
            exit()

    # 2. 執行核心邏輯
    mom_scores = calculate_momentum_features(df)
    top_n = _config.TOP_N
    port_returns, weight_matrix = backtest_momentum_strategy(
        df, mom_scores, top_n,
        min_score=_config.BASE_MIN_SCORE, fallback=_config.BASE_FALLBACK, cost_bps=_config.COST_BPS
    )

    # 取得 SPY 與 rf 的月報酬率，並嚴格對齊策略的時間軸
    monthly_returns, last_month_complete = month_end_returns(df)
    if not last_month_complete:
        print(f"⚠️ 最後一個月不完整 (資料止於 {df.index[-1].date()})，該月不納入回測。")
    spy_returns = monthly_returns['SPY'].reindex(port_returns.index)
    rf_returns = monthly_returns[_config.RF_TICKER]

    # 3. 計算並印出績效比較
    print("\n" + "="*40)
    print(f"🏆 績效報告比較 (總交易月數: {len(port_returns)} 個月, 成本 {_config.COST_BPS} bps)")
    print("="*40)
    strat_cum, strat_dd = calculate_performance_metrics(port_returns, name=f"基準動能策略 (Top {top_n})", rf=rf_returns)
    print("-" * 40)
    spy_cum, spy_dd = calculate_performance_metrics(spy_returns, name="大盤基準 (SPY)", rf=rf_returns)
    print("="*40)

    # 4. 繪製並儲存績效圖表
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    plt.plot(strat_cum.index, strat_cum.values, label=f'Strategy (Top {top_n} Momentum)', color='blue', linewidth=2)
    plt.plot(spy_cum.index, spy_cum.values, label='SPY (Benchmark)', color='gray', linestyle='--', linewidth=2)
    plt.title('Portfolio Equity Curve: Strategy vs SPY')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True, alpha=0.3)

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

    # 將兩者的月報酬率合併存檔，供 ML 模型對比
    comparison_df = pd.DataFrame({
        'Strategy_Return': port_returns,
        'SPY_Return': spy_returns
    })
    comparison_df.to_csv("baseline_monthly_returns.csv")
    print("💾 策略與大盤報酬率已合併儲存至: baseline_monthly_returns.csv")
