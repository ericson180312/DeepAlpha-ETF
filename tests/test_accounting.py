"""
回測會計的單元測試。

這些函式錯了不會拋例外，只會給出一個看起來合理的數字——所以每個斷言的期望值都是手算的，
不是從實作跑出來再抄回去。用小型合成資料，不碰真實 CSV。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _2_baseline_performance import (month_end_returns, build_weights, drifted_weights,
                                     traded_fraction, constant_weights, backtest_momentum_strategy,
                                     performance_metrics, calculate_momentum_features)


# ==========================================
# 輔助：合成日價格
# ==========================================
def daily_prices(monthly_factors, assets, start='2020-01-31'):
    """
    每個月底一個價格，月報酬由 monthly_factors 給定 (第一個月為基準 1.0)。
    monthly_factors: {asset: [r1, r2, ...]}，r 為該月報酬。
    """
    n = len(next(iter(monthly_factors.values())))
    idx = pd.date_range(start, periods=n + 1, freq='ME')
    data = {}
    for a in assets:
        rets = monthly_factors[a]
        px = [100.0]
        for r in rets:
            px.append(px[-1] * (1 + r))
        data[a] = px
    return pd.DataFrame(data, index=idx)


# ==========================================
# month_end_returns
# ==========================================
def test_month_end_returns_drops_incomplete_final_month():
    idx = pd.to_datetime(['2024-01-31', '2024-02-29', '2024-03-15'])   # 3/15 不是月底
    df = pd.DataFrame({'A': [100.0, 110.0, 120.0]}, index=idx)
    ret, complete = month_end_returns(df)
    assert complete is False
    assert list(ret.index.strftime('%Y-%m')) == ['2024-01', '2024-02']
    assert ret['A'].iloc[-1] == pytest.approx(0.10)     # 110/100 - 1，3 月完全不參與


def test_month_end_returns_keeps_complete_final_month():
    idx = pd.to_datetime(['2024-01-31', '2024-02-29'])
    df = pd.DataFrame({'A': [100.0, 110.0]}, index=idx)
    ret, complete = month_end_returns(df)
    assert complete is True
    assert len(ret) == 2


# ==========================================
# build_weights
# ==========================================
def _scores(rows, columns, dates=None):
    dates = dates or pd.date_range('2024-01-31', periods=len(rows), freq='ME')
    return pd.DataFrame(rows, index=dates, columns=columns)


def test_build_weights_equal_slots():
    w = build_weights(_scores([[0.3, 0.2, 0.1, 0.05]], list('ABCD')), top_n=2)
    assert w.loc[w.index[0], 'A'] == pytest.approx(0.5)
    assert w.loc[w.index[0], 'B'] == pytest.approx(0.5)
    assert w.loc[w.index[0], ['C', 'D']].sum() == 0.0


def test_build_weights_negative_slot_goes_to_cash_when_no_fallback():
    # top 3 中有一檔 <= 0：該槽位留現金，總曝險 2/3
    w = build_weights(_scores([[0.3, 0.2, -0.1, -0.5]], list('ABCD')), top_n=3, fallback=None)
    row = w.iloc[0]
    assert row['A'] == pytest.approx(1 / 3)
    assert row['B'] == pytest.approx(1 / 3)
    assert row['C'] == 0.0
    assert row.sum() == pytest.approx(2 / 3)


def test_build_weights_negative_slot_goes_to_fallback():
    # 同上，但 fallback='SPY'：空出來的 1/3 配置到 SPY，總曝險回到 1.0
    scores = _scores([[0.3, 0.2, -0.1, -0.5]], list('ABCD'))
    scores['SPY'] = 0.0
    w = build_weights(scores, top_n=3, fallback='SPY')
    row = w.iloc[0]
    assert row['SPY'] == pytest.approx(1 / 3)
    assert row.sum() == pytest.approx(1.0)


def test_build_weights_spy_never_ranked():
    # SPY 分數最高，但它只能當 fallback，不得被選入
    scores = _scores([[0.1, 0.2, 0.05, 0.9]], ['A', 'B', 'C', 'SPY'])
    w = build_weights(scores, top_n=2, fallback=None)
    assert 'SPY' not in w.columns or w.iloc[0].get('SPY', 0.0) == 0.0
    assert w.iloc[0]['B'] == pytest.approx(0.5)


def test_build_weights_all_negative_is_all_cash_not_nan():
    """關鍵區分：有訊號但全負 = 整列 0 (空手)，不是 NaN (無訊號)。"""
    w = build_weights(_scores([[-0.1, -0.2, -0.3]], list('ABC')), top_n=2, fallback=None)
    assert w.iloc[0].notna().all()
    assert w.iloc[0].sum() == 0.0


def test_build_weights_insufficient_scores_is_nan():
    w = build_weights(_scores([[0.1, np.nan, np.nan]], list('ABC')), top_n=2)
    assert w.iloc[0].isna().all()


# ==========================================
# 現金月與前瞻
# ==========================================
def test_cash_month_counts_as_zero_and_is_not_dropped():
    """
    舊版用 `returns != 0` 過濾，會把空手月整個刪掉，等於把避險成功的月份從紀錄中抹除。
    """
    df = daily_prices({'A': [0.10, -0.20, 0.10], 'SPY': [0.05, -0.10, 0.05]}, ['A', 'SPY'])
    # 第 2 個決策日分數為負 -> 第 3 個月空手
    scores = pd.DataFrame({'A': [0.5, 0.5, -0.5, 0.5]}, index=df.index)
    ret, w = backtest_momentum_strategy(df, scores, top_n=1, fallback=None)
    assert (w.iloc[2] == 0.0).all()                      # 決策日 3 空手
    assert ret.index[-1] == df.index[-1]                 # 對應的持有月仍在序列裡
    assert ret.iloc[-1] == pytest.approx(0.0)            # 而且計 0%，不是被刪掉


def test_weights_apply_to_next_month_no_lookahead():
    """決策日的權重必須乘下個月的報酬。若誤用同月報酬，這個測試會抓到。"""
    df = daily_prices({'A': [0.50, -0.30], 'SPY': [0.0, 0.0]}, ['A', 'SPY'])
    scores = pd.DataFrame({'A': [1.0, 1.0, 1.0]}, index=df.index)
    ret, _ = backtest_momentum_strategy(df, scores, top_n=1)
    # 只有第 2 個月有「前一個決策日」，報酬應為 -0.30 (第 2 個月的報酬)，而非 +0.50
    assert ret.iloc[-1] == pytest.approx(-0.30)


# ==========================================
# 換手率
# ==========================================
def test_drifted_weights_matches_hand_calculation():
    prev = pd.Series({'A': 0.5, 'B': 0.5})
    r = pd.Series({'A': 1.0, 'B': 0.0})          # A 翻倍、B 不動
    d = drifted_weights(prev, r)
    # 市值 1.0 : 0.5 -> 權重 2/3 : 1/3
    assert d['A'] == pytest.approx(2 / 3)
    assert d['B'] == pytest.approx(1 / 3)


def test_drifted_weights_accounts_for_cash():
    prev = pd.Series({'A': 0.5})                 # 另外 50% 是現金
    r = pd.Series({'A': 1.0})                    # A 翻倍 -> 1.0 vs 現金 0.5
    d = drifted_weights(prev, r)
    assert d['A'] == pytest.approx(1.0 / 1.5)


def test_traded_fraction_first_month_is_full_build():
    w = constant_weights(['A', 'B'], pd.date_range('2024-01-31', periods=2, freq='ME'))
    ret = pd.DataFrame({'A': [0.0, 0.0], 'B': [0.0, 0.0]}, index=w.index)
    assert traded_fraction(w, ret).iloc[0] == pytest.approx(1.0)


def test_traded_fraction_charges_rebalancing_of_constant_weights():
    """
    等權組合的目標權重每月相同。直接 diff 會得到 0 換手，但實際上每月都要再平衡。
    這正是 benchmark 被低估成本的來源。
    """
    idx = pd.date_range('2024-01-31', periods=2, freq='ME')
    w = constant_weights(['A', 'B'], idx)
    ret = pd.DataFrame({'A': [0.0, 1.0], 'B': [0.0, 0.0]}, index=idx)   # 第 2 個月 A 翻倍
    traded = traded_fraction(w, ret)
    # 漂移後 2/3 : 1/3，要賣 A 1/6、買 B 1/6 -> Σ|Δ| = 1/3
    assert traded.iloc[1] == pytest.approx(1 / 3)


def test_traded_fraction_zero_when_no_drift_and_no_change():
    idx = pd.date_range('2024-01-31', periods=2, freq='ME')
    w = constant_weights(['A', 'B'], idx)
    ret = pd.DataFrame({'A': [0.0, 0.1], 'B': [0.0, 0.1]}, index=idx)   # 兩檔同漲，比例不變
    assert traded_fraction(w, ret).iloc[1] == pytest.approx(0.0)


def test_buy_and_hold_single_asset_has_no_turnover_after_build():
    idx = pd.date_range('2024-01-31', periods=3, freq='ME')
    w = constant_weights(['SPY'], idx, weight=1.0)
    ret = pd.DataFrame({'SPY': [0.0, 0.2, -0.1]}, index=idx)
    traded = traded_fraction(w, ret)
    assert traded.iloc[0] == pytest.approx(1.0)
    assert traded.iloc[1:].abs().max() == pytest.approx(0.0)


def test_cost_reduces_return_by_traded_times_bps():
    df = daily_prices({'A': [0.0, 0.0], 'B': [0.0, 0.0], 'SPY': [0.0, 0.0]}, ['A', 'B', 'SPY'])
    scores = pd.DataFrame({'A': [1.0, 0.0, 0.0], 'B': [0.0, 1.0, 1.0]}, index=df.index)
    gross, w = backtest_momentum_strategy(df, scores, top_n=1, cost_bps=0)
    net, _ = backtest_momentum_strategy(df, scores, top_n=1, cost_bps=100)     # 100 bps = 1%
    traded = traded_fraction(w, month_end_returns(df)[0]).reindex(
        month_end_returns(df)[0].index).shift(1).reindex(net.index)
    assert (gross - net).values == pytest.approx((traded * 0.01).values)


# ==========================================
# 績效指標
# ==========================================
def test_cagr_and_mdd_hand_calculation():
    r = pd.Series([0.10, -0.20, 0.10] + [0.0] * 9, index=pd.date_range('2024-01-31', periods=12, freq='ME'))
    m = performance_metrics(r)
    total = float((1 + r).prod())
    assert m['CAGR'] == pytest.approx(total - 1)          # 恰好 12 個月 -> CAGR = 總報酬
    # 淨值 1.10 -> 0.88 -> 0.968 後持平；最深回撤 = 0.88/1.10 - 1
    assert m['MDD'] == pytest.approx(-0.20)
    assert m['n'] == 12


def test_mdd_counts_drawdown_from_initial_capital():
    """
    期初本金本身是一個峰值。若只用 cummax() 起算，第一個月就虧損的那段回撤會完全看不見
    （報成 0），而樣本外序列剛好常常以虧損月開頭。
    """
    r = pd.Series([-0.10, 0.05], index=pd.date_range('2024-01-31', periods=2, freq='ME'))
    assert performance_metrics(r)['MDD'] == pytest.approx(-0.10)


def test_sharpe_uses_excess_return_over_rf():
    idx = pd.date_range('2024-01-31', periods=12, freq='ME')
    r = pd.Series(0.01, index=idx)
    r.iloc[0] = 0.02                                       # 製造非零變異
    rf = pd.Series(0.004, index=idx)
    rf.iloc[0] = 0.003
    raw = performance_metrics(r)['Sharpe']
    ex = performance_metrics(r, rf=rf)['Sharpe']
    assert ex < raw
    manual = (r - rf).mean() / (r - rf).std(ddof=1) * np.sqrt(12)
    assert ex == pytest.approx(manual)


def test_exposure_column_reflects_cash_holdings():
    idx = pd.date_range('2024-01-31', periods=3, freq='ME')
    w = pd.DataFrame({'A': [0.5, 0.5, 0.5]}, index=idx)     # 永遠只投一半
    r = pd.Series([0.01, 0.03], index=idx[1:])
    m = performance_metrics(r, weights=w)
    assert m['Exposure'] == pytest.approx(0.5)
    assert m['CAGR_per_exposure'] == pytest.approx(m['CAGR'] / 0.5)


# ==========================================
# 動能特徵
# ==========================================
def test_momentum_score_is_mean_of_window_returns():
    idx = pd.date_range('2024-01-01', periods=10, freq='D')
    df = pd.DataFrame({'A': np.arange(100.0, 110.0)}, index=idx)
    score = calculate_momentum_features(df, windows=[2, 4], verbose=False)
    expected = (df['A'].pct_change(2) + df['A'].pct_change(4)) / 2
    assert score['A'].dropna().values == pytest.approx(expected.dropna().values)


def test_momentum_windows_come_from_config_by_default():
    import _config
    idx = pd.date_range('2020-01-01', periods=400, freq='D')
    df = pd.DataFrame({'A': np.linspace(100, 200, 400)}, index=idx)
    score = calculate_momentum_features(df, verbose=False)
    expected = sum(df['A'].pct_change(w) for w in _config.MOM_WINDOWS) / len(_config.MOM_WINDOWS)
    assert score['A'].dropna().values == pytest.approx(expected.dropna().values)
