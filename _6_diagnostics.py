"""
診斷模組：在看資金曲線之前，先回答兩個問題——
  (1) 模型在樣本外有沒有預測力？  → 逐月橫斷面 rank-IC，依 train / val / test 分段，與傳統動能並列
  (2) 這個績效比「隨機」好多少？  → 隨機選股 null 分佈、等權 null、Sharpe 的標準誤與 bootstrap 信賴區間

另外量化幾個會影響結論的量：Ensemble 各成員的分散度、換手率與交易成本敏感度、val/test 之間的 embargo 缺口。

與 _5_strategy_backtest.py 的差異（刻意的，數字不會完全相同）：
- 以「決策日」歸段：決策日落在 test 期的月份才算 OOS，所以月份對齊比 _5 晚一個月。
- 現金月（權重全 0）計 0% 報酬，不從序列中刪除；「尚無有效訊號」的月份才排除。
- Sharpe 同時報「未扣 rf」與「扣 RF_TICKER 月報酬」兩種。
- 若最後一個日曆月不完整（資料最後一天不是該月最後交易日），該月不納入。
本模組只讀取既有輸出（CSV 與 saved_models/），不改動它們。
"""

import io
import re
import glob
import contextlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import _config
from _2_baseline_performance import (calculate_momentum_features, backtest_momentum_strategy,
                                     month_end_returns, traded_fraction, performance_metrics as perf)
from _3_ml_data_pipeline import generate_ml_features
from _5_strategy_backtest import generate_ml_scores

TARGET_HORIZON = 20   # 與 _3_ml_data_pipeline.py 的 shift(-20) 一致 (那裡尚未參數化)
MIN_CROSS_SECTION = 5 # 橫斷面至少幾檔才算一個 IC 觀測值

# 圖表用色 (categorical slot 1/2/3 + 中性色)
PALETTE = {
    'ml': '#2a78d6', 'base': '#eb6834', 'ref': '#1baf7a', 'null': '#9ec5f4',
    'ink': '#0b0b0b', 'ink2': '#52514e', 'surface': '#fcfcfb', 'grid': '#e6e5e1',
}


@contextlib.contextmanager
def quiet():
    """壓掉被呼叫模組的進度輸出，避免迴圈裡印 20 次。"""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ==========================================
# 1. 時間軸與分段
# ==========================================
def split_boundaries(X):
    """回傳 (val_start, test_start)，與 _3_ml_data_pipeline.prepare_dataloaders 的切法一致。"""
    n = len(X)
    tr = int(n * _config.TRAIN_RATIO)
    va = int(n * (_config.TRAIN_RATIO + _config.VAL_RATIO))
    return X.index[tr], X.index[va]


def segment_of(dates, val_start, test_start):
    """把一組決策日歸到 train / val / test。"""
    dates = pd.DatetimeIndex(dates)
    seg = np.where(dates < val_start, 'train', np.where(dates < test_start, 'val', 'test'))
    return pd.Series(seg, index=dates)


# ==========================================
# 2. 預測力：橫斷面 rank-IC
# ==========================================
def monthly_rank_ic(scores_m, alpha_m):
    """
    scores_m: 決策日(月底) x 標的 的分數
    alpha_m : 持有月(月底) x 標的 的實現超額報酬 (相對 SPY)
    對每個決策日 d，算 scores_m.loc[d] 與「下個月」alpha 的 Spearman 相關。
    """
    alpha_next = alpha_m.shift(-1)   # 對齊到決策日
    out = {}
    for d in scores_m.index:
        if d not in alpha_next.index:
            continue
        a = scores_m.loc[d].dropna()
        b = alpha_next.loc[d].dropna()
        common = a.index.intersection(b.index)
        if len(common) < MIN_CROSS_SECTION:
            continue
        out[d] = spearmanr(a[common], b[common]).correlation
    return pd.Series(out, dtype=float)


def ic_summary(ic, seg):
    """依分段彙整 IC：觀測數、平均、勝率、t 值。"""
    rows = []
    for name in ['train', 'val', 'test']:
        x = ic[seg.reindex(ic.index) == name]
        if len(x) == 0:
            continue
        t = x.mean() / x.std(ddof=1) * np.sqrt(len(x)) if len(x) > 1 else np.nan
        rows.append({'segment': name, 'n': len(x), 'IC_mean': x.mean(),
                     'IC_hit': (x > 0).mean(), 'IC_t': t})
    return pd.DataFrame(rows).set_index('segment')


# ==========================================
# 3. 統計不確定性
# ==========================================
def bootstrap_sharpe_ci(r, n_boot, rng, level=0.95):
    """iid bootstrap 的年化 Sharpe 信賴區間。月數很少時區間會很寬——那正是重點。"""
    x = r.dropna().values
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    s = x[idx]
    sr = s.mean(axis=1) / s.std(axis=1, ddof=1) * np.sqrt(12)
    a = (1 - level) / 2
    return np.nanpercentile(sr, 100 * a), np.nanpercentile(sr, 100 * (1 - a))


# ==========================================
# 4. Null：隨機選股
# ==========================================
def random_selection_null(R, top_n, n_sims, rng):
    """
    每個月從 K 檔裡隨機抽 top_n 檔等權持有，重複 n_sims 次。
    R: (月數, 標的數) 的月報酬矩陣。回傳每次模擬的 Sharpe / CAGR / MDD。
    """
    T, K = R.shape
    out = np.empty((n_sims, 3))
    for i in range(n_sims):
        pick = np.argsort(rng.random((T, K)), axis=1)[:, :top_n]
        rs = np.take_along_axis(R, pick, axis=1).mean(axis=1)
        cum = np.cumprod(1 + rs)
        out[i] = [rs.mean() / rs.std(ddof=1) * np.sqrt(12),
                  cum[-1] ** (12 / T) - 1,
                  (cum / np.maximum.accumulate(cum) - 1).min()]
    return pd.DataFrame(out, columns=['Sharpe', 'CAGR', 'MDD'])


# ==========================================
# 5. Ensemble 成員
# ==========================================
def discover_model_paths(pattern='saved_models/ensemble_lstm_seed_*.pth'):
    """從磁碟找出所有 ensemble 權重，依 seed 排序 (不另外硬編 seed 範圍)。"""
    seed_of = lambda p: int(re.search(r'seed_(\d+)\.pth$', p).group(1))
    return sorted(glob.glob(pattern), key=seed_of)


def per_seed_scores(df, model_paths):
    """每個 seed 各跑一次 _5 的推論；ensemble 分數 = 各 seed 的平均 (與 _5 的算法相同)。"""
    scores = {}
    for p in model_paths:
        with quiet():
            scores[p] = generate_ml_scores(df, [p], seq_length=_config.SEQ_LENGTH)
    ensemble = pd.concat(scores.values()).groupby(level=0).mean()
    return scores, ensemble


# ==========================================
# 6. 繪圖
# ==========================================
def plot_report(ic_ml, ic_base, val_start, test_start, null_df, marks, seed_sharpes, out_path):
    P = PALETTE
    fig = plt.figure(figsize=(16, 10), facecolor=P['surface'])
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.35, wspace=0.2)

    def style(ax):
        ax.set_facecolor(P['surface'])
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(P['grid'])
        ax.tick_params(colors=P['ink2'])
        ax.grid(True, color=P['grid'], lw=0.8, alpha=0.6)
        ax.set_axisbelow(True)

    # --- (a) 逐月 IC ---
    ax = fig.add_subplot(gs[0, :]); style(ax)
    x_end = max(ic_ml.index.max(), ic_base.index.max()) + pd.Timedelta(days=20)
    ax.axvspan(val_start, test_start, color=P['grid'], alpha=0.6, lw=0)
    ax.axvspan(test_start, x_end, color=P['null'], alpha=0.25, lw=0)
    ax.axhline(0, color=P['ink2'], lw=1)
    oos_ml = ic_ml[ic_ml.index >= test_start].mean()
    oos_base = ic_base[ic_base.index >= test_start].mean()
    ax.plot(ic_ml.index, ic_ml.values, color=P['ml'], lw=2, marker='o', ms=4,
            label=f'Ensemble LSTM (OOS mean IC {oos_ml:+.2f})')
    ax.plot(ic_base.index, ic_base.values, color=P['base'], lw=2, marker='o', ms=4,
            label=f'Baseline momentum (OOS mean IC {oos_base:+.2f})')
    for label, x in [('train', ic_ml.index.min()), ('val', val_start), ('test (OOS)', test_start)]:
        ax.text(x, 0.93, f' {label}', color=P['ink2'], fontsize=10, va='top')
    ax.set_ylim(-1, 1)
    ax.set_ylabel('Spearman rank-IC', color=P['ink2'])
    ax.set_title('Monthly cross-sectional rank-IC: score vs next-month excess return over SPY',
                 color=P['ink'], loc='left')
    ax.legend(loc='lower left', frameon=False)

    # --- (b) 隨機選股 null vs 策略 ---
    ax = fig.add_subplot(gs[1, 0]); style(ax)
    ax.hist(null_df['Sharpe'], bins=40, color=P['null'], edgecolor=P['surface'], lw=0.5,
            label=f'Random {_config.TOP_N}-of-K equal-weight (n={len(null_df)})')
    for (name, val), color in zip(marks.items(), [P['ml'], P['base'], P['ref']]):
        pct = (null_df['Sharpe'] < val).mean()
        ax.axvline(val, color=color, lw=2, label=f'{name}: {val:.2f} (pctile {pct:.0%})')
    ax.set_xlabel('Annualised Sharpe (OOS holding months)', color=P['ink2'])
    ax.set_ylabel('Simulations', color=P['ink2'])
    ax.set_title('Where does the OOS Sharpe sit against random selection?', color=P['ink'], loc='left')
    ax.legend(loc='upper right', frameon=False, fontsize=9)

    # --- (c) 各 seed 的 OOS Sharpe ---
    ax = fig.add_subplot(gs[1, 1]); style(ax)
    vals = np.sort(seed_sharpes.values)
    ax.scatter(np.arange(len(vals)), vals, color=P['ml'], s=36, zorder=3, label='Individual seed')
    ens_name, ens_val = list(marks.items())[0]
    base_name, base_val = list(marks.items())[1]
    ax.axhline(ens_val, color=P['ml'], lw=2, ls='--', label=f'{ens_name}: {ens_val:.2f}')
    ax.axhline(base_val, color=P['base'], lw=2, ls='--', label=f'{base_name}: {base_val:.2f}')
    ax.set_xticks(np.arange(0, len(vals), max(1, len(vals) // 10)))
    ax.set_xlabel('Ensemble member (sorted by OOS Sharpe)', color=P['ink2'])
    ax.set_ylabel('Annualised Sharpe (OOS)', color=P['ink2'])
    ax.set_title(f'Dispersion across {len(vals)} seeds: range {vals.max() - vals.min():.2f}',
                 color=P['ink'], loc='left')
    ax.legend(loc='upper left', frameon=False, fontsize=9)

    fig.subplots_adjust(top=0.94, bottom=0.07, left=0.06, right=0.98)
    fig.savefig(out_path, dpi=120, facecolor=P['surface'])
    plt.close(fig)


# ==========================================
# 7. 主流程
# ==========================================
if __name__ == "__main__":
    rng = np.random.default_rng(_config.DIAG_SEED)
    top_n = _config.TOP_N

    # --- 讀資料、切分邊界 ---
    df = pd.read_csv("etf_adj_close_clean.csv", index_col="Date", parse_dates=True)
    if 'SPY' not in df.columns or _config.RF_TICKER not in df.columns:
        print(f"❌ 資料需同時包含 'SPY' 與 '{_config.RF_TICKER}'。")
        exit()
    assets = [c for c in df.columns if c != 'SPY']

    with quiet():
        X, _ = generate_ml_features(df)
    val_start, test_start = split_boundaries(X)

    month_ret, last_month_complete = month_end_returns(df)
    alpha_m = month_ret[assets].sub(month_ret['SPY'], axis=0)
    rf_m = month_ret[_config.RF_TICKER]

    print("=" * 60)
    print("🔬 診斷報告：預測力、統計不確定性、null 比較")
    print("=" * 60)
    print(f"📅 價格資料: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)} 天)")
    print(f"✂️ 決策日分段: train < {val_start.date()} ≤ val < {test_start.date()} ≤ test")
    if not last_month_complete:
        print(f"⚠️ 最後一個月不完整 (資料止於 {df.index[-1].date()})，該月不納入任何統計。")

    # embargo 缺口：val 最後一筆 target 覆蓋到 test 起點之後多少天
    va_pos = df.index.get_loc(test_start)
    val_last_target_end = df.index[min(va_pos - 1 + TARGET_HORIZON, len(df) - 1)]
    print(f"⚠️ Embargo 缺口: val 最後一筆 target 覆蓋至 {val_last_target_end.date()}，"
          f"晚於 test 起點 {test_start.date()} → early stopping 已看到 test 前 {TARGET_HORIZON - 1} 個交易日的實現報酬。")

    # --- 分數：ML ensemble (逐 seed) 與 baseline 動能 ---
    model_paths = discover_model_paths()
    if not model_paths:
        print("❌ saved_models/ 內找不到任何 ensemble 權重，請先執行訓練。")
        exit()
    print(f"\n🧠 載入 {len(model_paths)} 個 ensemble 成員並逐一推論...")
    seed_scores, ml_daily = per_seed_scores(df, model_paths)
    ml_m = ml_daily.resample('ME').last()

    mom_daily = calculate_momentum_features(df)
    mom_m = mom_daily.drop(columns='SPY').resample('ME').last()

    # --- (1) 預測力 ---
    ic_ml = monthly_rank_ic(ml_m, alpha_m)
    ic_base = monthly_rank_ic(mom_m, alpha_m)
    seg_ic = segment_of(ic_ml.index.union(ic_base.index), val_start, test_start)

    print("\n" + "-" * 60)
    print("📐 (1) 橫斷面 rank-IC：分數 vs 下個月相對 SPY 的實現超額報酬")
    print("-" * 60)
    ic_tab = pd.concat({'Ensemble LSTM': ic_summary(ic_ml, seg_ic),
                        'Baseline momentum': ic_summary(ic_base, seg_ic)}, names=['strategy'])
    print(ic_tab.to_string(formatters={'IC_hit': '{:.0%}'.format}, float_format=lambda v: f"{v:+.3f}"))
    print("   解讀：IC 是模型「排序」能力，與後面的資金曲線無關。train 很高而 test 接近 0 = 記憶，不是預測。")

    # --- (2) 報酬序列 (現金月計 0%) ---
    with quiet():
        ml_r, ml_w = backtest_momentum_strategy(df, ml_daily, top_n, _config.ML_MIN_SCORE, _config.ML_FALLBACK, cost_bps=0)
        base_r, base_w = backtest_momentum_strategy(df, mom_daily, top_n, _config.BASE_MIN_SCORE, _config.BASE_FALLBACK, cost_bps=0)

    # 持有月的分段 = 其決策日 (前一個月底) 的分段
    seg_hold = segment_of(ml_r.index - pd.offsets.MonthEnd(1), val_start, test_start)
    seg_hold.index = ml_r.index
    oos_idx = ml_r.index[seg_hold.values == 'test']
    is_idx = ml_r.index[seg_hold.values != 'test']
    if len(oos_idx) == 0:
        print("❌ 沒有任何決策日落在 test 期，無法做 OOS 診斷。")
        exit()

    cash_months = ml_r[oos_idx][ml_r[oos_idx] == 0]
    print(f"\n💰 OOS 持有月數 {len(oos_idx)}（{oos_idx[0].strftime('%Y-%m')} ~ {oos_idx[-1].strftime('%Y-%m')}），"
          f"其中 ML 空手月 {len(cash_months)} 個（計 0%）。")

    # --- (3) 績效 + 不確定性 ---
    series = {
        'Ensemble LSTM': ml_r,
        'Baseline momentum': base_r,
        'SPY': month_ret['SPY'],
        f'Equal-weight {len(assets)} ETFs': month_ret[assets].mean(axis=1),
    }
    rows = []
    for seg_name, idx in [('test (OOS)', oos_idx), ('train+val (in-sample)', is_idx)]:
        for name, r in series.items():
            r_seg = r.reindex(idx).dropna()
            if len(r_seg) < 3:
                continue
            m = perf(r_seg)
            m_ex = perf(r_seg, rf=rf_m)
            lo, hi = bootstrap_sharpe_ci(r_seg, _config.N_BOOTSTRAP, rng)
            rows.append({'segment': seg_name, 'strategy': name, **m,
                         'Sharpe_CI95_lo': lo, 'Sharpe_CI95_hi': hi,
                         f'Sharpe_ex{_config.RF_TICKER}': m_ex['Sharpe']})
    perf_tab = pd.DataFrame(rows).set_index(['segment', 'strategy'])

    print("\n" + "-" * 60)
    print("📊 (2) 績效與統計不確定性（Sharpe_SE 為 iid 近似；CI 為 bootstrap 95%）")
    print("-" * 60)
    fmt = {'CAGR': '{:.2%}'.format, 'MDD': '{:.2%}'.format}
    print(perf_tab.to_string(formatters=fmt, float_format=lambda v: f"{v:.2f}"))
    oos_ml = perf_tab.loc[('test (OOS)', 'Ensemble LSTM')]
    oos_base = perf_tab.loc[('test (OOS)', 'Baseline momentum')]
    print(f"   解讀：OOS Sharpe 差 {oos_ml['Sharpe'] - oos_base['Sharpe']:+.2f}，"
          f"而單一策略的 SE 就有 ±{oos_ml['Sharpe_SE']:.2f}。差距落在一個標準誤內時，方向不可信。")

    # --- (4) Null：隨機選股 ---
    print("\n" + "-" * 60)
    print(f"🎲 (3) 隨機選股 null：每月隨機抽 {top_n}/{len(assets)} 檔等權，{_config.N_NULL_SIMS} 次（OOS 持有月）")
    print("-" * 60)
    R = month_ret.loc[oos_idx, assets].values
    null_df = random_selection_null(R, top_n, _config.N_NULL_SIMS, rng)
    ew_oos = perf(series[f'Equal-weight {len(assets)} ETFs'].reindex(oos_idx))
    marks = {'Ensemble LSTM': oos_ml['Sharpe'], 'Baseline momentum': oos_base['Sharpe'],
             'Equal-weight all': ew_oos['Sharpe']}
    q = null_df.quantile([0.05, 0.5, 0.95]).T
    q.columns = ['null_5%', 'null_50%', 'null_95%']
    for name, key in [('Ensemble LSTM', ('test (OOS)', 'Ensemble LSTM')),
                      ('Baseline momentum', ('test (OOS)', 'Baseline momentum'))]:
        v = perf_tab.loc[key]
        q[name] = [v['Sharpe'], v['CAGR'], v['MDD']]
        # MDD 是負數，「比 null 好」= 比較淺
        q[f'{name} pctile'] = [(null_df['Sharpe'] < v['Sharpe']).mean(),
                               (null_df['CAGR'] < v['CAGR']).mean(),
                               (null_df['MDD'] < v['MDD']).mean()]
    print(q.to_string(float_format=lambda v: f"{v:.3f}"))
    print("   解讀：pctile 是「贏過多少比例的隨機組合」。< 95% 表示與亂抽分不開。")

    # --- (5) Ensemble 分散度 ---
    print("\n" + "-" * 60)
    print(f"🧩 (4) Ensemble 各成員的 OOS 表現分散度（{len(seed_scores)} 個 seed）")
    print("-" * 60)
    seed_rows = {}
    for p, sc in seed_scores.items():
        with quiet():
            r, _ = backtest_momentum_strategy(df, sc, top_n, _config.ML_MIN_SCORE, _config.ML_FALLBACK, cost_bps=0)
        seed_rows[p] = perf(r.reindex(oos_idx).dropna())
    seed_tab = pd.DataFrame(seed_rows).T[['CAGR', 'MDD', 'Sharpe']]
    print(seed_tab.describe().loc[['min', '25%', '50%', '75%', 'max']]
          .to_string(formatters=fmt, float_format=lambda v: f"{v:.2f}"))
    print(f"   Ensemble 平均後 = Sharpe {oos_ml['Sharpe']:.2f}；成員全距 {seed_tab['Sharpe'].max() - seed_tab['Sharpe'].min():.2f}。"
          f"平均消掉的是初始化雜訊，消不掉的是資料太少。")

    # --- (6) 換手率與交易成本敏感度 ---
    print("\n" + "-" * 60)
    print("💸 (5) 換手率與交易成本敏感度（OOS；成本 = 成交金額 × bps）")
    print("-" * 60)
    ml_traded = traded_fraction(ml_w).reindex(month_ret.index).shift(1).reindex(oos_idx)
    base_traded = traded_fraction(base_w).reindex(month_ret.index).shift(1).reindex(oos_idx)
    print(f"   平均單邊月換手: ML {ml_traded.mean() / 2:.1%} | Baseline {base_traded.mean() / 2:.1%}")
    cost_rows = []
    for bps in _config.COST_GRID_BPS:
        m1 = perf(ml_r.reindex(oos_idx) - ml_traded * bps / 1e4)
        m2 = perf(base_r.reindex(oos_idx) - base_traded * bps / 1e4)
        cost_rows.append({'cost_bps': bps, 'ML Sharpe': m1['Sharpe'], 'ML CAGR': m1['CAGR'],
                          'Base Sharpe': m2['Sharpe'], 'Base CAGR': m2['CAGR']})
    cost_tab = pd.DataFrame(cost_rows).set_index('cost_bps')
    print(cost_tab.to_string(formatters={'ML CAGR': '{:.2%}'.format, 'Base CAGR': '{:.2%}'.format},
                             float_format=lambda v: f"{v:.2f}"))
    if oos_ml['Sharpe'] > oos_base['Sharpe']:
        for bps in range(0, 501):
            s1 = perf(ml_r.reindex(oos_idx) - ml_traded * bps / 1e4)['Sharpe']
            s2 = perf(base_r.reindex(oos_idx) - base_traded * bps / 1e4)['Sharpe']
            if s1 <= s2:
                print(f"   ML 與 Baseline 的 Sharpe 在成本約 {bps} bps 時打平。")
                break
        else:
            print("   成本到 500 bps 仍未打平（換手率差異不足以翻轉）。")

    # --- 存檔 ---
    summary = perf_tab.reset_index()
    summary.to_csv("diagnostics_summary.csv", index=False)
    pd.DataFrame({'IC_LSTM': ic_ml, 'IC_baseline': ic_base}).join(seg_ic.rename('segment')) \
        .to_csv("diagnostics_ic_monthly.csv")
    plot_report(ic_ml, ic_base, val_start, test_start, null_df, marks,
                seed_tab['Sharpe'], "diagnostics_report.png")

    print("\n" + "=" * 60)
    print("📝 誠實聲明（每份報告都該附）")
    print("=" * 60)
    print(f"  • OOS 只有 {len(oos_idx)} 個月、單一 regime（SPY 同期 CAGR {perf_tab.loc[('test (OOS)', 'SPY'), 'CAGR']:.1%}）。")
    print(f"  • 標的池 {len(assets)} 檔為事後自選，期間止於資料抓取日；兩者都會灌水。")
    print("  • 若 test set 曾在 V1→V2 之間被看過，它就是第二個 validation set，上面的 OOS 數字要再打折。")
    print(f"  • 上表 (2)(3)(4) 為毛報酬；_5 的正式回測扣 {_config.COST_BPS} bps，成本敏感度見 (5)。")
    print("\n💾 已儲存: diagnostics_summary.csv, diagnostics_ic_monthly.csv, diagnostics_report.png")
