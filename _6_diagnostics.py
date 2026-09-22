"""
診斷模組：在看資金曲線之前，先回答兩個問題——
  (1) 模型在樣本外有沒有預測力？  → 逐月橫斷面 rank-IC (walk-forward OOS)，與 ridge 線性基準、傳統動能並列，
                                  並以每個 fold 的模型對自己的訓練期評分，量化「樣本內 vs 樣本外」的落差
  (2) 這個績效比「隨機」好多少？  → 隨機選股 null 分佈 (同樣扣成本、扣 rf)、Sharpe 的標準誤與 bootstrap 信賴區間

另外量化：Ensemble 各成員的分散度、換手率與交易成本敏感度。
報告開頭原文印出 _config.PRE_REGISTRATION，結尾依其判準機械化給出結論。

會計口徑與 _5 相同 (_2_baseline_performance.py)：現金月計 0%、不完整的最後一個月剔除。
本模組只讀取既有輸出，不改動它們。
"""

import io
import contextlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import _config
from _2_baseline_performance import (calculate_momentum_features, backtest_momentum_strategy,
                                     month_end_returns, traded_fraction, constant_weights,
                                     performance_metrics as perf)
from _3_ml_data_pipeline import generate_ml_features
from _4b_ridge_baseline import ridge_scores
from _5_strategy_backtest import usable_folds, generate_ml_scores, fold_of_holding_month

MIN_CROSS_SECTION = 5   # 橫斷面至少幾檔才算一個 IC 觀測值

# 圖表用色 (categorical slot 1/2/3 + 中性色)
PALETTE = {
    'lstm': '#2a78d6', 'ridge': '#eb6834', 'mom': '#1baf7a', 'null': '#9ec5f4',
    'ink': '#0b0b0b', 'ink2': '#52514e', 'surface': '#fcfcfb', 'grid': '#e6e5e1',
}


@contextlib.contextmanager
def quiet():
    """壓掉被呼叫模組的進度輸出。"""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ==========================================
# 1. 預測力：橫斷面 rank-IC
# ==========================================
def monthly_rank_ic(scores_m, alpha_m):
    """
    scores_m: 決策日(月底) x 標的 的分數 (索引可為實際交易日，先對齊到日曆月底)
    alpha_m : 持有月(月底) x 標的 的實現超額報酬 (相對 SPY)
    對每個決策日 d，算分數與「下個月」alpha 的 Spearman 相關。
    """
    scores_m = scores_m.resample('ME').last()
    alpha_next = alpha_m.shift(-1)
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
    return pd.Series(out, dtype=float).dropna()


def ic_stats(ic):
    """觀測數、平均、勝率、t 值。"""
    n = len(ic)
    t = ic.mean() / ic.std(ddof=1) * np.sqrt(n) if n > 1 else np.nan
    return {'n': n, 'IC_mean': ic.mean(), 'IC_hit': (ic > 0).mean(), 'IC_t': t}


def paired_t(a, b):
    """兩個逐月 IC 序列在共同月份的差之配對 t 值。"""
    d = (a - b).dropna()
    return d.mean() / d.std(ddof=1) * np.sqrt(len(d)), len(d)


# ==========================================
# 2. 統計不確定性與 null
# ==========================================
def bootstrap_sharpe_ci(r, n_boot, rng, level=0.95):
    """iid bootstrap 的年化 Sharpe 信賴區間。"""
    x = r.dropna().values
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    s = x[idx]
    sr = s.mean(axis=1) / s.std(axis=1, ddof=1) * np.sqrt(12)
    a = (1 - level) / 2
    return np.nanpercentile(sr, 100 * a), np.nanpercentile(sr, 100 * (1 - a))


def random_selection_null(R, rf, top_n, n_sims, rng, cost_bps=0.0):
    """
    每個月從 K 檔裡隨機抽 top_n 檔等權持有，重複 n_sims 次；與策略同樣扣成本、扣 rf。
    R: (月數, 標的數) 月報酬矩陣；rf: (月數,) 無風險月報酬。回傳每次模擬的 Sharpe / CAGR / MDD。
    """
    T, K = R.shape
    out = np.empty((n_sims, 3))
    for i in range(n_sims):
        pick = np.argsort(rng.random((T, K)), axis=1)[:, :top_n]
        W = np.zeros((T, K))
        np.put_along_axis(W, pick, 1.0 / top_n, axis=1)
        traded = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, K)))).sum(axis=1)
        rs = (W * R).sum(axis=1) - traded * cost_bps / 1e4
        cum = np.cumprod(1 + rs)
        ex = rs - rf
        out[i] = [ex.mean() / ex.std(ddof=1) * np.sqrt(12),
                  cum[-1] ** (12 / T) - 1,
                  (cum / np.maximum(np.maximum.accumulate(cum), 1.0) - 1).min()]
    return pd.DataFrame(out, columns=['Sharpe', 'CAGR', 'MDD'])


# ==========================================
# 3. 繪圖
# ==========================================
def plot_report(ics, fold_starts, null_df, marks, seed_sharpes, out_path):
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

    # --- (a) 逐月 OOS IC ---
    ax = fig.add_subplot(gs[0, :]); style(ax)
    for i, fs in enumerate(fold_starts):
        ax.axvline(fs, color=P['grid'], lw=1.2)
        ax.text(fs, 0.93, f' fold {i + 1}', color=P['ink2'], fontsize=9, va='top')
    ax.axhline(0, color=P['ink2'], lw=1)
    for (name, ic), color in zip(ics.items(), [P['lstm'], P['ridge'], P['mom']]):
        ax.plot(ic.index, ic.values, color=color, lw=2, marker='o', ms=4,
                label=f'{name} (mean IC {ic.mean():+.2f}, hit {(ic > 0).mean():.0%})')
    ax.set_ylim(-1, 1)
    ax.set_ylabel('Spearman rank-IC', color=P['ink2'])
    ax.set_title('Walk-forward OOS monthly rank-IC: score vs next-month excess return over SPY', color=P['ink'], loc='left')
    ax.legend(loc='lower left', frameon=False, fontsize=9)

    # --- (b) 隨機選股 null vs 策略 ---
    ax = fig.add_subplot(gs[1, 0]); style(ax)
    ax.hist(null_df['Sharpe'], bins=40, color=P['null'], edgecolor=P['surface'], lw=0.5,
            label=f'Random {_config.TOP_N}-of-K equal-weight (n={len(null_df)})')
    for (name, val), color in zip(marks.items(), [P['lstm'], P['ridge'], P['mom']]):
        pct = (null_df['Sharpe'] < val).mean()
        ax.axvline(val, color=color, lw=2, label=f'{name}: {val:.2f} (pctile {pct:.0%})')
    ax.set_xlabel(f'Annualised Sharpe, OOS, net of {_config.COST_BPS} bps, ex-{_config.RF_TICKER}', color=P['ink2'])
    ax.set_ylabel('Simulations', color=P['ink2'])
    ax.set_title('Where does the OOS Sharpe sit against random selection?', color=P['ink'], loc='left')
    ax.legend(loc='upper right', frameon=False, fontsize=9)

    # --- (c) 各 seed 的 OOS Sharpe ---
    ax = fig.add_subplot(gs[1, 1]); style(ax)
    vals = np.sort(seed_sharpes.values)
    ax.scatter(np.arange(len(vals)), vals, color=P['lstm'], s=36, zorder=3, label='Individual seed')
    items = list(marks.items())
    ax.axhline(items[0][1], color=P['lstm'], lw=2, ls='--', label=f'{items[0][0]}: {items[0][1]:.2f}')
    ax.axhline(items[2][1], color=P['mom'], lw=2, ls='--', label=f'{items[2][0]}: {items[2][1]:.2f}')
    ax.set_xticks(np.arange(0, len(vals), max(1, len(vals) // 10)))
    ax.set_xlabel('Ensemble member (sorted by OOS Sharpe)', color=P['ink2'])
    ax.set_ylabel('Annualised Sharpe (OOS)', color=P['ink2'])
    ax.set_title(f'Dispersion across {len(vals)} seeds: range {vals.max() - vals.min():.2f}', color=P['ink'], loc='left')
    ax.legend(loc='center right', frameon=False, fontsize=9)

    fig.subplots_adjust(top=0.94, bottom=0.07, left=0.06, right=0.98)
    fig.savefig(out_path, dpi=120, facecolor=P['surface'])
    plt.close(fig)


# ==========================================
# 4. 主流程
# ==========================================
if __name__ == "__main__":
    rng = np.random.default_rng(_config.DIAG_SEED)
    top_n = _config.TOP_N

    print("=" * 60)
    print("📜 Pre-registration (原文，來自 _config.PRE_REGISTRATION)")
    print("=" * 60)
    print(_config.PRE_REGISTRATION)

    # --- 讀資料、fold ---
    df = pd.read_csv("etf_adj_close_clean.csv", index_col="Date", parse_dates=True)
    for col in ('SPY', _config.RF_TICKER):
        if col not in df.columns:
            print(f"❌ 資料需包含 '{col}'。")
            exit()
    assets = [c for c in df.columns if c != 'SPY']
    with quiet():
        X, Y = generate_ml_features(df)
    folds = usable_folds(X, Y)
    fold_starts = [f['test_start'] for _, f in folds]

    month_ret, last_month_complete = month_end_returns(df)
    alpha_m = month_ret[assets].sub(month_ret['SPY'], axis=0)
    rf_m = month_ret[_config.RF_TICKER]

    print("=" * 60)
    print("🔬 診斷報告：預測力、統計不確定性、null 比較")
    print("=" * 60)
    print(f"📅 價格資料: {df.index[0].date()} ~ {df.index[-1].date()} ({len(df)} 天)")
    print(f"✂️ Walk-forward folds: {', '.join(n for n, _ in folds)}")
    if not last_month_complete:
        print(f"⚠️ 最後一個月不完整 (資料止於 {df.index[-1].date()})，該月不納入任何統計。")

    # --- 分數 ---
    print(f"\n🧠 推論：{len(folds)} 個 fold × {_config.N_MODELS} 個成員 (OOS)，再以各 fold 模型評分自己的訓練期 (in-sample)...")
    with quiet():
        ml_oos, members = generate_ml_scores(X, Y, folds, assets, where='test')
        ridge_sc, _ = ridge_scores(X, Y, folds, assets)
        mom_sc = calculate_momentum_features(df)
        ml_is_by_fold = {name: generate_ml_scores(X, Y, [(name, fold)], assets, where='train')[0] for name, fold in folds}

    # --- (1) 預測力 ---
    ic = {'LSTM ensemble': monthly_rank_ic(ml_oos, alpha_m),
          'Ridge (linear)': monthly_rank_ic(ridge_sc, alpha_m),
          'Momentum': monthly_rank_ic(mom_sc.drop(columns='SPY'), alpha_m)}
    oos_months = ic['LSTM ensemble'].index
    ic['Momentum'] = ic['Momentum'].reindex(oos_months).dropna()

    print("\n" + "-" * 60)
    print("📐 (1) 橫斷面 rank-IC：分數 vs 下個月相對 SPY 的實現超額報酬 (walk-forward OOS)")
    print("-" * 60)
    ic_tab = pd.DataFrame({k: ic_stats(v) for k, v in ic.items()}).T
    print(ic_tab.to_string(formatters={'IC_hit': '{:.0%}'.format, 'n': '{:.0f}'.format}, float_format=lambda v: f"{v:+.3f}"))
    t_lr, n_lr = paired_t(ic['LSTM ensemble'], ic['Ridge (linear)'])
    t_lm, n_lm = paired_t(ic['LSTM ensemble'], ic['Momentum'])
    print(f"   配對 t (LSTM − Ridge) = {t_lr:+.2f} (n={n_lr}) | 配對 t (LSTM − Momentum) = {t_lm:+.2f} (n={n_lm})")

    # 樣本內 vs 樣本外 (逐 fold)
    fold_label = fold_of_holding_month(oos_months + pd.offsets.MonthEnd(1), folds)
    fold_label.index = oos_months
    rows = []
    for name, fold in folds:
        ic_is = monthly_rank_ic(ml_is_by_fold[name], alpha_m)
        ic_oos = ic['LSTM ensemble'][fold_label == name]
        rows.append({'fold': name, 'n_train_months': len(ic_is), 'IC_in_sample': ic_is.mean(),
                     'n_oos_months': len(ic_oos), 'IC_oos': ic_oos.mean(), 'IC_oos_hit': (ic_oos > 0).mean()})
    fold_ic_tab = pd.DataFrame(rows).set_index('fold')
    print("\n   逐 fold：各 fold 的 ensemble 對自己訓練期 (in-sample) vs 對 test 期 (OOS) 的平均 IC")
    print(fold_ic_tab.to_string(formatters={'IC_oos_hit': '{:.0%}'.format, 'n_train_months': '{:.0f}'.format,
                                            'n_oos_months': '{:.0f}'.format}, float_format=lambda v: f"{v:+.3f}"))
    print("   解讀：IC 是「排序」能力。in-sample 很高而 OOS 接近 0 = 記憶，不是預測。")

    # --- (2) 報酬與績效 ---
    with quiet():
        ml_g, ml_w = backtest_momentum_strategy(df, ml_oos, top_n, _config.ML_MIN_SCORE, _config.ML_FALLBACK, cost_bps=0)
        ridge_g, ridge_w = backtest_momentum_strategy(df, ridge_sc, top_n, _config.ML_MIN_SCORE, _config.ML_FALLBACK, cost_bps=0)
        base_g, base_w = backtest_momentum_strategy(df, mom_sc, top_n, _config.BASE_MIN_SCORE, _config.BASE_FALLBACK, cost_bps=0)
    oos_start = folds[0][1]['test_start']
    oos_idx = ml_g.index[(ml_g.index - pd.offsets.MonthEnd(1)) >= oos_start]
    oos_idx = oos_idx.intersection(ridge_g.index).intersection(base_g.index)

    def net(gross, weights):
        traded = traded_fraction(weights, month_ret).reindex(month_ret.index).shift(1)
        return (gross - traded * _config.COST_BPS / 1e4).reindex(oos_idx)

    # Benchmark 也走同一套會計：等權每月再平衡有真實換手 (漂移再平衡)，SPY buy & hold 幾乎沒有。
    # 若 benchmark 報毛報酬而策略報淨報酬，比較就偏向 benchmark。
    decision_dates = base_w.dropna(how='all').index
    ew_w = constant_weights(assets, decision_dates)
    spy_w = constant_weights(['SPY'], decision_dates, weight=1.0)
    with quiet():
        ew_g = (ew_w.reindex(month_ret.index).shift(1) * month_ret[assets]).sum(axis=1, min_count=1).dropna()
        spy_g = month_ret['SPY'].reindex(ew_g.index)

    series = {
        'LSTM ensemble': net(ml_g, ml_w),
        'Ridge (linear)': net(ridge_g, ridge_w),
        'Momentum': net(base_g, base_w),
        'SPY': net(spy_g, spy_w),
        f'Equal-weight {len(assets)} ETFs': net(ew_g, ew_w),
    }
    weights_of = {'LSTM ensemble': ml_w, 'Ridge (linear)': ridge_w, 'Momentum': base_w,
                  'SPY': spy_w, f'Equal-weight {len(assets)} ETFs': ew_w}
    cash_months = int((ml_g.reindex(oos_idx) == 0).sum())
    print(f"\n💰 OOS 持有月數 {len(oos_idx)} ({oos_idx[0].strftime('%Y-%m')} ~ {oos_idx[-1].strftime('%Y-%m')})，"
          f"LSTM 空手月 {cash_months} 個。")

    rows = []
    for name, r in series.items():
        m = perf(r, rf=rf_m, weights=weights_of[name])
        lo, hi = bootstrap_sharpe_ci(r - rf_m.reindex(r.index), _config.N_BOOTSTRAP, rng)
        rows.append({'strategy': name, **m, 'Sharpe_CI95_lo': lo, 'Sharpe_CI95_hi': hi, 'Sharpe_raw': perf(r)['Sharpe']})
    perf_tab = pd.DataFrame(rows).set_index('strategy')
    print("\n" + "-" * 60)
    print(f"📊 (2) 績效與統計不確定性 (淨 {_config.COST_BPS} bps；Sharpe 扣 {_config.RF_TICKER}，SE 為 iid 近似，CI 為 bootstrap 95%)")
    print("-" * 60)
    print(perf_tab.to_string(formatters={'CAGR': '{:.2%}'.format, 'MDD': '{:.2%}'.format, 'n': '{:.0f}'.format},
                             float_format=lambda v: f"{v:.2f}"))
    lstm, mom = perf_tab.loc['LSTM ensemble'], perf_tab.loc['Momentum']
    print(f"   解讀：OOS Sharpe 差 (LSTM − Momentum) = {lstm['Sharpe'] - mom['Sharpe']:+.2f}，LSTM 的 SE = ±{lstm['Sharpe_SE']:.2f}。")

    # --- (3) Null ---
    print("\n" + "-" * 60)
    print(f"🎲 (3) 隨機選股 null：每月隨機抽 {top_n}/{len(assets)} 檔等權，{_config.N_NULL_SIMS} 次，同樣扣成本與 rf")
    print("-" * 60)
    null_df = random_selection_null(month_ret.loc[oos_idx, assets].values, rf_m.reindex(oos_idx).values,
                                    top_n, _config.N_NULL_SIMS, rng, cost_bps=_config.COST_BPS)
    marks = {k: perf_tab.loc[k, 'Sharpe'] for k in ['LSTM ensemble', 'Ridge (linear)', 'Momentum']}
    q = null_df.quantile([0.05, 0.5, 0.95]).T
    q.columns = ['null_5%', 'null_50%', 'null_95%']
    for name in marks:
        v = perf_tab.loc[name]
        q[name] = [v['Sharpe'], v['CAGR'], v['MDD']]
        q[f'{name} pctile'] = [(null_df['Sharpe'] < v['Sharpe']).mean(),
                               (null_df['CAGR'] < v['CAGR']).mean(),
                               (null_df['MDD'] < v['MDD']).mean()]   # MDD 為負，「較好」= 較淺
    print(q.to_string(float_format=lambda v: f"{v:.3f}"))
    ew = perf_tab.loc[f'Equal-weight {len(assets)} ETFs']
    print(f"   等權 {len(assets)} 檔: Sharpe {ew['Sharpe']:.2f} (null 百分位 {(null_df['Sharpe'] < ew['Sharpe']).mean():.0%})")
    print("   解讀：pctile 是「贏過多少比例的隨機組合」。< 95% 表示與亂抽分不開。")

    # --- (4) Ensemble 分散度 ---
    print("\n" + "-" * 60)
    print(f"🧩 (4) Ensemble 各成員的 OOS 表現分散度 ({len(members)} 個 seed，淨成本、扣 rf)")
    print("-" * 60)
    seed_rows = {}
    for i, sc in members.items():
        with quiet():
            g, w = backtest_momentum_strategy(df, sc, top_n, _config.ML_MIN_SCORE, _config.ML_FALLBACK, cost_bps=0)
        seed_rows[i] = perf(net(g, w), rf=rf_m)
    seed_tab = pd.DataFrame(seed_rows).T[['CAGR', 'MDD', 'Sharpe']]
    print(seed_tab.describe().loc[['min', '25%', '50%', '75%', 'max']]
          .to_string(formatters={'CAGR': '{:.2%}'.format, 'MDD': '{:.2%}'.format}, float_format=lambda v: f"{v:.2f}"))
    print(f"   Ensemble 平均後 = Sharpe {lstm['Sharpe']:.2f}；成員全距 {seed_tab['Sharpe'].max() - seed_tab['Sharpe'].min():.2f}。")

    # --- (5) 換手率與成本敏感度 ---
    print("\n" + "-" * 60)
    print("💸 (5) 換手率與交易成本敏感度 (OOS；成本 = 成交金額 × bps；Sharpe 扣 rf)")
    print("-" * 60)
    traded = {name: traded_fraction(w, month_ret).reindex(month_ret.index).shift(1).reindex(oos_idx)
              for name, w in [('LSTM ensemble', ml_w), ('Ridge (linear)', ridge_w), ('Momentum', base_w),
                              (f'Equal-weight {len(assets)} ETFs', ew_w)]}
    gross = {'LSTM ensemble': ml_g.reindex(oos_idx), 'Ridge (linear)': ridge_g.reindex(oos_idx),
             'Momentum': base_g.reindex(oos_idx), f'Equal-weight {len(assets)} ETFs': ew_g.reindex(oos_idx)}
    print("   平均單邊月換手: " + " | ".join(f"{k} {v.mean() / 2:.1%}" for k, v in traded.items()))
    cost_rows = []
    for bps in _config.COST_GRID_BPS:
        row = {'cost_bps': bps}
        for name in gross:
            row[f'{name} Sharpe'] = perf(gross[name] - traded[name] * bps / 1e4, rf=rf_m)['Sharpe']
        cost_rows.append(row)
    print(pd.DataFrame(cost_rows).set_index('cost_bps').to_string(float_format=lambda v: f"{v:.2f}"))
    s_l0 = perf(gross['LSTM ensemble'], rf=rf_m)['Sharpe']; s_m0 = perf(gross['Momentum'], rf=rf_m)['Sharpe']
    if s_l0 > s_m0:
        for bps in range(0, 501):
            s1 = perf(gross['LSTM ensemble'] - traded['LSTM ensemble'] * bps / 1e4, rf=rf_m)['Sharpe']
            s2 = perf(gross['Momentum'] - traded['Momentum'] * bps / 1e4, rf=rf_m)['Sharpe']
            if s1 <= s2:
                print(f"   LSTM 與 Momentum 的 Sharpe 在成本約 {bps} bps 時打平。")
                break
        else:
            print("   成本到 500 bps 仍未打平。")
    else:
        print("   LSTM 在零成本時就已不高於 Momentum。")

    # --- 存檔 ---
    perf_tab.reset_index().to_csv("diagnostics_summary.csv", index=False)
    pd.DataFrame(ic).join(fold_label.rename('fold')).to_csv("diagnostics_ic_monthly.csv")
    fold_ic_tab.to_csv("diagnostics_ic_by_fold.csv")
    plot_report(ic, fold_starts, null_df, marks, seed_tab['Sharpe'], "diagnostics_report.png")

    # --- 依 pre-registration 機械化判定 ---
    a = ic_tab.loc['LSTM ensemble']
    A = (a['IC_mean'] > 0) and (a['IC_t'] > 2)
    B = t_lr > 2
    lstm_pct = (null_df['Sharpe'] < lstm['Sharpe']).mean()
    C1 = (lstm['Sharpe'] - mom['Sharpe']) >= lstm['Sharpe_SE']
    C2 = lstm_pct >= 0.95
    fold_signs = fold_ic_tab['IC_oos'] > 0
    print("\n" + "=" * 60)
    print("⚖️ 依 pre-registration 判定 (由程式計算，不得事後放寬)")
    print("=" * 60)
    print(f"  判準 A 預測力      : IC 均值 {a['IC_mean']:+.3f}, t = {a['IC_t']:+.2f} → "
          + ("成立：LSTM 有樣本外排序能力" if A else "不成立：無證據顯示有排序能力"))
    print(f"  判準 B 相對 ridge   : 配對 t = {t_lr:+.2f} → " + ("成立：LSTM 優於線性" if B else "不成立：LSTM 不具存在理由"))
    print(f"  判準 C 策略層       : ΔSharpe {lstm['Sharpe'] - mom['Sharpe']:+.2f} vs SE {lstm['Sharpe_SE']:.2f} ({'≥' if C1 else '<'}), "
          f"null 百分位 {lstm_pct:.0%} ({'≥' if C2 else '<'} 95%) → " + ("勝出" if (C1 and C2) else "與 baseline / 隨機分不開"))
    print(f"  判準 D 一致性       : 逐 fold OOS IC 為正的 fold {int(fold_signs.sum())}/{len(fold_signs)}"
          + ("" if fold_signs.all() or (~fold_signs).all() else "；fold 間符號不一致，見上表，匯總結論不得省略此點"))

    print("\n" + "=" * 60)
    print("📝 誠實聲明")
    print("=" * 60)
    print(f"  • OOS {len(oos_idx)} 個月 ({oos_idx[0].year}~{oos_idx[-1].year})，第一個 fold 的訓練樣本僅 {len(folds[0][1]['train_pos'])} 筆。")
    print(f"  • 標的池 {len(assets)} 檔為事後自選，期間止於 {_config.END_DATE}；兩者都會灌水。")
    print("  • 判準與超參數在本輪開跑前寫定於 _config.PRE_REGISTRATION；若曾修改，git log 會留下紀錄。")
    print("\n💾 已儲存: diagnostics_summary.csv, diagnostics_ic_monthly.csv, diagnostics_ic_by_fold.csv, diagnostics_report.png")
