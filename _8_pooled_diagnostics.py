"""
Pooled 橫斷面 GBDT 的診斷與判定。

對手是**傳統動能**，不是已被否決的 LSTM（判準 C）。報告開頭原文印出
_config.PRE_REGISTRATION_POOLED，結尾依其判準 A~F 機械化判定。
會計與樣本外月份完全沿用 _5 / _6，確保可比。
"""

import io
import contextlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _config
from _2_baseline_performance import (calculate_momentum_features, backtest_momentum_strategy,
                                     month_end_returns, traded_fraction, constant_weights,
                                     performance_metrics as perf)
from _3_ml_data_pipeline import generate_ml_features
from _4c_pooled_gbdt import pooled_scores
from _5_strategy_backtest import usable_folds, generate_ml_scores, fold_of_holding_month
from _6_diagnostics import (monthly_rank_ic, ic_stats, paired_t,
                            bootstrap_sharpe_ci, random_selection_null)

PALETTE = {'gbdt': '#2a78d6', 'pridge': '#eb6834', 'mom': '#1baf7a', 'lstm': '#eda100',
           'null': '#9ec5f4', 'ink': '#0b0b0b', 'ink2': '#52514e', 'surface': '#fcfcfb', 'grid': '#e6e5e1'}

# 本輪競爭的上限（開跑前量定，寫在 pre-registration 裡）
SELECTION_CEILING_SHARPE = 0.19


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def plot_report(ics, fold_starts, null_df, marks, out_path):
    P = PALETTE
    fig = plt.figure(figsize=(16, 9), facecolor=P['surface'])
    gs = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.35)

    def style(ax):
        ax.set_facecolor(P['surface'])
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(P['grid'])
        ax.tick_params(colors=P['ink2'])
        ax.grid(True, color=P['grid'], lw=0.8, alpha=0.6)
        ax.set_axisbelow(True)

    ax = fig.add_subplot(gs[0]); style(ax)
    for i, fs in enumerate(fold_starts):
        ax.axvline(fs, color=P['grid'], lw=1.2)
        ax.text(fs, 0.93, f' fold {i + 1}', color=P['ink2'], fontsize=9, va='top')
    ax.axhline(0, color=P['ink2'], lw=1)
    for (name, ic), color in zip(ics.items(), [P['gbdt'], P['pridge'], P['mom']]):
        ax.plot(ic.index, ic.values, color=color, lw=2, marker='o', ms=4,
                label=f'{name} (mean IC {ic.mean():+.2f}, hit {(ic > 0).mean():.0%})')
    ax.set_ylim(-1, 1)
    ax.set_ylabel('Spearman rank-IC', color=P['ink2'])
    ax.set_title('Pooled cross-sectional models vs plain momentum: walk-forward OOS rank-IC',
                 color=P['ink'], loc='left')
    ax.legend(loc='lower left', frameon=False, fontsize=9)

    ax = fig.add_subplot(gs[1]); style(ax)
    ax.hist(null_df['Sharpe'], bins=40, color=P['null'], edgecolor=P['surface'], lw=0.5,
            label=f'Random {_config.TOP_N}-of-K equal-weight (n={len(null_df)})')
    for (name, val), color in zip(marks.items(), [P['gbdt'], P['pridge'], P['mom']]):
        pct = (null_df['Sharpe'] < val).mean()
        ax.axvline(val, color=color, lw=2, label=f'{name}: {val:.2f} (pctile {pct:.0%})')
    ax.set_xlabel(f'Annualised Sharpe, OOS, net of {_config.COST_BPS} bps, ex-{_config.RF_TICKER}', color=P['ink2'])
    ax.set_ylabel('Simulations', color=P['ink2'])
    ax.set_title('Where does the pooled GBDT sit against random selection?', color=P['ink'], loc='left')
    ax.legend(loc='upper right', frameon=False, fontsize=9)

    fig.subplots_adjust(top=0.94, bottom=0.08, left=0.06, right=0.98)
    fig.savefig(out_path, dpi=120, facecolor=P['surface'])
    plt.close(fig)


if __name__ == "__main__":
    rng = np.random.default_rng(_config.DIAG_SEED)
    top_n = _config.TOP_N

    print("=" * 70)
    print("📜 Pre-registration (原文，來自 _config.PRE_REGISTRATION_POOLED)")
    print("=" * 70)
    print(_config.PRE_REGISTRATION_POOLED)

    df = pd.read_csv("etf_adj_close_clean.csv", index_col="Date", parse_dates=True)
    assets = [c for c in df.columns if c != 'SPY']
    with quiet():
        X, Y = generate_ml_features(df)
        folds = usable_folds(X, Y)
    month_ret, complete = month_end_returns(df)
    alpha_m = month_ret[assets].sub(month_ret['SPY'], axis=0)
    rf = month_ret[_config.RF_TICKER]

    print("=" * 70)
    print("🔬 Pooled 橫斷面 GBDT 診斷")
    print("=" * 70)

    print("🌲 訓練 pooled 模型 (每個 fold 各自訓練 GBDT 與同表徵 ridge)...")
    gb_sc, rg_sc, info, imp = pooled_scores(X, Y, folds, assets)
    with quiet():
        mom_sc = calculate_momentum_features(df, verbose=False)
        lstm_sc, _ = generate_ml_scores(X, Y, folds, assets)

    print("\n📐 各 fold 訓練摘要 (zero_pred = 永遠預測 0 的 val MSE)")
    print(info.to_string(formatters={'n_train_rows': '{:,.0f}'.format, 'n_val_rows': '{:,.0f}'.format,
                                     'gbdt_best_iter': '{:.0f}'.format, 'ridge_alpha': '{:.0f}'.format},
                         float_format=lambda v: f"{v:.6f}"))
    beat_floor = info['gbdt_val_mse'] < info['zero_pred_val_mse']
    print(f"   GBDT 的 val MSE 低於「永遠預測 0」的 fold：{int(beat_floor.sum())}/{len(beat_floor)}")

    # --- (1) 預測力 ---
    ic = {'Pooled GBDT': monthly_rank_ic(gb_sc, alpha_m),
          'Pooled ridge': monthly_rank_ic(rg_sc, alpha_m),
          'Momentum': monthly_rank_ic(mom_sc.drop(columns='SPY'), alpha_m)}
    oos_months = ic['Pooled GBDT'].index
    ic['Momentum'] = ic['Momentum'].reindex(oos_months).dropna()
    ic_lstm = monthly_rank_ic(lstm_sc, alpha_m).reindex(oos_months).dropna()

    print("\n" + "-" * 70)
    print("📐 (1) 橫斷面 rank-IC（walk-forward OOS）")
    print("-" * 70)
    tab = pd.DataFrame({**{k: ic_stats(v) for k, v in ic.items()},
                        'LSTM (前一輪，僅供對照)': ic_stats(ic_lstm)}).T
    print(tab.to_string(formatters={'IC_hit': '{:.0%}'.format, 'n': '{:.0f}'.format},
                        float_format=lambda v: f"{v:+.3f}"))
    t_ridge, n_r = paired_t(ic['Pooled GBDT'], ic['Pooled ridge'])
    t_mom, n_m = paired_t(ic['Pooled GBDT'], ic['Momentum'])
    t_lstm, _ = paired_t(ic['Pooled GBDT'], ic_lstm)
    print(f"   配對 t：GBDT − pooled ridge = {t_ridge:+.2f} (n={n_r}) | "
          f"GBDT − 動能 = {t_mom:+.2f} (n={n_m}) | GBDT − LSTM = {t_lstm:+.2f}")

    # --- (2) 策略層 ---
    with quiet():
        gb_g, gb_w = backtest_momentum_strategy(df, gb_sc, top_n, _config.ML_MIN_SCORE, _config.ML_FALLBACK, 0)
        rg_g, rg_w = backtest_momentum_strategy(df, rg_sc, top_n, _config.ML_MIN_SCORE, _config.ML_FALLBACK, 0)
        mo_g, mo_w = backtest_momentum_strategy(df, mom_sc, top_n, _config.BASE_MIN_SCORE, _config.BASE_FALLBACK, 0)
    oos_idx = gb_g.index[(gb_g.index - pd.offsets.MonthEnd(1)) >= folds[0][1]['test_start']]
    oos_idx = oos_idx.intersection(mo_g.index).intersection(rg_g.index)
    dd = mo_w.dropna(how='all').index
    ew_w = constant_weights(assets, dd); spy_w = constant_weights(['SPY'], dd, weight=1.0)
    ew_g = (ew_w.reindex(month_ret.index).shift(1) * month_ret[assets]).sum(axis=1, min_count=1).dropna()

    def net(g, w):
        t = traded_fraction(w, month_ret).reindex(month_ret.index).shift(1)
        return (g - t * _config.COST_BPS / 1e4).reindex(oos_idx)

    series = {'Pooled GBDT': net(gb_g, gb_w), 'Pooled ridge': net(rg_g, rg_w),
              'Momentum': net(mo_g, mo_w), 'SPY': net(month_ret['SPY'].reindex(ew_g.index), spy_w),
              f'Equal-weight {len(assets)} ETFs': net(ew_g, ew_w)}
    wts = {'Pooled GBDT': gb_w, 'Pooled ridge': rg_w, 'Momentum': mo_w,
           'SPY': spy_w, f'Equal-weight {len(assets)} ETFs': ew_w}

    rows = []
    for name, r in series.items():
        m = perf(r, rf=rf, weights=wts[name])
        lo, hi = bootstrap_sharpe_ci(r - rf.reindex(r.index), _config.N_BOOTSTRAP, rng)
        rows.append({'strategy': name, **m, 'Sharpe_CI95_lo': lo, 'Sharpe_CI95_hi': hi})
    perf_tab = pd.DataFrame(rows).set_index('strategy')
    print("\n" + "-" * 70)
    print(f"📊 (2) 策略層（{len(oos_idx)} 個月，淨 {_config.COST_BPS} bps，Sharpe 扣 {_config.RF_TICKER}）")
    print("-" * 70)
    print(perf_tab.to_string(formatters={'CAGR': '{:.2%}'.format, 'MDD': '{:.2%}'.format, 'n': '{:.0f}'.format},
                             float_format=lambda v: f"{v:.2f}"))
    gbdt, mom = perf_tab.loc['Pooled GBDT'], perf_tab.loc['Momentum']
    se = gbdt['Sharpe_SE']
    d_sharpe = gbdt['Sharpe'] - mom['Sharpe']

    # --- (3) Null ---
    null_df = random_selection_null(month_ret.loc[oos_idx, assets].values, rf.reindex(oos_idx).values,
                                    top_n, _config.N_NULL_SIMS, rng, cost_bps=_config.COST_BPS)
    gb_pct = float((null_df['Sharpe'] < gbdt['Sharpe']).mean())
    print(f"\n🎲 (3) 隨機選股 null（{_config.N_NULL_SIMS} 次）：5% {null_df['Sharpe'].quantile(.05):.2f} / "
          f"50% {null_df['Sharpe'].quantile(.5):.2f} / 95% {null_df['Sharpe'].quantile(.95):.2f}")
    print(f"   GBDT {gbdt['Sharpe']:.2f} → {gb_pct:.0%} 百分位 | "
          f"動能 {mom['Sharpe']:.2f} → {(null_df['Sharpe'] < mom['Sharpe']).mean():.0%} 百分位")

    # --- (4) 逐 fold（判準 E）---
    lbl = fold_of_holding_month(oos_idx, folds)
    lbl_ic = fold_of_holding_month(oos_months + pd.offsets.MonthEnd(1), folds); lbl_ic.index = oos_months
    print("\n" + "-" * 70)
    print("📆 (4) 逐 fold（判準 E：符號不一致的 fold 不得省略）")
    print("-" * 70)
    rows = []
    for name, _ in folds:
        i_ic = lbl_ic.index[lbl_ic == name]
        i_r = lbl.index[lbl == name]
        rows.append({'fold': name.replace('test_', '').replace('-01-01', ''),
                     'IC_GBDT': ic['Pooled GBDT'].reindex(i_ic).mean(),
                     'IC_momentum': ic['Momentum'].reindex(i_ic).mean(),
                     'Sharpe_GBDT': perf(series['Pooled GBDT'].reindex(i_r).dropna(), rf)['Sharpe'],
                     'Sharpe_momentum': perf(series['Momentum'].reindex(i_r).dropna(), rf)['Sharpe']})
    fold_tab = pd.DataFrame(rows).set_index('fold')
    fold_tab['IC_diff'] = fold_tab['IC_GBDT'] - fold_tab['IC_momentum']
    print(fold_tab.to_string(float_format=lambda v: f"{v:+.2f}"))

    # --- 判定 ---
    a = tab.loc['Pooled GBDT']
    A = (a['IC_mean'] > 0) and (a['IC_t'] > 2)
    B = t_ridge > 2
    C = t_mom > 2
    D = (d_sharpe >= se) and (gb_pct >= 0.95)
    E_pos = (fold_tab['IC_diff'] > 0)
    F = abs(d_sharpe) >= 0.5 * se

    print("\n" + "=" * 70)
    print("⚖️ 依 pre-registration 判定（由程式計算，不得事後放寬）")
    print("=" * 70)
    print(f"  A 預測力         : IC {a['IC_mean']:+.3f}, t {a['IC_t']:+.2f} → "
          + ("成立：有樣本外排序能力" if A else "不成立：無證據顯示有排序能力"))
    print(f"  B 相對 pooled ridge: 配對 t {t_ridge:+.2f} → "
          + ("成立：非線性有存在理由" if B else "不成立：非線性不具存在理由"))
    print(f"  C 相對傳統動能    : 配對 t {t_mom:+.2f} → "
          + ("**成立：ML 勝過三行規則**" if C else "不成立：ML 沒有勝過一條三行的規則"))
    print(f"  D 策略層         : ΔSharpe {d_sharpe:+.2f} vs SE {se:.2f} ({'≥' if d_sharpe >= se else '<'}), "
          f"null 百分位 {gb_pct:.0%} ({'≥' if gb_pct >= 0.95 else '<'} 95%) → " + ("勝出" if D else "未勝出"))
    print(f"  E 一致性         : 逐 fold IC 優於動能者 {int(E_pos.sum())}/{len(E_pos)}"
          + ("" if E_pos.all() or not E_pos.any() else "；符號不一致，見上表"))
    print(f"  F 材料性         : |ΔSharpe| {abs(d_sharpe):.2f} = {abs(d_sharpe) / se:.2f} × SE → "
          + ("達材料性門檻 (≥ 0.5 SE)" if F else "**< 0.5 SE，報成「無實質改善」**"))
    print(f"      本輪競爭的上限（開跑前量定）：動能的『選股』成分僅值 "
          f"+{SELECTION_CEILING_SHARPE:.2f} Sharpe = {SELECTION_CEILING_SHARPE / se:.2f} × SE，且逐 fold 變號。")

    print("\n" + "=" * 70)
    print("📝 誠實聲明")
    print("=" * 70)
    print(f"  • Pooled 後列數 ×{len(assets)}，但同一日的 {len(assets)} 列高度相關，有效樣本增幅遠小於此。")
    print("  • Early stopping 用 val MSE，把同一日的相關列當獨立，會讓 early stopping 偏樂觀（開跑前已記錄）。")
    print(f"  • OOS 仍是 {len(oos_idx)} 個月、SE {se:.2f}：本輪即使『通過』也分不出真假。")
    print("  • 標的池事後自選、期間止於資料抓取日，兩者都會灌水。")

    perf_tab.reset_index().to_csv("pooled_summary.csv", index=False)
    pd.DataFrame(ic).to_csv("pooled_ic_monthly.csv")
    fold_tab.to_csv("pooled_by_fold.csv")
    info.to_csv("pooled_fold_info.csv")
    marks = {k: perf_tab.loc[k, 'Sharpe'] for k in ['Pooled GBDT', 'Pooled ridge', 'Momentum']}
    plot_report(ic, [f['test_start'] for _, f in folds], null_df, marks, "pooled_report.png")
    print("\n💾 已儲存: pooled_summary.csv, pooled_ic_monthly.csv, pooled_by_fold.csv, "
          "pooled_fold_info.csv, pooled_report.png")
