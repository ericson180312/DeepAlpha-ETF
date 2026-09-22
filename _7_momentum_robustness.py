"""
動能基準的穩健性檢驗：基準值 (MOM_WINDOWS × TOP_N) 是坐在高原上，還是一根孤峰？

上一輪判定 LSTM 無樣本外排序能力後，傳統動能成為唯一支撐結論的訊號——而它的三個窗口與 TOP_N
是承襲自舊程式的未檢驗選擇。若基準格是網格裡的孤峰，「動能有效」這個結論就是過擬合的產物。

做法：把窗口三元組整體縮放 k 倍 (MOM_SCALE_GRID) × TOP_N (MOM_TOPN_GRID)，
在**同一段樣本外月份、同一套會計** (淨成本、扣 rf) 下重算每一格，再依 _config.PRE_REGISTRATION_MOMENTUM
機械化判定。判準原文印在報告開頭，判定結果印在結尾。

⚠️ 30 格共用同一段月份，彼此高度相關，網格離散程度低估真實不確定性。
   本模組只回答「基準值在網格內是否異常」，不回答「動能的 Sharpe 有多可靠」。
"""

import io
import contextlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import _config
from _2_baseline_performance import (calculate_momentum_features, backtest_momentum_strategy,
                                     month_end_returns, performance_metrics)
from _3_ml_data_pipeline import generate_ml_features
from _5_strategy_backtest import usable_folds, fold_of_holding_month

PALETTE = {'ink': '#0b0b0b', 'ink2': '#52514e', 'surface': '#fcfcfb', 'grid': '#e6e5e1',
           'target': '#2a78d6'}
SEQ_COLORS = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def scaled_windows(base_windows, k):
    """把窗口三元組整體縮放 k 倍，四捨五入到整數交易日，並確保嚴格遞增。"""
    out = []
    for w in base_windows:
        v = max(2, int(round(w * k)))
        out.append(max(v, out[-1] + 1) if out else v)
    return out


def grid_returns(df, oos_index, scales=None, top_ns=None):
    """
    回傳 {(k, top_n): 該格的淨報酬序列 (已對齊 oos_index)}，以及各格的窗口設定。
    每一格都走 _2 的同一套會計，唯一不同的是動能窗口與 TOP_N。
    """
    scales = _config.MOM_SCALE_GRID if scales is None else scales
    top_ns = _config.MOM_TOPN_GRID if top_ns is None else top_ns
    returns, windows = {}, {}
    for k in scales:
        w = scaled_windows(_config.MOM_WINDOWS, k)
        windows[k] = w
        with quiet():
            score = calculate_momentum_features(df, windows=w, verbose=False)
            for n in top_ns:
                r, _ = backtest_momentum_strategy(df, score, n, _config.BASE_MIN_SCORE,
                                                  _config.BASE_FALLBACK, _config.COST_BPS)
                returns[(k, n)] = r.reindex(oos_index)
    return returns, windows


def grid_table(returns, rf, metric='Sharpe'):
    """列 = scale，欄 = TOP_N。"""
    cells = {}
    for (k, n), r in returns.items():
        cells.setdefault(k, {})[n] = performance_metrics(r.dropna(), rf=rf)[metric]
    return pd.DataFrame(cells).T.sort_index(axis=0).sort_index(axis=1)


def neighbours(tab, k, n):
    """基準格在兩條軸上各 ±1 階的相鄰格數值 (不繞邊界)。"""
    rows, cols = list(tab.index), list(tab.columns)
    i, j = rows.index(k), cols.index(n)
    out = []
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        a, b = i + di, j + dj
        if 0 <= a < len(rows) and 0 <= b < len(cols):
            out.append(tab.loc[rows[a], cols[b]])
    return np.array(out)


def plateau_centre(tab):
    """平原中心：鄰域平均最高、且鄰域全距最小者。只在判準 C 觸發時使用。"""
    best, score = None, None
    for k in tab.index:
        for n in tab.columns:
            nb = neighbours(tab, k, n)
            s = (nb.mean(), -(nb.max() - nb.min()))
            if score is None or s > score:
                best, score = (k, n), s
    return best


def plot_grid(tab, base_cell, se, out_path):
    P = PALETTE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor=P['surface'],
                                   gridspec_kw={'width_ratios': [1.15, 1]})
    for ax in (ax1, ax2):
        ax.set_facecolor(P['surface'])
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(P['grid'])
        ax.tick_params(colors=P['ink2'])

    # (a) 熱圖：以「與基準格的差距 ÷ SE」著色，發散色階、灰色中點 = 無實質差異
    diff = (tab - tab.loc[base_cell]) / se
    lim = max(0.75, float(np.abs(diff.values).max()))
    im = ax1.imshow(diff.values, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
    ax1.set_xticks(range(len(tab.columns)), [f'Top {c}' for c in tab.columns])
    ax1.set_yticks(range(len(tab.index)), [f'×{i:g}' for i in tab.index])
    for i, k in enumerate(tab.index):
        for j, n in enumerate(tab.columns):
            is_base = (k, n) == base_cell
            ax1.text(j, i, f"{tab.loc[k, n]:.2f}", ha='center', va='center', fontsize=10,
                     color=P['ink'], fontweight='bold' if is_base else 'normal')
            if is_base:
                ax1.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                            edgecolor=P['target'], lw=3))
    ax1.set_xlabel('TOP_N', color=P['ink2'])
    ax1.set_ylabel('Window scale k', color=P['ink2'])
    ax1.set_title('OOS Sharpe across the momentum grid (box = current setting)', color=P['ink'], loc='left')
    cb = fig.colorbar(im, ax=ax1, fraction=0.046)
    cb.set_label('(cell − current) / SE', color=P['ink2'])
    cb.ax.tick_params(colors=P['ink2'])

    # (b) 沿 TOP_N 的剖面，每條線一個 scale：看基準格是平台還是尖峰
    for color, k in zip(SEQ_COLORS[1:], tab.index):
        ax2.plot(tab.columns, tab.loc[k], color=color, lw=2, marker='o', ms=5, label=f'k = {k:g}')
    ax2.axhline(tab.loc[base_cell], color=P['target'], ls='--', lw=1.5)
    ax2.scatter([base_cell[1]], [tab.loc[base_cell]], s=110, facecolor='none',
                edgecolor=P['target'], lw=2.5, zorder=5, label='current setting')
    ax2.fill_between(tab.columns, tab.loc[base_cell] - se, tab.loc[base_cell] + se,
                     color=P['target'], alpha=0.08, lw=0)
    ax2.set_xticks(list(tab.columns))
    ax2.set_xlabel('TOP_N', color=P['ink2'])
    ax2.set_ylabel('OOS Sharpe (net, ex-rf)', color=P['ink2'])
    ax2.set_title(f'Profiles along TOP_N (shaded band = ±1 SE of {se:.2f})', color=P['ink'], loc='left')
    ax2.grid(True, color=P['grid'], lw=0.8, alpha=0.6)
    ax2.set_axisbelow(True)
    ax2.legend(frameon=False, fontsize=9, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, facecolor=P['surface'])
    plt.close(fig)


if __name__ == "__main__":
    print("=" * 66)
    print("📜 Pre-registration (原文，來自 _config.PRE_REGISTRATION_MOMENTUM)")
    print("=" * 66)
    print(_config.PRE_REGISTRATION_MOMENTUM)

    df = pd.read_csv("etf_adj_close_clean.csv", index_col="Date", parse_dates=True)
    with quiet():
        X, Y = generate_ml_features(df)
        folds = usable_folds(X, Y)
        base_score = calculate_momentum_features(df, verbose=False)
        base_ret, _ = backtest_momentum_strategy(df, base_score, _config.TOP_N, _config.BASE_MIN_SCORE,
                                                 _config.BASE_FALLBACK, _config.COST_BPS)
    month_ret, complete = month_end_returns(df)
    rf = month_ret[_config.RF_TICKER]

    # 與 _5 / _6 完全相同的樣本外月份，確保可比
    oos_start = folds[0][1]['test_start']
    oos_index = base_ret.index[(base_ret.index - pd.offsets.MonthEnd(1)) >= oos_start]

    base_cell = (1.0, _config.TOP_N)
    base_metrics = performance_metrics(base_ret.reindex(oos_index).dropna(), rf=rf)
    se = base_metrics['Sharpe_SE']

    print("=" * 66)
    print("🔬 動能穩健性檢驗")
    print("=" * 66)
    print(f"📅 樣本外月份 {len(oos_index)} 個 ({oos_index[0].strftime('%Y-%m')} ~ {oos_index[-1].strftime('%Y-%m')})，"
          f"成本 {_config.COST_BPS} bps，Sharpe 扣 {_config.RF_TICKER}")
    print(f"🎯 基準格 = 窗口 {_config.MOM_WINDOWS} × Top {_config.TOP_N}："
          f"Sharpe {base_metrics['Sharpe']:.2f} ± {se:.2f}，CAGR {base_metrics['CAGR']:.2%}")

    returns, windows = grid_returns(df, oos_index)
    sharpe = grid_table(returns, rf, 'Sharpe')
    cagr = grid_table(returns, rf, 'CAGR')

    print("\n窗口縮放對照：" + " | ".join(f"×{k:g} → {w}" for k, w in windows.items()))
    print("\n" + "-" * 66)
    print("📊 網格 OOS Sharpe (列 = 窗口縮放 k，欄 = TOP_N)")
    print("-" * 66)
    print(sharpe.to_string(float_format=lambda v: f"{v:6.2f}"))
    print("\n📊 網格 OOS CAGR")
    print(cagr.to_string(float_format=lambda v: f"{v:7.1%}"))

    flat = sharpe.stack()
    base_sharpe = sharpe.loc[base_cell]
    pct = float((flat < base_sharpe).mean())
    nb = neighbours(sharpe, *base_cell)
    spike = base_sharpe - nb.mean()
    unquantised = (base_sharpe - flat.median()) / se

    print("\n" + "-" * 66)
    print("📐 讀數")
    print("-" * 66)
    print(f"  網格 Sharpe：min {flat.min():.2f} / 中位數 {flat.median():.2f} / max {flat.max():.2f}"
          f"（全距 {flat.max() - flat.min():.2f} = {(flat.max() - flat.min()) / se:.1f} × SE）")
    print(f"  基準格百分位（量化，解析度 1/{len(flat)}）：{pct:.0%}")
    print(f"  基準格 − 相鄰 {len(nb)} 格平均：{spike:+.2f}（{spike / se:+.2f} × SE）")
    print(f"  未量化交叉檢查：(基準 − 中位數) ÷ SE = {unquantised:+.2f}")

    # --- 逐 fold (判準 D) ---
    fold_label = fold_of_holding_month(oos_index, folds)
    print("\n📆 逐 fold 的基準格百分位（判準 D）")
    fold_rows = []
    for name in [n for n, _ in folds]:
        idx = fold_label.index[fold_label == name]
        if len(idx) < 6:
            continue
        f_tab = grid_table({key: r.reindex(idx) for key, r in returns.items()}, rf, 'Sharpe')
        f_flat = f_tab.stack()
        fold_rows.append({'fold': name, 'n_months': len(idx), 'base_Sharpe': f_tab.loc[base_cell],
                          'grid_median': f_flat.median(), 'pctile': float((f_flat < f_tab.loc[base_cell]).mean())})
    fold_tab = pd.DataFrame(fold_rows).set_index('fold')
    print(fold_tab.to_string(formatters={'pctile': '{:.0%}'.format, 'n_months': '{:.0f}'.format},
                             float_format=lambda v: f"{v:.2f}"))

    # --- 機械化判定 ---
    A = 0.25 <= pct <= 0.75
    B = spike > se
    C = pct < 0.25
    fold_A = fold_tab['pctile'].between(0.25, 0.75)

    print("\n" + "=" * 66)
    print("⚖️ 依 pre-registration 判定（由程式計算，不得事後放寬）")
    print("=" * 66)
    print(f"  判準 A 高原      : 基準格百分位 {pct:.0%} → "
          + (f"落在 25~75%，判為高原，**維持 {_config.MOM_WINDOWS} × Top {_config.TOP_N}**" if A
             else "未落在 25~75%，見判準 B / C"))
    print(f"  判準 B 尖峰否決  : 基準 − 鄰域平均 = {spike / se:+.2f} × SE → "
          + ("**> 1 SE，判為過擬合產物，上一輪「動能是唯一有效訊號」的結論作廢**" if B
             else "≤ 1 SE，不是孤峰"))
    print(f"  判準 C 劣位      : " + (f"百分位 {pct:.0%} < 25%，判為劣位；平原中心為 {plateau_centre(sharpe)}"
                                       if C else "未觸發，因此**禁止**輸出任何「建議改用 X」"))
    print(f"  判準 D 逐期間    : {int(fold_A.sum())}/{len(fold_A)} 個 fold 的基準格落在 25~75%"
          + ("" if fold_A.all() or not fold_A.any() else "；與匯總不一致的 fold 見上表，不得省略"))
    print(f"  判準 E 交叉檢查  : 未量化讀數 {unquantised:+.2f} × SE → "
          + ("與 A 一致（同判為無實質差異）" if (abs(unquantised) <= 0.5) == A
             else "**與 A 相左，依 pre-registration 以未量化者為準**"))

    print("\n" + "=" * 66)
    print("📝 誠實聲明")
    print("=" * 66)
    print(f"  • {len(flat)} 格共用同一段 {len(oos_index)} 個月，彼此高度相關；網格離散程度低估真實不確定性。")
    print("  • 本輪只回答「基準值在網格內是否異常」，不回答「動能的 Sharpe 有多可靠」——後者見 _6 的 SE 與 null 百分位。")
    print("  • 窗口以整體縮放的方式檢驗，未獨立擾動三個窗口；這是刻意的降維，代價是測不到三者的交互作用。")

    sharpe.to_csv("momentum_grid_sharpe.csv")
    cagr.to_csv("momentum_grid_cagr.csv")
    fold_tab.to_csv("momentum_grid_by_fold.csv")
    plot_grid(sharpe, base_cell, se, "momentum_robustness.png")
    print("\n💾 已儲存: momentum_grid_sharpe.csv, momentum_grid_cagr.csv, momentum_grid_by_fold.csv, momentum_robustness.png")
