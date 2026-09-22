"""
第四輪：point-in-time 標的池、約三倍樣本上的重測與判定。

只換樣本，不換模型。參賽者為傳統動能、pooled ridge、pooled GBDT；
LSTM 不參賽（13 個輸出 head 假設固定池，改架構是本輪明文禁止的事——結構性排除，非績效原因）。

三條線共用**同一個入池規則**（252 個交易日歷史）、同一套 purge 與會計，確保可比。
報告開頭原文印出 _config.PRE_REGISTRATION_PIT，結尾依判準 A~F 機械化判定。
"""

import io
import contextlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import _config
from _1b_fetch_pit_data import eligibility, apply_pit_eligibility
from _2_baseline_performance import (calculate_momentum_features, backtest_momentum_strategy,
                                     month_end_returns, traded_fraction, constant_weights,
                                     slots_for_width, performance_metrics as perf)
from _3_ml_data_pipeline import generate_ml_features, fold_specs, make_fold
from _4c_pooled_gbdt import pooled_scores, to_panel, ASSET_FEATURES
from _6_diagnostics import bootstrap_sharpe_ci

PALETTE = {'gbdt': '#2a78d6', 'pridge': '#eb6834', 'mom': '#1baf7a', 'null': '#9ec5f4',
           'ink': '#0b0b0b', 'ink2': '#52514e', 'surface': '#fcfcfb', 'grid': '#e6e5e1'}
CONTROL_START = pd.Timestamp('2022-02-28')   # 對照軌：前三輪的樣本起點


@contextlib.contextmanager
def quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# ==========================================
# 1. PIT 輸入：一致地套用入池規則
# ==========================================
def pit_inputs(df):
    """
    回傳 (X, Y, mom_scores, width, assets)。
    X/Y/動能分數三者都經過同一個入池遮罩，並限制在橫斷面寬度 >= PIT_MIN_ASSETS 的日子。
    """
    assets = [c for c in df.columns if c != 'SPY']
    elig, _ = eligibility(df)

    with quiet():
        X, Y = generate_ml_features(df, require_all_assets=False)
        mom = calculate_momentum_features(df, verbose=False)

    X = apply_pit_eligibility(X, elig, assets, [f'_{f}' for f in ASSET_FEATURES])
    Y = apply_pit_eligibility(Y, elig, assets, ['_Target'])
    mom = apply_pit_eligibility(mom, elig, assets, [''])

    # 逐日橫斷面寬度（以入池為準），並限制在可交易的日子
    width = elig.reindex(X.index).ffill().fillna(False).sum(axis=1)
    keep = width[width >= _config.PIT_MIN_ASSETS].index
    return X.loc[keep], Y.loc[keep], mom, width.loc[keep], assets


# ==========================================
# 2. 寬度可比的 rank-IC
# ==========================================
def monthly_rank_ic_pit(scores, alpha_m, width_m, min_k=3):
    """
    回傳 DataFrame，欄位 IC / K / z。
      IC = 該決策日的橫斷面 Spearman 相關（分數 vs 下個月相對 SPY 的實現超額報酬）
      K  = 該月可排名檔數
      z  = IC × sqrt(K − 1)，虛無下變異約為 1，與寬度無關（寫定於 pre-registration）
    """
    s = scores.resample('ME').last()
    nxt = alpha_m.shift(-1)
    rows = {}
    for d in s.index:
        if d not in nxt.index:
            continue
        a, b = s.loc[d].dropna(), nxt.loc[d].dropna()
        common = a.index.intersection(b.index)
        if len(common) < min_k:
            continue
        ic = spearmanr(a[common], b[common]).correlation
        if np.isnan(ic):
            continue
        k = len(common)
        rows[d] = {'IC': ic, 'K': k, 'z': ic * np.sqrt(k - 1)}
    return pd.DataFrame(rows).T


def ic_summary(tab):
    """加權平均 IC（權重 K−1）、未加權平均、z 的 t 值、勝率。"""
    w = tab['K'] - 1
    z = tab['z']
    return {'n': len(tab),
            'IC_weighted': float((tab['IC'] * w).sum() / w.sum()),
            'IC_unweighted': float(tab['IC'].mean()),
            'z_t': float(z.mean() / z.std(ddof=1) * np.sqrt(len(z))),
            'hit': float((tab['IC'] > 0).mean())}


def paired_z_t(a, b):
    """兩個模型的 z 序列在共同月份的配對 t（寬度已標準化，可直接相減）。"""
    d = (a['z'] - b['z']).dropna()
    return float(d.mean() / d.std(ddof=1) * np.sqrt(len(d))), len(d)


# ==========================================
# 3. 隨機選股 null（寬度隨月變動）
# ==========================================
def pit_null(R, rf, widths, elig_m, rng, n_sims, cost_bps):
    """每個月從當月「可排名」的標的中隨機抽 slots_for_width(K) 檔等權，與策略同樣扣成本與 rf。"""
    T, K = R.shape
    picks_n = [slots_for_width(int(k)) for k in widths]
    masks = [np.flatnonzero(elig_m[t]) for t in range(T)]
    out = np.empty((n_sims, 3))
    for i in range(n_sims):
        W = np.zeros((T, K))
        for t in range(T):
            pool = masks[t]
            chosen = rng.choice(pool, min(picks_n[t], len(pool)), replace=False)
            W[t, chosen] = 1.0 / len(chosen)
        traded = np.abs(np.diff(W, axis=0, prepend=np.zeros((1, K)))).sum(axis=1)
        rs = np.nansum(W * R, axis=1) - traded * cost_bps / 1e4
        cum = np.cumprod(1 + rs)
        ex = rs - rf
        out[i] = [ex.mean() / ex.std(ddof=1) * np.sqrt(12),
                  cum[-1] ** (12 / T) - 1,
                  (cum / np.maximum(np.maximum.accumulate(cum), 1.0) - 1).min()]
    return pd.DataFrame(out, columns=['Sharpe', 'CAGR', 'MDD'])


# ==========================================
# 4. 繪圖
# ==========================================
def plot_report(ics, width_m, fold_starts, null_df, marks, out_path):
    P = PALETTE
    fig = plt.figure(figsize=(16, 11), facecolor=P['surface'])
    gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 0.45, 1], hspace=0.42)

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
    for fs in fold_starts:
        ax.axvline(fs, color=P['grid'], lw=1)
    ax.axhline(0, color=P['ink2'], lw=1)
    for (name, tab), color in zip(ics.items(), [P['mom'], P['pridge'], P['gbdt']]):
        r = tab['z'].rolling(12, min_periods=6).mean()
        ax.plot(r.index, r.values, color=color, lw=2.2, label=f"{name} (full-period mean z {tab['z'].mean():+.2f})")
    ax.set_ylabel('12-month rolling mean of width-adjusted IC (z)', color=P['ink2'])
    ax.set_title('Point-in-time universe, 2013–2026: rolling ranking ability', color=P['ink'], loc='left')
    ax.legend(loc='lower left', frameon=False, fontsize=9)

    ax = fig.add_subplot(gs[1]); style(ax)
    ax.fill_between(width_m.index, width_m.values, color=P['null'], alpha=0.7, step='mid', lw=0)
    ax.axhline(_config.PIT_NARROW_MAX_K + 0.5, color=P['ink2'], lw=1, ls='--')
    ax.text(width_m.index[2], _config.PIT_NARROW_MAX_K + 0.9, ' narrow / wide split',
            color=P['ink2'], fontsize=8)
    ax.set_ylabel('Assets in universe (K)', color=P['ink2'])
    ax.set_title('Cross-section width over time', color=P['ink'], loc='left')

    ax = fig.add_subplot(gs[2]); style(ax)
    ax.hist(null_df['Sharpe'], bins=45, color=P['null'], edgecolor=P['surface'], lw=0.5,
            label=f'Random picks from the same PIT universe (n={len(null_df)})')
    for (name, val), color in zip(marks.items(), [P['mom'], P['pridge'], P['gbdt']]):
        pct = (null_df['Sharpe'] < val).mean()
        ax.axvline(val, color=color, lw=2, label=f'{name}: {val:.2f} (pctile {pct:.0%})')
    ax.set_xlabel(f'Annualised Sharpe, OOS, net of {_config.COST_BPS} bps, ex-{_config.RF_TICKER}', color=P['ink2'])
    ax.set_ylabel('Simulations', color=P['ink2'])
    ax.set_title('Against random selection from the same point-in-time universe', color=P['ink'], loc='left')
    ax.legend(loc='upper right', frameon=False, fontsize=9)

    fig.subplots_adjust(top=0.95, bottom=0.06, left=0.07, right=0.98)
    fig.savefig(out_path, dpi=120, facecolor=P['surface'])
    plt.close(fig)


# ==========================================
# 5. 主流程
# ==========================================
if __name__ == "__main__":
    rng = np.random.default_rng(_config.DIAG_SEED)

    print("=" * 74)
    print("📜 Pre-registration (原文，來自 _config.PRE_REGISTRATION_PIT)")
    print("=" * 74)
    print(_config.PRE_REGISTRATION_PIT)

    df = pd.read_csv(_config.PIT_DATA_FILE, index_col="Date", parse_dates=True)
    X, Y, mom_sc, width, assets = pit_inputs(df)
    folds_all = fold_specs(X, _config.PIT_WF_TEST_STARTS)
    folds = [(n, make_fold(X, Y, ts, te, min_targets=_config.PIT_MIN_ASSETS)) for n, ts, te in folds_all]
    folds = [(n, f) for n, f in folds if len(f['train_pos']) >= _config.WF_MIN_TRAIN_SAMPLES]

    month_ret, complete = month_end_returns(df)
    alpha_m = month_ret[assets].sub(month_ret['SPY'], axis=0)
    rf = month_ret[_config.RF_TICKER]
    width_m = width.resample('ME').last().dropna()

    print("=" * 74)
    print("🔬 PIT 長樣本重測")
    print("=" * 74)
    print(f"📅 PIT 資料 {df.index[0].date()} ~ {df.index[-1].date()}；"
          f"可交易日 {X.index[0].date()} ~ {X.index[-1].date()}")
    print(f"✂️ {len(folds)} 個 fold（{folds[0][0]} ~ {folds[-1][0]}），入池門檻 {_config.PIT_LOOKBACK_DAYS} 天，"
          f"最小橫斷面 {_config.PIT_MIN_ASSETS}")
    print(f"📏 橫斷面寬度：{int(width_m.min())} ~ {int(width_m.max())}；"
          f"持股數 = max({_config.PIT_MIN_TOP_N}, round({_config.TOP_FRACTION}×K))")

    # --- 分數 ---
    print("\n🌲 訓練 pooled 模型（每個 fold 各自訓練）...")
    gb_sc, rg_sc, info, imp = pooled_scores(X, Y, folds, assets)

    # --- 回測（三條線同一套 PIT 會計）---
    bt = lambda sc, minsc, fb: backtest_momentum_strategy(
        df, sc, top_n=None, min_score=minsc, fallback=fb, cost_bps=0,
        top_fraction=_config.TOP_FRACTION, min_assets=_config.PIT_MIN_ASSETS)
    with quiet():
        mo_g, mo_w = bt(mom_sc, _config.BASE_MIN_SCORE, _config.BASE_FALLBACK)
        rg_g, rg_w = bt(rg_sc, _config.ML_MIN_SCORE, _config.ML_FALLBACK)
        gb_g, gb_w = bt(gb_sc, _config.ML_MIN_SCORE, _config.ML_FALLBACK)

    oos_start = folds[0][1]['test_start']
    oos_idx = mo_g.index[(mo_g.index - pd.offsets.MonthEnd(1)) >= oos_start]
    oos_idx = oos_idx.intersection(gb_g.index).intersection(rg_g.index)
    dd = mo_w.dropna(how='all').index
    ew_w = constant_weights(assets, dd); spy_w = constant_weights(['SPY'], dd, weight=1.0)
    ew_g = (ew_w.reindex(month_ret.index).shift(1) * month_ret[assets]).sum(axis=1, min_count=1).dropna()

    def net(g, w):
        t = traded_fraction(w, month_ret).reindex(month_ret.index).shift(1)
        return (g - t * _config.COST_BPS / 1e4).reindex(oos_idx)

    series = {'Momentum': net(mo_g, mo_w), 'Pooled ridge': net(rg_g, rg_w), 'Pooled GBDT': net(gb_g, gb_w),
              'SPY': net(month_ret['SPY'].reindex(ew_g.index), spy_w),
              'Equal-weight (PIT)': net(ew_g, ew_w)}
    wts = {'Momentum': mo_w, 'Pooled ridge': rg_w, 'Pooled GBDT': gb_w,
           'SPY': spy_w, 'Equal-weight (PIT)': ew_w}

    # --- (1) 寬度可比的 IC ---
    ics = {'Momentum': monthly_rank_ic_pit(mom_sc, alpha_m, width_m),
           'Pooled ridge': monthly_rank_ic_pit(rg_sc, alpha_m, width_m),
           'Pooled GBDT': monthly_rank_ic_pit(gb_sc, alpha_m, width_m)}
    oos_months = ics['Pooled GBDT'].index
    ics = {k: v.reindex(oos_months).dropna() for k, v in ics.items()}

    print("\n" + "-" * 74)
    print("📐 (1) 橫斷面 rank-IC（寬度標準化：檢定用 z = IC×sqrt(K−1)，效果量用 (K−1) 加權）")
    print("-" * 74)
    ic_tab = pd.DataFrame({k: ic_summary(v) for k, v in ics.items()}).T
    print(ic_tab.to_string(formatters={'hit': '{:.0%}'.format, 'n': '{:.0f}'.format},
                           float_format=lambda v: f"{v:+.3f}"))
    t_rg, _ = paired_z_t(ics['Pooled ridge'], ics['Momentum'])
    t_gb, _ = paired_z_t(ics['Pooled GBDT'], ics['Momentum'])
    t_gr, _ = paired_z_t(ics['Pooled GBDT'], ics['Pooled ridge'])
    print(f"   配對 t（z 序列）：ridge − 動能 = {t_rg:+.2f} | GBDT − 動能 = {t_gb:+.2f} | GBDT − ridge = {t_gr:+.2f}")

    # --- (2) 策略層 ---
    rows = []
    for name, r in series.items():
        m = perf(r, rf=rf, weights=wts[name])
        lo, hi = bootstrap_sharpe_ci(r - rf.reindex(r.index), _config.N_BOOTSTRAP, rng)
        rows.append({'strategy': name, **m, 'CI95_lo': lo, 'CI95_hi': hi})
    perf_tab = pd.DataFrame(rows).set_index('strategy')
    print("\n" + "-" * 74)
    print(f"📊 (2) 策略層（{len(oos_idx)} 個月 {oos_idx[0]:%Y-%m}~{oos_idx[-1]:%Y-%m}，"
          f"淨 {_config.COST_BPS} bps，Sharpe 扣 {_config.RF_TICKER}）")
    print("-" * 74)
    print(perf_tab.to_string(formatters={'CAGR': '{:.2%}'.format, 'MDD': '{:.2%}'.format, 'n': '{:.0f}'.format},
                             float_format=lambda v: f"{v:.2f}"))
    momp = perf_tab.loc['Momentum']
    se = momp['Sharpe_SE']

    # --- (3) null ---
    elig_full, _ = eligibility(df)
    elig_m = elig_full.resample('ME').last().reindex(oos_idx).fillna(False)
    R = month_ret.loc[oos_idx, assets].values
    null_df = pit_null(np.nan_to_num(R), rf.reindex(oos_idx).values,
                       width_m.reindex(oos_idx - pd.offsets.MonthEnd(1)).values,
                       elig_m[assets].values, rng, _config.N_NULL_SIMS, _config.COST_BPS)
    pct = {k: float((null_df['Sharpe'] < perf_tab.loc[k, 'Sharpe']).mean())
           for k in ['Momentum', 'Pooled ridge', 'Pooled GBDT']}
    print(f"\n🎲 (3) 隨機選股 null（同一 PIT 池、同樣的逐月持股數，{_config.N_NULL_SIMS} 次）：")
    print(f"   5% {null_df['Sharpe'].quantile(.05):.2f} / 50% {null_df['Sharpe'].quantile(.5):.2f} / "
          f"95% {null_df['Sharpe'].quantile(.95):.2f}")
    for k, v in pct.items():
        print(f"   {k:<14} {perf_tab.loc[k, 'Sharpe']:.2f} → {v:.0%} 百分位")

    # --- (4) 逐 fold 與危機年（判準 D）---
    year = pd.Series(oos_idx.year, index=oos_idx)
    print("\n" + "-" * 74)
    print("📆 (4) 逐年（判準 D；★ = 預先指定的壓力年）")
    print("-" * 74)
    rows = {}
    for y, idx in year.groupby(year):
        i = idx.index
        if len(i) < 6:
            continue
        rows[f"{y}{'★' if y in _config.PIT_CRISIS_YEARS else ''}"] = {
            k: perf(v.reindex(i).dropna(), rf)['Sharpe'] for k, v in series.items()}
    year_tab = pd.DataFrame(rows).T
    print(year_tab.to_string(float_format=lambda v: f"{v:6.2f}"))
    crisis_idx = oos_idx[oos_idx.year.isin(_config.PIT_CRISIS_YEARS)]
    print("\n   壓力年合併（{} 個月）：".format(len(crisis_idx)))
    print("   " + " | ".join(f"{k} {perf(v.reindex(crisis_idx).dropna(), rf)['Sharpe']:+.2f}"
                             for k, v in series.items()))

    # --- (5) 窄/寬分段（判準 E）---
    print("\n" + "-" * 74)
    print(f"📏 (5) 窄（K ≤ {_config.PIT_NARROW_MAX_K}）/ 寬（K > {_config.PIT_NARROW_MAX_K}）分段（判準 E）")
    print("-" * 74)
    seg_rows = {}
    for label, sel in [('narrow', lambda t: t['K'] <= _config.PIT_NARROW_MAX_K),
                       ('wide', lambda t: t['K'] > _config.PIT_NARROW_MAX_K)]:
        seg_rows[label] = {k: ic_summary(v[sel(v)])['z_t'] if sel(v).sum() > 2 else np.nan
                           for k, v in ics.items()}
        seg_rows[label]['n_months'] = int(sel(ics['Momentum']).sum())
    print(pd.DataFrame(seg_rows).T.to_string(float_format=lambda v: f"{v:+.2f}"))
    print("   (值為 z 序列的 t；n_months 為該段月數)")

    # --- (6) 對照軌：2022-02 起、K=13（判準的強制揭露）---
    ctrl = oos_idx[oos_idx >= CONTROL_START]
    print("\n" + "-" * 74)
    print(f"🔁 (6) 對照軌：同一管線在 {CONTROL_START:%Y-%m} 之後的子期（{len(ctrl)} 個月，K=13）")
    print("-" * 74)
    ctrl_tab = pd.DataFrame({k: perf(v.reindex(ctrl).dropna(), rf) for k, v in series.items()}).T
    print(ctrl_tab[['CAGR', 'MDD', 'Sharpe', 'Sharpe_SE']].to_string(
        formatters={'CAGR': '{:.2%}'.format, 'MDD': '{:.2%}'.format}, float_format=lambda v: f"{v:.2f}"))
    print("   前三輪（固定池、TOP_N=5、2022-02~2026-08）的對應數字：動能 Sharpe 1.06、ridge 0.78、GBDT 0.46。")

    # --- 判定 ---
    m_ic = ic_tab.loc['Momentum']
    A = (m_ic['IC_weighted'] > 0) and (m_ic['z_t'] > 2)
    B = pct['Momentum'] >= 0.95
    ml_win = {}
    for k, t in [('Pooled ridge', t_rg), ('Pooled GBDT', t_gb)]:
        d = perf_tab.loc[k, 'Sharpe'] - momp['Sharpe']
        ml_win[k] = (t > 2 and d >= se, t, d)
    seg = pd.DataFrame(seg_rows).T
    E_flags = {k: np.sign(seg.loc['narrow', k]) == np.sign(seg.loc['wide', k]) for k in ics}

    print("\n" + "=" * 74)
    print("⚖️ 依 pre-registration 判定（由程式計算，不得事後放寬）")
    print("=" * 74)
    print(f"  A 動能存續   : 加權 IC {m_ic['IC_weighted']:+.3f}, z 的 t {m_ic['z_t']:+.2f} → "
          + ("成立：動能的排序能力在長樣本、多 regime 下站得住"
             if A else "**不成立：前三輪「動能是唯一有效訊號」的結論作廢**"))
    print(f"  B 動能策略層 : Sharpe {momp['Sharpe']:.2f}，null 百分位 {pct['Momentum']:.0%} → "
          + ("成立" if B else "不成立：與同池隨機選股分不開"))
    for k, (win, t, d) in ml_win.items():
        print(f"  C {k:<13}: 配對 t {t:+.2f}（需 > 2），ΔSharpe {d:+.2f} vs SE {se:.2f} → "
              + ("**勝出**" if win else "未勝出"))
    print(f"  D regime     : 逐年見上表；壓力年 {_config.PIT_CRISIS_YEARS} 已單獨並列")
    for k, ok in E_flags.items():
        print(f"  E 窄/寬一致  : {k:<13} narrow t {seg.loc['narrow', k]:+.2f} / wide t {seg.loc['wide', k]:+.2f} → "
              + ("方向一致" if ok else "**方向不一致 → 該模型的結論即為『不一致』，不得以匯總掩蓋**"))
    print(f"  F 材料性     : " + " | ".join(
        f"{k} {abs(perf_tab.loc[k, 'Sharpe'] - momp['Sharpe']) / se:.2f}×SE"
        f"{'' if abs(perf_tab.loc[k, 'Sharpe'] - momp['Sharpe']) >= 0.5 * se else '(無實質差異)'}"
        for k in ['Pooled ridge', 'Pooled GBDT']))

    print("\n" + "=" * 74)
    print("📝 誠實聲明（pre-registration 強制）")
    print("=" * 74)
    print(f"  • 早期橫斷面只有 {int(width_m.min())} 檔，資產構成與後期不同（無商品、國際、小型價值、管理期貨）。")
    print("    結論若與前三輪不同，可能來自樣本量，也可能來自池組成，本輪無法分離——對照軌見 (6)。")
    print(f"  • OOS {len(oos_idx)} 個月，動能的 SE {se:.2f}（前三輪為 0.48）。真實效果若小於約 {0.5 * se:.2f} Sharpe 仍測不出。")
    print("  • BIL 同時在可排名池內又是 rf 代理；窄橫斷面時期挑到 BIL 等於持有現金，動能與擇時在此混合。")
    print("  • LSTM 未參賽：結構性排除（固定 13 head），非績效原因。")

    perf_tab.reset_index().to_csv("pit_summary.csv", index=False)
    pd.concat({k: v for k, v in ics.items()}, names=['model']).to_csv("pit_ic_monthly.csv")
    year_tab.to_csv("pit_by_year.csv")
    info.to_csv("pit_fold_info.csv")
    marks = {k: perf_tab.loc[k, 'Sharpe'] for k in ['Momentum', 'Pooled ridge', 'Pooled GBDT']}
    plot_report(ics, width_m[width_m.index >= oos_idx[0]], [f['test_start'] for _, f in folds],
                null_df, marks, "pit_report.png")
    print("\n💾 已儲存: pit_summary.csv, pit_ic_monthly.csv, pit_by_year.csv, pit_fold_info.csv, pit_report.png")
