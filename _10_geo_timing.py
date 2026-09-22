"""
第五輪：動能在「美股 vs 國際股」軸上有沒有擇時能力。

設計理由（完整版見 _config.PRE_REGISTRATION_GEO）：
  基準 VT 由標的池（VTI + 國際）本身組成，策略與基準共用絕大部分變異，
  所以 ΔSharpe 的雜訊 SD 只有 0.06~0.08，而 13 檔對 SPY 的設定是 0.255。

核心分解——把「平均傾斜」與「擇時」分開：
  a_t ∈ {0,1} 為策略當月的美股權重，w̄ = mean(a_t)。
  擇時報酬 r_timing,t = (a_t − w̄) × (R_US,t − R_intl,t)
  主動權重的平均恰為 0，平均傾斜被完全移除，剩下的就是純擇時。
  這是必要的，因為美股在每個可取得的窗口都大勝國際股——
  任何長期持有美股的規則都會贏過 VT，那不是擇時能力。

動能是固定規則，不需訓練，因此沒有 fold：全樣本皆為樣本外。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest

import _config
from _2_baseline_performance import month_end_returns, performance_metrics as perf

PALETTE = {'mom': '#2a78d6', 'tilt': '#eb6834', 'vt': '#1baf7a', 'ink': '#0b0b0b',
           'ink2': '#52514e', 'surface': '#fcfcfb', 'grid': '#e6e5e1', 'band': '#cde2fb'}


# ==========================================
# 1. 訊號與部位
# ==========================================
def eligible_from(prices, lookback=None):
    """每檔標的擁有 lookback 個交易日歷史之後才可入池（與第四輪同一規則）。"""
    lookback = _config.R5_LOOKBACK_DAYS if lookback is None else lookback
    return prices.notna().cumsum() >= lookback


def momentum_pick(prices, us, intl, windows=None):
    """
    回傳以決策日（月底）為索引的 Series：True = 全押美股，False = 全押國際。
    分數為 MOM_WINDOWS 各窗口報酬的等權平均，與本專案其他輪完全相同的定義。
    """
    windows = _config.MOM_WINDOWS if windows is None else windows
    score = sum(prices[[us, intl]].pct_change(w) for w in windows) / len(windows)
    elig = eligible_from(prices[[us, intl]])
    score = score.where(elig)
    monthly = score.resample('ME').last().dropna()
    return monthly[us] > monthly[intl]


def strategy_returns(pick, monthly_ret, us, intl, cost_bps=0.0):
    """
    決策日的選擇 -> 持有月報酬（shift(1)，避免未來函數）。
    全押單一標的，所以成交金額比例在換邊的月份為 2.0（賣光一檔、買滿另一檔），首月建倉為 1.0。
    """
    a = pick.astype(float)                      # 決策日的美股權重 ∈ {0,1}
    a_held = a.reindex(monthly_ret.index).shift(1).dropna()
    spread = (monthly_ret[us] - monthly_ret[intl]).reindex(a_held.index)
    gross = a_held * monthly_ret[us].reindex(a_held.index) + (1 - a_held) * monthly_ret[intl].reindex(a_held.index)

    switched = a_held.diff().abs().fillna(0.0)  # 0 或 1
    traded = switched * 2.0
    traded.iloc[0] = 1.0                        # 首月建倉
    return (gross - traded * cost_bps / 1e4).dropna(), a_held, spread, traded


def timing_decomposition(a_held, spread):
    """
    r_timing = (a − w̄) × 價差。主動權重均值恰為 0，平均傾斜的貢獻被移除。
    回傳 (擇時報酬序列, w̄)。
    """
    w_bar = float(a_held.mean())
    return (a_held - w_bar) * spread, w_bar


def static_tilt_returns(w, monthly_ret, index, us, intl, cost_bps=0.0):
    """固定權重 w 的美股 / (1−w) 國際，月再平衡。成本用漂移後的再平衡換手。"""
    r_us, r_intl = monthly_ret[us].reindex(index), monthly_ret[intl].reindex(index)
    gross = w * r_us + (1 - w) * r_intl
    grown_us = w * (1 + r_us)
    drift = grown_us / (grown_us + (1 - w) * (1 + r_intl))
    traded = (w - drift).abs() * 2.0            # 兩邊各調整一次
    traded.iloc[0] = 1.0
    return gross - traded * cost_bps / 1e4


# ==========================================
# 2. 統計
# ==========================================
def t_stat(x):
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def paired_sharpe_ci(a, b, rf, rng, n=20000, block=None):
    """配對 bootstrap 的 ΔSharpe 信賴區間（同時重抽兩條線的同一批月份，保留相關結構）。"""
    idx = a.index.intersection(b.index)
    A, B, R = a.reindex(idx).values, b.reindex(idx).values, rf.reindex(idx).values
    T = len(idx)
    d = np.empty(n)
    for i in range(n):
        if block:
            starts = rng.integers(0, T, size=int(np.ceil(T / block)))
            sel = np.concatenate([np.arange(s, s + block) % T for s in starts])[:T]
        else:
            sel = rng.integers(0, T, size=T)
        ea, eb = A[sel] - R[sel], B[sel] - R[sel]
        d[i] = ea.mean() / ea.std(ddof=1) * np.sqrt(12) - eb.mean() / eb.std(ddof=1) * np.sqrt(12)
    sh = lambda x: ((x - R).mean() / (x - R).std(ddof=1) * np.sqrt(12))
    obs = sh(A) - sh(B)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return obs, lo, hi, float(2 * min((d <= 0).mean(), (d >= 0).mean()))


# ==========================================
# 3. 單一軌道
# ==========================================
def run_track(prices, monthly_ret, rf, intl, rng, ceiling):
    us, bm = _config.R5_US, _config.R5_BENCHMARK
    pick = momentum_pick(prices, us, intl)
    strat, a_held, spread, traded = strategy_returns(pick, monthly_ret, us, intl, _config.COST_BPS)
    idx = strat.index[monthly_ret[bm].reindex(strat.index).notna()]
    strat, a_held, spread, traded = strat.loc[idx], a_held.loc[idx], spread.loc[idx], traded.loc[idx]

    r_timing, w_bar = timing_decomposition(a_held, spread)
    tilt = static_tilt_returns(w_bar, monthly_ret, idx, us, intl, _config.COST_BPS)
    vt = monthly_ret[bm].reindex(idx)

    lines = {'Momentum (all-in)': strat, f'Static tilt w={w_bar:.2f}': tilt, bm: vt,
             f'{us} buy & hold': monthly_ret[us].reindex(idx),
             f'{intl} buy & hold': monthly_ret[intl].reindex(idx)}

    # 完美 / 最差預知：本輪的材料性尺規
    oracle = pd.concat([monthly_ret[us], monthly_ret[intl]], axis=1).max(axis=1).reindex(idx)
    hits = (a_held.astype(bool) == (spread > 0))

    return {'idx': idx, 'lines': lines, 'strat': strat, 'tilt': tilt, 'vt': vt,
            'r_timing': r_timing, 'w_bar': w_bar, 'hits': hits, 'traded': traded,
            'oracle': oracle, 'spread': spread, 'a': a_held, 'ceiling': ceiling,
            'label_en': f"{_config.R5_US}+{intl}"}


def report_track(name, tr, rf, rng):
    idx, lines = tr['idx'], tr['lines']
    print("\n" + "=" * 74)
    print(f"🌍 {name}：{len(idx)} 個月 {idx[0]:%Y-%m} ~ {idx[-1]:%Y-%m}（淨 {_config.COST_BPS} bps，Sharpe 扣 {_config.RF_TICKER}）")
    print("=" * 74)
    tab = pd.DataFrame({k: perf(v.dropna(), rf) for k, v in lines.items()}).T
    print(tab[['n', 'CAGR', 'MDD', 'Sharpe', 'Sharpe_SE']].to_string(
        formatters={'CAGR': '{:.2%}'.format, 'MDD': '{:.2%}'.format, 'n': '{:.0f}'.format},
        float_format=lambda v: f"{v:.2f}"))
    print(f"   策略平均美股權重 w̄ = {tr['w_bar']:.3f}；換邊 {int((tr['traded'] > 1.5).sum())} 次／{len(idx)} 個月")

    # --- A 擇時 ---
    rt = tr['r_timing']
    tA = t_stat(rt)
    print(f"\n📐 A 擇時報酬 (a − w̄)×價差：均值 {rt.mean() * 12:+.2%}/年，t = {tA:+.2f}")

    # --- B / C 組合層 ---
    print("\n📊 B/C 配對 bootstrap 的 ΔSharpe（20000 次，保留相關結構）")
    res = {}
    for label, bench in [('vs 曝險配對靜態傾斜 (B)', tr['tilt']), (f'vs {_config.R5_BENCHMARK} (C, 僅報告)', tr['vt'])]:
        obs, lo, hi, p = paired_sharpe_ci(tr['strat'], bench, rf, rng)
        _, lo2, hi2, _ = paired_sharpe_ci(tr['strat'], bench, rf, rng, n=8000, block=6)
        res[label] = (obs, lo, hi, p)
        print(f"   {label:28s} Δ {obs:+.2f}  95%CI [{lo:+.2f}, {hi:+.2f}]  p={p:.3f}"
              f"  | 區塊(6月) [{lo2:+.2f}, {hi2:+.2f}]")

    # --- D 命中率 ---
    h = tr['hits']
    bt = binomtest(int(h.sum()), len(h), 0.5)
    print(f"\n🎯 D 方向命中率：{h.mean():.1%}（{int(h.sum())}/{len(h)}），二項檢定 p = {bt.pvalue:.3f}")

    # --- E 分期 ---
    print("\n📆 E 分期一致性（擇時報酬的 t 與 vs 傾斜的 ΔSharpe）")
    half = len(idx) // 2
    segs = [('前半', idx[:half]), ('後半', idx[half:]),
            ('壓力年', idx[idx.year.isin(_config.R5_CRISIS_YEARS)]),
            ('非壓力年', idx[~idx.year.isin(_config.R5_CRISIS_YEARS)])]
    seg_rows = {}
    for lbl, sel in segs:
        if len(sel) < 12:
            continue
        d = perf(tr['strat'].reindex(sel), rf)['Sharpe'] - perf(tr['tilt'].reindex(sel), rf)['Sharpe']
        seg_rows[lbl] = {'n': len(sel), 't(擇時)': t_stat(rt.reindex(sel)), 'ΔSharpe vs 傾斜': d}
    seg_tab = pd.DataFrame(seg_rows).T
    print(seg_tab.to_string(formatters={'n': '{:.0f}'.format}, float_format=lambda v: f"{v:+.2f}"))

    # --- G 檢定力 ---
    obsB = res['vs 曝險配對靜態傾斜 (B)'][0]
    frac = obsB / tr['ceiling']
    print(f"\n🔋 G 檢定力：效果 {obsB:+.2f} = 完美預知天花板 {tr['ceiling']:+.2f} 的 {frac:+.0%}"
          f"（門檻 {_config.R5_MIN_DETECTABLE_FRACTION:.0%}）")
    return {'tA': tA, 'B': res['vs 曝險配對靜態傾斜 (B)'], 'C': res[f'vs {_config.R5_BENCHMARK} (C, 僅報告)'],
            'hit_p': bt.pvalue, 'hit': h.mean(), 'seg': seg_tab, 'frac': frac, 'tab': tab, 'w_bar': tr['w_bar']}


# ==========================================
# 4. 繪圖
# ==========================================
def plot_report(tracks, out_path):
    P = PALETTE
    fig, axes = plt.subplots(2, 2, figsize=(16, 9), facecolor=P['surface'])
    for ax in axes.ravel():
        ax.set_facecolor(P['surface'])
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            ax.spines[s].set_color(P['grid'])
        ax.tick_params(colors=P['ink2'])
        ax.grid(True, color=P['grid'], lw=0.8, alpha=0.6)
        ax.set_axisbelow(True)

    for col, (_, tr) in enumerate(tracks.items()):
        name = tr['label_en']
        ax = axes[0, col]
        for (lbl, r), color in zip([('Momentum (all-in)', tr['strat']),
                                    (f"Static tilt w={tr['w_bar']:.2f}", tr['tilt']),
                                    (_config.R5_BENCHMARK, tr['vt'])],
                                   [P['mom'], P['tilt'], P['vt']]):
            ax.plot(r.index, (1 + r).cumprod().values, color=color, lw=2, label=lbl)
        ax.set_yscale('log')
        ax.set_title(f'{name}: growth of 1 (log)', color=P['ink'], loc='left')
        ax.legend(frameon=False, fontsize=9)

        ax = axes[1, col]
        cum = tr['r_timing'].cumsum()
        ax.fill_between(cum.index, cum.values, 0, color=P['band'], lw=0)
        ax.plot(cum.index, cum.values, color=P['mom'], lw=2)
        ax.axhline(0, color=P['ink2'], lw=1)
        ax.set_title(f'{name}: cumulative timing return  (a − w̄) × spread', color=P['ink'], loc='left')
        ax.set_ylabel('cumulative', color=P['ink2'])

    fig.suptitle('Round 5 — does momentum time US vs international?', color=P['ink'], x=0.07, ha='left', fontsize=13)
    fig.subplots_adjust(top=0.90, bottom=0.07, left=0.07, right=0.98, hspace=0.32, wspace=0.18)
    fig.savefig(out_path, dpi=120, facecolor=P['surface'])
    plt.close(fig)


# ==========================================
# 5. 主流程
# ==========================================
if __name__ == "__main__":
    import yfinance as yf
    rng = np.random.default_rng(_config.DIAG_SEED)

    print("=" * 74)
    print("📜 Pre-registration (原文，來自 _config.PRE_REGISTRATION_GEO)")
    print("=" * 74)
    print(_config.PRE_REGISTRATION_GEO)

    try:
        prices = pd.read_csv(_config.R5_DATA_FILE, index_col="Date", parse_dates=True)
    except FileNotFoundError:
        print(f"📥 下載 {_config.R5_TICKERS}（{_config.R5_START_DATE} ~ {_config.END_DATE}）...")
        raw = yf.download(_config.R5_TICKERS, start=_config.R5_START_DATE, end=_config.END_DATE,
                          auto_adjust=False, progress=False)['Adj Close']
        prices = raw[_config.R5_TICKERS].ffill()   # 保留上市前的 NaN
        prices.to_csv(_config.R5_DATA_FILE)
        print(f"💾 已儲存至 {_config.R5_DATA_FILE}")

    monthly_ret, complete = month_end_returns(prices)
    rf = monthly_ret[_config.RF_TICKER]
    if not complete:
        print(f"⚠️ 最後一個月不完整（資料止於 {prices.index[-1].date()}），該月不納入。")

    tracks, results = {}, {}
    for label, intl, ceiling in [(f'主軌 {_config.R5_US}+{_config.R5_INTL_PRIMARY}',
                                  _config.R5_INTL_PRIMARY, _config.R5_CEILING_VEU),
                                 (f'對照軌 {_config.R5_US}+{_config.R5_INTL_CONTROL}',
                                  _config.R5_INTL_CONTROL, _config.R5_CEILING_VXUS)]:
        tr = run_track(prices, monthly_ret, rf, intl, rng, ceiling)
        tracks[label] = tr
        results[label] = report_track(label, tr, rf, rng)

    # --- 判定 ---
    primary = list(results)[0]
    r = results[primary]
    A = r['tA'] > 2
    B = not (r['B'][1] <= 0 <= r['B'][2])
    D = r['hit_p'] < 0.05 and r['hit'] > 0.5
    seg = r['seg']
    E = (np.sign(seg['t(擇時)']).abs().sum() == 0) or (len(set(np.sign(seg['t(擇時)']))) == 1)
    F = np.sign(r['tA']) == np.sign(results[list(results)[1]]['tA'])
    G_under = abs(r['frac']) < _config.R5_MIN_DETECTABLE_FRACTION

    print("\n" + "=" * 74)
    print("⚖️ 依 pre-registration 判定（主軌；由程式計算，不得事後放寬）")
    print("=" * 74)
    print(f"  A 擇時能力   : t = {r['tA']:+.2f} → " + ("成立：有擇時能力" if A else "不成立：無證據"))
    print(f"  B 對曝險配對 : Δ {r['B'][0]:+.2f}, 95%CI [{r['B'][1]:+.2f}, {r['B'][2]:+.2f}] → "
          + ("成立：信賴區間不含 0" if B else "不成立：信賴區間含 0"))
    print(f"  C 對 {_config.R5_BENCHMARK}（僅報告）: Δ {r['C'][0]:+.2f}, 95%CI [{r['C'][1]:+.2f}, {r['C'][2]:+.2f}], p={r['C'][3]:.3f}"
          + "  ← 靜態美股傾斜本身即可跨過此門檻，故不作為通過條件")
    print(f"  D 命中率     : {r['hit']:.1%}, p = {r['hit_p']:.3f} → " + ("成立" if D else "不成立"))
    print(f"  E 分期一致   : 擇時 t 的符號 " + ", ".join(f"{k} {v:+.2f}" for k, v in seg['t(擇時)'].items())
          + (" → 方向一致" if E else " → **符號反轉，依 §3 淘汰**"))
    print(f"  F 雙軌一致   : 主軌 t {r['tA']:+.2f} / 對照軌 t {results[list(results)[1]]['tA']:+.2f} → "
          + ("方向一致" if F else "**方向不一致，淘汰**"))
    print(f"  G 檢定力     : 效果為天花板的 {r['frac']:+.0%} → "
          + (f"**低於 {_config.R5_MIN_DETECTABLE_FRACTION:.0%}，判為「檢定力不足、無法回答」，不得寫成「無效果」**"
             if G_under else "在可偵測範圍內"))

    print("\n" + "=" * 74)
    print("📝 誠實聲明")
    print("=" * 74)
    print("  • w̄ 取自樣本內，使主動權重均值恰為 0；這對擇時檢定是保守的，但確實用到樣本內資訊。")
    print("  • 全押的主動部位約為 ±40%（相對 VT）；資訊比率不受規模影響，但判準 B 的結論只適用全押版本。")
    print("  • K=2 使 Spearman IC 恆為 ±1，改用二項檢定（判準 D），非省略。")
    print("  • 本輪為本專案第五輪，多重比較負擔累積；單一輪的 p 值應據此折價。")
    print("  • 美股在每個可取得窗口都大勝國際股，這是已知洩漏，也是 C 不作為判準的原因。")

    rows = []
    for k, v in results.items():
        rows.append({'track': k, 'w_bar': v['w_bar'], 't_timing': v['tA'],
                     'dSharpe_vs_tilt': v['B'][0], 'ci_lo': v['B'][1], 'ci_hi': v['B'][2], 'p': v['B'][3],
                     'dSharpe_vs_VT': v['C'][0], 'hit_rate': v['hit'], 'hit_p': v['hit_p'],
                     'pct_of_ceiling': v['frac']})
    pd.DataFrame(rows).to_csv("geo_timing_summary.csv", index=False)
    pd.concat({k: v['tab'] for k, v in results.items()}, names=['track']).to_csv("geo_timing_lines.csv")
    plot_report(tracks, "geo_timing_report.png")
    print("\n💾 已儲存: geo_timing_summary.csv, geo_timing_lines.csv, geo_timing_report.png")
