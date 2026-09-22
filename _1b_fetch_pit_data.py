"""
Point-in-time 標的池的資料抓取（第四輪）。

與 _1_fetch_and_clean_data.py 的差別，只有一個但很關鍵：
  _1 在開頭有缺值時直接報錯，因為固定池的設計要求所有標的自始存在。
  本模組**保留開頭的缺值**——那正是「該標的當時還沒上市」的事實，是 PIT 設計要用的資訊。
  ffill 只往後填，不會跨越首筆有效日，所以不會製造任何上市前的假價格。

輸出到 _config.PIT_DATA_FILE，與固定池的 CSV 並存，兩輪都要可重現。
"""

import pandas as pd
import yfinance as yf

import _config


def fetch_pit_data(tickers, start_date, end_date):
    print(f"📥 下載 PIT 標的池: {tickers}")
    print(f"📅 期間: {start_date} 至 {end_date} (end 為不含)")
    try:
        raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
    except Exception as e:
        print(f"❌ 下載失敗: {e}")
        return None

    adj = raw['Adj Close'] if isinstance(raw.columns, pd.MultiIndex) else raw[['Adj Close']]
    missing = [t for t in tickers if t not in adj.columns]
    if missing:
        print(f"❌ 下載結果缺少標的: {missing}")
        return None
    adj = adj[tickers]

    # 只做前向填補：不跨越首筆有效日，所以上市前仍為 NaN
    clean = adj.ffill()

    first = clean.apply(lambda s: s.first_valid_index())
    print("\n📊 各標的首筆有效日（NaN 保留，代表當時尚未上市）:")
    for t, d in first.sort_values().items():
        print(f"   {t:<5} {d.date()}")
    return clean


def eligibility(df, lookback=None, exclude=('SPY',)):
    """
    入池矩陣：標的在擁有 lookback 個交易日歷史之後才可被排名。
    回傳 (布林矩陣 [日期 × 可排名標的], 每日可排名檔數 K)。
    """
    lookback = _config.PIT_LOOKBACK_DAYS if lookback is None else lookback
    cols = [c for c in df.columns if c not in exclude]
    # 累計至今的有效觀測數 >= lookback
    counts = df[cols].notna().cumsum()
    elig = counts >= lookback
    return elig, elig.sum(axis=1)


def apply_pit_eligibility(frame, elig, assets, suffixes):
    """
    把入池遮罩套到寬表上：標的尚未入池的日子，其所有相關欄位設為 NaN。

    ⚠️ 必須對特徵、目標、動能分數**一致地**套用。否則各模型看到的池子不同：
    例如 Mom_120d 只需 120 天歷史，會比需要 252 天的動能分數早 132 天「看見」某檔標的，
    三條線就不在同一個 universe 上比較了。
    """
    out = frame.copy()
    for a in assets:
        cols = [f'{a}{suf}' for suf in suffixes if f'{a}{suf}' in out.columns]
        if not cols:
            continue
        mask = elig[a].reindex(out.index).ffill().fillna(False).astype(bool)
        out.loc[~mask, cols] = float('nan')
    return out


if __name__ == "__main__":
    df = fetch_pit_data(_config.ETF_POOL, _config.PIT_START_DATE, _config.END_DATE)
    if df is None:
        exit(1)

    elig, K = eligibility(df)
    monthly_elig = elig.resample('ME').last()      # 以每月最後一個交易日的狀態為準
    monthly_K = monthly_elig.sum(axis=1)
    tradable = monthly_K[monthly_K >= _config.PIT_MIN_ASSETS]

    print(f"\n⚙️ 入池門檻: {_config.PIT_LOOKBACK_DAYS} 個交易日歷史；最小橫斷面 {_config.PIT_MIN_ASSETS} 檔")
    print(f"   可交易決策月: {len(tradable)} 個 ({tradable.index[0]:%Y-%m} ~ {tradable.index[-1]:%Y-%m})")
    print("\n   橫斷面寬度隨時間變化:")
    for k, grp in tradable.groupby(tradable):
        row = monthly_elig.loc[grp.index[-1]]
        names = sorted(row[row].index)
        print(f"     K={k:2d}: {grp.index[0]:%Y-%m} ~ {grp.index[-1]:%Y-%m} ({len(grp):3d} 個月)  {' '.join(names)}")

    df.to_csv(_config.PIT_DATA_FILE)
    print(f"\n💾 已儲存至 {_config.PIT_DATA_FILE} ({len(df)} 個交易日, {df.notna().sum().sum():,} 個有效觀測)")
