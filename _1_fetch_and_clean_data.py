"""
下載標的池的每日收盤價，並進行資料清洗以供後續特徵工程與模型訓練使用。

- 使用 yfinance API 下載資料，並提取「已調整收盤價」(Adjusted Close) 以考慮除權息與股票分割的影響。
- 時間區間固定在 _config.START_DATE ~ END_DATE，不用「今天往回推」：每次執行才會得到同一份資料、同一組切分日。
  (yfinance 的 Adj Close 仍會隨每次配息回頭修正歷史，重跑時小數位可能不同——這是資料源的限制，不是程式的。)
- 清洗只做前向填補 (Forward Fill)。不做後向填補：START_DATE 已設在標的池中最晚上市者之後，
  若仍出現開頭的 NaN，代表 universe 或起始日設錯，直接報錯而不是用假資料補平。
- 最終輸出為一個 CSV 檔案，供後續模組使用。
"""

import yfinance as yf
import pandas as pd
import _config

def fetch_and_clean_data(tickers, start_date, end_date):
    """
    下載指定標的池的歷史 K 線資料，並進行清洗與提取已調整收盤價。
    """
    print(f"📥 開始從 yfinance 下載資料: {tickers}")
    print(f"📅 期間: {start_date} 至 {end_date} (end 為不含)")

    # 設定 auto_adjust=False 確保 API 會回傳明確的 'Adj Close' 欄位
    try:
         df_raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
    except Exception as e:
         print(f"❌ 下載失敗: {e}")
         return None

    print("⚙️ 提取已調整收盤價 (Adj Close)...")
    if isinstance(df_raw.columns, pd.MultiIndex):
        df_adj_close = df_raw['Adj Close']
    else:
        df_adj_close = df_raw[['Adj Close']].rename(columns={'Adj Close': tickers[0]})

    missing_cols = [t for t in tickers if t not in df_adj_close.columns]
    if missing_cols:
        print(f"❌ 下載結果缺少標的: {missing_cols}")
        return None
    df_adj_close = df_adj_close[tickers]

    # --- 資料清洗 ---
    print("🧹 執行資料清洗 (處理 NaN 缺失值)...")

    # 開頭的 NaN = 該標的在 START_DATE 時尚未上市。這不能用 bfill 補：
    # 補出來的是一段平的假價格，動能為 0、特徵全 0，會靜默污染訓練。
    first_valid = df_adj_close.apply(lambda s: s.first_valid_index())
    late = first_valid[first_valid > df_adj_close.index[0]]
    if not late.empty:
        print("❌ 下列標的在 START_DATE 時尚未有資料，請調整 _config.START_DATE 或 ETF_POOL：")
        for t, d in late.items():
            print(f"   {t}: 首筆有效日 {d.date()}")
        return None

    # 前向填補：某檔 ETF 當天無交易紀錄 (單一市場休市或資料遺失) 時沿用前一交易日收盤價。
    # 只用過去資料，不引入未來函數。
    df_clean = df_adj_close.ffill()

    missing_data = df_clean.isna().sum().sum()
    if missing_data > 0:
        print(f"⚠️ 警告：清洗後資料中仍有 {missing_data} 個缺失值！")
    else:
        print("✅ 資料清洗完成，無任何缺失值。")

    return df_clean

if __name__ == "__main__":
    cleaned_df = fetch_and_clean_data(_config.ETF_POOL, _config.START_DATE, _config.END_DATE)

    if cleaned_df is None:
        exit(1)

    print("\n📊 清洗後資料預覽 (前 5 筆):")
    print(cleaned_df.head())
    print("\n📊 清洗後資料預覽 (最後 5 筆):")
    print(cleaned_df.tail())

    output_filename = "etf_adj_close_clean.csv"
    cleaned_df.to_csv(output_filename)
    print(f"\n💾 資料已成功儲存至當前目錄: {output_filename} ({len(cleaned_df)} 個交易日)")
