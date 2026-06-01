"""
下載各ETF過去6年內的每日收盤價，並進行資料清洗以供後續特徵工程與模型訓練使用。

- 使用 yfinance API 下載資料，並提取「已調整收盤價」(Adjusted Close) 以考慮除權息與股票分割的影響。
- 資料清洗步驟包括前向填補 (Forward Fill) 和後向填補 (Backward Fill)，確保資料完整且適合矩陣運算。
- 最終輸出為一個 CSV 檔案，供後續模組使用。
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_and_clean_data(tickers, start_date, end_date):
    """
    下載指定標的池的歷史 K 線資料，並進行清洗與提取已調整收盤價。
    """
    print(f"📥 開始從 yfinance 下載資料: {tickers}")
    print(f"📅 期間: {start_date} 至 {end_date}")
    
    # 使用 yf.download 一次性下載所有標的
    # 設定 auto_adjust=False 確保 API 會回傳明確的 'Adj Close' 欄位
    try:
         df_raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False)
    except Exception as e:
         print(f"❌ 下載失敗: {e}")
         return None

    # 提取「已調整收盤價」(Adjusted Close)
    # 這是量化回測的標準，因為它已經將除權息與股票分割的影響計算在內
    print("⚙️ 提取已調整收盤價 (Adj Close)...")
    if isinstance(df_raw.columns, pd.MultiIndex):
        # 當下載多檔股票時，yfinance 會回傳 MultiIndex 的 DataFrame
        df_adj_close = df_raw['Adj Close']
    else:
        # 防呆機制：如果標的池只有一檔股票的處理方式
        df_adj_close = df_raw[['Adj Close']].rename(columns={'Adj Close': tickers[0]})

    # --- 實作資料清洗邏輯 ---
    print("🧹 執行資料清洗 (處理 NaN 缺失值)...")
    
    # 步驟 1：前向填補 (Forward Fill)
    # 邏輯：如果某檔 ETF 當天沒有交易紀錄（例如單一市場休市或資料遺失），則沿用前一個交易日的收盤價。
    # 這是處理金融時間序列最標準且不引入「未來函數」的做法。
    df_clean = df_adj_close.ffill()
    
    # 步驟 2：後向填補 (Backward Fill)
    # 邏輯：有些 ETF（例如 FRDM 成立時間較晚），在回測期間的前段會全是 NaN。
    # 為了維持 DataFrame 結構完整以利矩陣運算，我們將最早的有效價格往回填補。
    # (註：在後續實作策略邏輯時，需注意過濾掉這種被往回填補的平移假資料)
    df_clean = df_clean.bfill()

    # 最終檢查
    missing_data = df_clean.isna().sum().sum()
    if missing_data > 0:
        print(f"⚠️ 警告：清洗後資料中仍有 {missing_data} 個缺失值！")
    else:
        print("✅ 資料清洗完成，無任何缺失值。")

    return df_clean

if __name__ == "__main__":
    # 1. 定義標的池 (Universe)
    ETF_POOL = ['SCHG', 'SPMO', 'AVUV', 'SPHQ', 'IDMO', 'AVDV', 'FRDM', 'DBMF', 'PDBC', 'BIL', 'IEF', 'TLT', 'GLD', 'SPY']
    
    # 2. 設定時間區間
    # 這裡設定抓取過去 6 年的資料作為範例
    END_DATE = datetime.today().strftime('%Y-%m-%d')
    START_DATE = (datetime.today() - timedelta(days=6*365)).strftime('%Y-%m-%d')
    
    # 3. 執行資料管線模組
    cleaned_df = fetch_and_clean_data(ETF_POOL, START_DATE, END_DATE)
    
    if cleaned_df is not None:
        # 4. 檢視成果
        print("\n📊 清洗後資料預覽 (前 5 筆):")
        print(cleaned_df.head())
        print("\n📊 清洗後資料預覽 (最後 5 筆):")
        print(cleaned_df.tail())
        
        # 5. 輸出存檔 (供後續特徵工程與 PyTorch 模型讀取)
        output_filename = "etf_adj_close_clean.csv"
        cleaned_df.to_csv(output_filename)
        print(f"\n💾 資料已成功儲存至當前目錄: {output_filename}")