"""
機器學習資料管線模組：從原始歷史價格資料中生成特徵矩陣 (X) 和目標矩陣 (Y)，並建立 PyTorch 的 Dataset 和 DataLoader 以供模型訓練使用。
"""

import pandas as pd
import numpy as np
import torch
import _config
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 特徵工程模組 (Feature Engineering)
# ==========================================
def generate_ml_features(df):
    """
    V2 優化版：加入大盤總經特徵，並將預測目標改為「超額報酬 (Alpha)」
    """
    print("⚙️ 開始進行優化版特徵工程 (加入總經趨勢與 Alpha 目標)...")
    
    data = df.copy()
    features = pd.DataFrame(index=data.index)
    
    # ==========================================
    # 🌟 新增：全域總經特徵 (Global Macro Features)
    # ==========================================
    # 1. 計算大盤 (SPY) 的 200 日移動平均線 (牛熊分界線)
    spy_200ma = data['SPY'].rolling(window=200).mean()
    
    # 2. 大盤乖離率：(現在價格 - 200MA) / 200MA。大於 0 代表多頭，小於 0 代表空頭
    features['Global_SPY_Trend_200'] = (data['SPY'] - spy_200ma) / spy_200ma
    
    # 3. 大盤短期波動率：當大盤開始劇烈震盪，通常是反轉或大跌的前兆
    features['Global_SPY_Vol_20d'] = data['SPY'].pct_change().rolling(20).std()
    
    # ==========================================
    # 既有：各資產個體特徵 (Local Features)
    # ==========================================
    for col in data.columns:
        if col == 'SPY': 
            continue # 大盤已經作為全域特徵，不再納入個體比較
            
        features[f'{col}_Ret'] = data[col].pct_change()
        features[f'{col}_Vol_20d'] = data[col].pct_change().rolling(20).std()
        features[f'{col}_Mom_20d'] = data[col].pct_change(20)
        features[f'{col}_Mom_60d'] = data[col].pct_change(60)

        features[f'{col}_Mom_120d'] = data[col].pct_change(120)
        # features[f'{col}_Mom_240d'] = data[col].pct_change(240)
        
    # ==========================================
    # 🌟 關鍵修改：預測目標 (Target) 改為「超額報酬 (Excess Return)」
    # ==========================================
    targets = pd.DataFrame(index=data.index)
    
    # 計算大盤未來 20 天的真實報酬
    spy_future_ret = data['SPY'].pct_change(20).shift(-20)
    
    for col in data.columns:
        if col != 'SPY':
            # 該資產未來 20 天的真實報酬
            asset_future_ret = data[col].pct_change(20).shift(-20)
            
            # 目標 Y = 該資產報酬 - 大盤報酬 (算出 Alpha)
            # 這樣模型就會被迫去尋找「能跑贏大盤」的資產，而不是「不會虧錢」的資產
            targets[f'{col}_Target'] = asset_future_ret - spy_future_ret
            
    # 清除 NaN
    combined = pd.concat([features, targets], axis=1).dropna()
    
    X = combined[features.columns]
    Y = combined[targets.columns]
    
    print(f"✅ 特徵工程 V2 完成！特徵矩陣大小: {X.shape}, 目標矩陣大小: {Y.shape}")
    return X, Y

# ==========================================
# 2. PyTorch 資料集模組 (Custom Dataset)
# ==========================================
class TimeSeriesDataset(Dataset):
    """
    自定義 PyTorch Dataset，處理時間序列的滾動視窗 (Sliding Window)
    """
    def __init__(self, X, Y, seq_length=60):
        """
        X: 特徵矩陣 (numpy array)
        Y: 目標矩陣 (numpy array)
        seq_length: 模型回溯的天數 (Lookback window)
        """
        self.X = torch.tensor(X.values, dtype=torch.float32)
        self.Y = torch.tensor(Y.values, dtype=torch.float32)
        self.seq_length = seq_length

    def __len__(self):
        # 總共能切出多少個完整的視窗
        return len(self.X) - self.seq_length

    def __getitem__(self, idx):
        # 取得從 idx 開始，長度為 seq_length 的一段歷史特徵
        x_window = self.X[idx : idx + self.seq_length]
        # 取得這段歷史「最後一天」對應的未來預測目標
        y_target = self.Y[idx + self.seq_length - 1]
        
        return x_window, y_target

# ==========================================
# 3. 資料切分與標準化模組
# ==========================================
def prepare_dataloaders(X, Y, 
                        seq_length=_config.SEQ_LENGTH, 
                        batch_size=_config.BATCH_SIZE, 
                        train_ratio=_config.TRAIN_RATIO, 
                        val_ratio=_config.VAL_RATIO):
    """
    嚴格按時間軸切分 Train/Val/Test，並建立 DataLoader
    """
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    print(f"✂️ 依時間軸切分資料 (Train: {train_ratio*100}%, Val: {val_ratio*100}%, Test: {(1-train_ratio-val_ratio)*100:.1f}%)")
    
    # ⚠️ 時間序列絕對不能用 sklearn 的 train_test_split (預設會打亂順序)
    X_train, Y_train = X.iloc[:train_end], Y.iloc[:train_end]
    X_val, Y_val = X.iloc[train_end:val_end], Y.iloc[train_end:val_end]
    X_test, Y_test = X.iloc[val_end:], Y.iloc[val_end:]
    
    # --- 特徵標準化 (Z-Score Normalization) ---
    # 深度學習對數值範圍很敏感，必須標準化
    # ⚠️ 關鍵：只能用 Train set 的均值與標準差來縮放 Val 和 Test，否則會資料外洩
    train_mean = X_train.mean()
    train_std = X_train.std() + 1e-8 # 加上微小值避免除以零
    
    X_train_scaled = (X_train - train_mean) / train_std
    X_val_scaled = (X_val - train_mean) / train_std
    X_test_scaled = (X_test - train_mean) / train_std
    
    # --- 建立 PyTorch Dataset ---
    train_dataset = TimeSeriesDataset(X_train_scaled, Y_train, seq_length)
    val_dataset = TimeSeriesDataset(X_val_scaled, Y_val, seq_length)
    test_dataset = TimeSeriesDataset(X_test_scaled, Y_test, seq_length)
    
    # --- 建立 DataLoader ---
    # train_loader 可以 shuffle，因為每筆資料 (x_window, y_target) 的內部時間順序是固定的
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print("📦 PyTorch DataLoaders 準備就緒！")
    return train_loader, val_loader, test_loader, train_mean, train_std

if __name__ == "__main__":
    # 1. 讀取乾淨的歷史資料
    df = pd.read_csv("etf_adj_close_clean.csv", index_col="Date", parse_dates=True)
    
    # 2. 執行特徵工程
    X, Y = generate_ml_features(df)
    
    # 3. 建立 PyTorch 資料管線
    # 假設回看過去 60 個交易日 (約一季) 來預測未來 20 個交易日 (約一個月)
    train_loader, val_loader, test_loader, scaler_mean, scaler_std = prepare_dataloaders(
        X, Y, seq_length=_config.SEQ_LENGTH, batch_size=_config.BATCH_SIZE
    )
    
    # 4. 測試取出一個 Batch 看看形狀
    for x_batch, y_batch in train_loader:
        print("\n🔍 檢視單一 Batch 的 Tensor 形狀:")
        print(f"X_batch shape: {x_batch.shape} --> (Batch Size, Sequence Length, Features)")
        print(f"Y_batch shape: {y_batch.shape} --> (Batch Size, Target ETFs)")
        break # 只印出第一組