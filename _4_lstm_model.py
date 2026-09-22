"""
LSTM 深度學習模型模組：定義一個專為金融時間序列預測設計的 LSTM 模型，並實現訓練迴圈 (Training Loop) 以優化模型參數。   
- 模型架構：包含 LSTM 層、層標準化 (Layer Normalization)、過渡全連接層與激勵函數，以及 Dropout 層以防止過度擬合。
- 損失函數：使用 Huber Loss 以提高對金融資料中極端離群值的魯棒性。
- 優化器：Adam 優化器，並加入 L2 正則化以進一步防止過度擬合。
- 學習率調度器：當驗證損失停滯時自動降低學習率，幫助模型更好地收斂。
- 整合 Ensemble Learning：透過迴圈與不同的 Random Seed，一次訓練並儲存多個模型供後續投資委員會推論使用。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import _config

# ==========================================
# 🌟 新增：固定亂數種子函式 (確保可重現性與差異化)
# ==========================================
def set_all_seeds(seed=42):
    """固定所有的隨機性，確保給定 seed 時模型具備可重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # 強制 cuDNN 使用確定性演算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. 定義 LSTM 深度學習模型 (V2 優化版)
# ==========================================
class MomentumLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.3):
        super(MomentumLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 1. LSTM 層 (處理時間序列)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # 2. 層標準化 (Layer Normalization) 穩定金融資料的分佈漂移
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 3. 過渡全連接層與激勵函數 (幫助總經與個體特徵進行非線性交叉融合)
        # 將維度從 hidden_size 壓縮到一半
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.activation = nn.GELU() # GELU 在現代深度學習中通常比傳統 ReLU 表現更好
        
        # 4. 防止過度擬合的 Dropout 層
        self.dropout = nn.Dropout(dropout)
        
        # 5. 最終輸出層
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        # 只取最後一個時間步 (站在今天看未來)
        out = out[:, -1, :] 
        
        # 網路拓樸流程
        out = self.layer_norm(out) # 穩定分佈
        out = self.fc1(out)        # 特徵交叉
        out = self.activation(out) # 非線性映射
        out = self.dropout(out)    # 隨機失活防過擬合
        out = self.fc2(out)        # 輸出預測 Alpha
        
        return out

# ==========================================
# 2. 定義訓練迴圈 (🌟 新增 save_path 參數)
# ==========================================
def train_model(model, train_loader, val_loader, epochs=60, learning_rate=0.001, device='cpu', save_path='best_lstm_model.pth'):
    print(f"🚀 開始訓練 LSTM 模型 (設備: {device})，預計存至: {save_path} ...")
    model = model.to(device)
    
    # 使用 Huber Loss 對極端離群值更具魯棒性
    criterion = nn.HuberLoss(delta=_config.HUBER_DELTA)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4) 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # --- 訓練階段 ---
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # --- 驗證階段 ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        # 更新 Learning Rate
        scheduler.step(val_loss)
        
        # --- 儲存最佳模型 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # 🌟 使用動態傳入的檔名儲存模型
            torch.save(model.state_dict(), save_path)
            is_best = "⭐"
        else:
            is_best = ""
            
        # 每 10 個 Epoch 印出一次進度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] | LR: {current_lr:.5f} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} {is_best}")
            
    print(f"✅ 訓練完成！最佳驗證損失 (Val Loss): {best_val_loss:.6f}")
    return history

if __name__ == "__main__":
    from _3_ml_data_pipeline import generate_ml_features, prepare_dataloaders
    import pandas as pd
    
    # 1. 讀取與準備資料
    df = pd.read_csv("etf_adj_close_clean.csv", index_col="Date", parse_dates=True)
    X, Y = generate_ml_features(df)
    train_loader, val_loader, test_loader, _, _ = prepare_dataloaders(
        X, Y, 
        seq_length=_config.SEQ_LENGTH, 
        batch_size=_config.BATCH_SIZE
    )
    
    INPUT_SIZE = X.shape[1]   
    OUTPUT_SIZE = Y.shape[1]  
    
    # 從設定檔讀取隱藏層大小 (假設 config 內有，若無請直接改為 64)
    try:
        HIDDEN_SIZE = _config.HIDDEN_SIZE
    except AttributeError:
        HIDDEN_SIZE = 64
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # ==========================================
    # 🌟 核心：訓練 Ensemble (集成) 的 20 個模型
    # ==========================================
    n_models = 20
    print(f"\n{'='*50}\n啟動 Ensemble Learning: 準備訓練 {n_models} 個模型\n{'='*50}")
    
    for i in range(n_models):
        # 產生不同的 Seed: 42~61
        seed_value = 42 + i 
        
        # 🌟 1. 固定本次迴圈的亂數種子
        set_all_seeds(seed_value)
        
        print(f"\n--- 正在訓練第 {i+1}/{n_models} 個模型 (Seed: {seed_value}) ---")
        
        # 🌟 2. 每次都要重新初始化一個乾淨的模型，避免權重污染
        model = MomentumLSTM(
            input_size=INPUT_SIZE, 
            hidden_size=HIDDEN_SIZE, 
            output_size=OUTPUT_SIZE, 
            num_layers=1, 
            dropout=0.3
        )
        
        # 🌟 3. 動態設定該模型的存檔名稱
        save_file = f'saved_models/ensemble_lstm_seed_{seed_value}.pth'
        
        # 4. 開始訓練
        train_model(
            model, 
            train_loader, 
            val_loader, 
            epochs=60, 
            learning_rate=0.002, 
            device=device, 
            save_path=save_file
        )
        
    print(f"\n🎉 {n_models} 個 Ensemble 模型已全部訓練完畢！權重已分別儲存。")