"""
LSTM 深度學習模型模組：定義一個專為金融時間序列預測設計的 LSTM 模型，並實現 walk-forward 的訓練迴圈。
- 模型架構：LSTM 層、層標準化 (Layer Normalization)、過渡全連接層與 GELU、Dropout。
- 損失函數：Huber Loss，delta 設在 target 尺度 (_config.HUBER_DELTA)，極端月份才進入線性區。
- 優化器：Adam + L2 正則化；ReduceLROnPlateau 在驗證損失停滯時降低學習率。
- Walk-forward：每個 fold 各自訓練一組 ensemble (N_MODELS 個 seed)，權重存到 saved_models/<fold>/seed_<s>.pth。
  每個 fold 只看得到自己 test 起點之前的資料，所以 _5 用它對 test 期推論就是真正的樣本外。
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import _config

# ==========================================
# 固定亂數種子 (確保可重現性與成員間的差異化)
# ==========================================
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. 定義 LSTM 深度學習模型
# ==========================================
class MomentumLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=_config.DROPOUT):
        super(MomentumLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]            # 只取最後一個時間步 (站在今天看未來)
        out = self.layer_norm(out)
        out = self.fc1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# ==========================================
# 2. 定義訓練迴圈
# ==========================================
def train_model(model, train_loader, val_loader,
                epochs=_config.EPOCHS, learning_rate=_config.LEARNING_RATE,
                device='cpu', save_path='best_lstm_model.pth', verbose=True):
    """以 val loss 做 early stopping (保留最佳 epoch 的權重)。回傳 history 與最佳 val loss。"""
    model = model.to(device)
    criterion = nn.HuberLoss(delta=_config.HUBER_DELTA)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=_config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss, best_epoch = float('inf'), -1

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_loader.dataset)
        history['train_loss'].append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                val_loss += criterion(model(x_batch), y_batch).item() * x_batch.size(0)
        val_loss /= len(val_loader.dataset)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch + 1
            torch.save(model.state_dict(), save_path)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch+1}/{epochs}] | LR: {current_lr:.5f} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    history['best_val_loss'], history['best_epoch'] = best_val_loss, best_epoch
    return history


def model_dir(fold_name):
    return os.path.join('saved_models', fold_name)


def model_paths(fold_name, n_models=_config.N_MODELS, seed_start=_config.SEED_START):
    """一個 fold 的 ensemble 權重檔路徑清單 (與 _5 共用同一個命名規則)。"""
    return [os.path.join(model_dir(fold_name), f'seed_{seed_start + i}.pth') for i in range(n_models)]


if __name__ == "__main__":
    import pandas as pd
    from _3_ml_data_pipeline import generate_ml_features, fold_specs, make_fold, fold_dataloaders, describe_fold

    df = pd.read_csv("etf_adj_close_clean.csv", index_col="Date", parse_dates=True)
    X, Y = generate_ml_features(df)
    INPUT_SIZE, OUTPUT_SIZE = X.shape[1], Y.shape[1]
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    folds = fold_specs(X)
    print(f"\n{'='*60}\nWalk-forward 訓練: {len(folds)} 個 fold × {_config.N_MODELS} 個 seed (設備: {device})\n{'='*60}")

    total_start = time.time()
    for name, ts, te in folds:
        fold = make_fold(X, Y, ts, te)
        print("\n" + describe_fold(name, fold, X))
        if len(fold['train_pos']) < _config.WF_MIN_TRAIN_SAMPLES:
            print(f"   ⚠️ train 樣本 < {_config.WF_MIN_TRAIN_SAMPLES}，跳過此 fold。")
            continue
        if len(fold['train_pos']) < 300:
            print(f"   ⚠️ train 樣本僅 {len(fold['train_pos'])} 筆 (約 {len(fold['train_pos']) // _config.TARGET_HORIZON} 個獨立月)，結果解讀要保守。")

        os.makedirs(model_dir(name), exist_ok=True)
        train_loader, val_loader = fold_dataloaders(X, Y, fold)

        fold_start = time.time()
        best_losses = []
        for i, path in enumerate(model_paths(name)):
            seed_value = _config.SEED_START + i
            set_all_seeds(seed_value)
            model = MomentumLSTM(INPUT_SIZE, _config.HIDDEN_SIZE, OUTPUT_SIZE)   # 每個成員重新初始化
            hist = train_model(model, train_loader, val_loader, device=device, save_path=path, verbose=False)
            best_losses.append(hist['best_val_loss'])
        print(f"   ✅ {len(best_losses)} 個成員完成，val loss 中位數 {np.median(best_losses):.6f} "
              f"(全距 {min(best_losses):.6f}~{max(best_losses):.6f})，耗時 {time.time() - fold_start:.0f} 秒")

    print(f"\n🎉 全部 fold 訓練完畢，總耗時 {(time.time() - total_start) / 60:.1f} 分鐘。權重在 saved_models/<fold>/")
