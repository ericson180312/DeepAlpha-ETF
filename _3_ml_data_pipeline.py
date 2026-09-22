"""
機器學習資料管線模組：從歷史價格生成特徵矩陣 (X) 與目標矩陣 (Y)，定義 walk-forward 的 fold，
並建立 PyTorch 的 Dataset / DataLoader。

- 特徵與 target 分開處理：X 保留所有特徵齊全的日子，Y 在尾端 TARGET_HORIZON 天為 NaN (尚未實現)。
  這樣推論時可以用到最新一天的特徵，而不是因為 target 缺值把最近 20 天一起丟掉。
- 樣本以「視窗結束日」定義：第 t 天的樣本 = X[t-SEQ+1 : t+1] 與 Y[t]。視窗可以往前跨到別的 segment
  (特徵是回顧性的，不構成洩漏)；洩漏只會經由 target，所以 segment 交界處 purge TARGET_HORIZON 個樣本。
- Walk-forward：每個 fold 用 test 起點之前的資料訓練，val 為 train 尾端固定月數，供 early stopping。
- 標準化只用 val 起點之前的特徵統計量。
"""

import pandas as pd
import numpy as np
import torch
import _config
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 特徵工程模組 (Feature Engineering)
# ==========================================
def generate_ml_features(df, horizon=_config.TARGET_HORIZON, require_all_assets=True):
    """
    回傳 (X, Y)，索引相同。X 為所有特徵齊全的日子；Y 對齊 X，尾端尚未實現的 target 為 NaN。

    require_all_assets=False 為 PIT 模式：只要求全域總經特徵齊全，**保留個別標的的 NaN**
    (那代表該標的當時尚未有足夠歷史)。此時 X 會含 NaN，只有 pooled 表徵能用——
    固定池的 LSTM 假設 13 個輸出恆存在，不適用。
    """
    print("⚙️ 開始進行特徵工程 (總經趨勢 + 個體動能, target = 相對 SPY 的超額報酬)...")

    data = df.copy()
    features = pd.DataFrame(index=data.index)

    # --- 全域總經特徵 (Global Macro Features) ---
    spy_200ma = data['SPY'].rolling(window=200).mean()
    features['Global_SPY_Trend_200'] = (data['SPY'] - spy_200ma) / spy_200ma       # 大盤乖離率
    features['Global_SPY_Vol_20d'] = data['SPY'].pct_change().rolling(20).std()     # 大盤短期波動率

    # --- 各資產個體特徵 (Local Features) ---
    for col in data.columns:
        if col == 'SPY':
            continue
        features[f'{col}_Ret'] = data[col].pct_change()
        features[f'{col}_Vol_20d'] = data[col].pct_change().rolling(20).std()
        features[f'{col}_Mom_20d'] = data[col].pct_change(20)
        features[f'{col}_Mom_60d'] = data[col].pct_change(60)
        features[f'{col}_Mom_120d'] = data[col].pct_change(120)

    # --- 預測目標：未來 horizon 天的超額報酬 (Alpha) ---
    targets = pd.DataFrame(index=data.index)
    spy_future_ret = data['SPY'].pct_change(horizon).shift(-horizon)
    for col in data.columns:
        if col != 'SPY':
            asset_future_ret = data[col].pct_change(horizon).shift(-horizon)
            targets[f'{col}_Target'] = asset_future_ret - spy_future_ret

    if require_all_assets:
        X = features.dropna()
    else:
        # PIT：只要求總經特徵齊全 (SPY 的 200MA 與 20 日波動)
        X = features[features[['Global_SPY_Trend_200', 'Global_SPY_Vol_20d']].notna().all(axis=1)]
    Y = targets.reindex(X.index)

    mode = "固定池" if require_all_assets else "PIT"
    print(f"✅ 特徵工程完成 ({mode})！X: {X.shape} ({X.index[0].date()} ~ {X.index[-1].date()}), "
          f"Y 完整已實現: {int(Y.notna().all(axis=1).sum())} 筆")
    return X, Y


# ==========================================
# 2. Walk-forward fold 定義
# ==========================================
def fold_specs(X, test_starts=None):
    """把設定檔的 test 起點清單展開成 [(name, test_start, test_end), ...]，最後一個 fold 到資料尾端。"""
    starts = [pd.Timestamp(s) for s in (_config.WF_TEST_STARTS if test_starts is None else test_starts)]
    ends = starts[1:] + [X.index[-1] + pd.Timedelta(days=1)]
    return [(f"test_{s.date()}", s, e) for s, e in zip(starts, ends)]


def make_fold(X, Y, test_start, test_end,
              val_months=_config.WF_VAL_MONTHS,
              horizon=_config.TARGET_HORIZON,
              seq_length=_config.SEQ_LENGTH,
              min_targets=None):
    """
    建立一個 fold：回傳 dict，含各 segment 的樣本位置 (X 的整數列索引)、標準化參數與標準化後的 X。

    位置定義 (皆為 X 的列索引)：
      t0 = 第一個 >= test_start 的列, t1 = 第一個 >= test_end 的列, v0 = 第一個 >= (test_start - val_months) 的列
      train: [seq_length-1, v0 - horizon)   且 Y 已實現   ← purge 尾端 horizon 個樣本
      val  : [v0, t0 - horizon)             且 Y 已實現   ← purge 尾端 horizon 個樣本，early stopping 看不到 test 期報酬
      test : [t0, t1)                        (Y 可為 NaN；推論不需要 Y)

    min_targets=None 時要求該日所有標的的 target 都已實現 (固定池)；
    給整數時只要求至少這麼多個標的已實現 (PIT 池：當時還沒上市的標的本來就沒有 target)。
    """
    idx = X.index
    t0 = int(idx.searchsorted(test_start))
    t1 = int(idx.searchsorted(test_end))
    v0 = int(idx.searchsorted(test_start - pd.DateOffset(months=val_months)))
    y_ok = (Y.notna().all(axis=1) if min_targets is None
            else Y.notna().sum(axis=1) >= min_targets).values

    def positions(lo, hi, need_y=True):
        pos = np.arange(max(lo, seq_length - 1), max(hi, 0))
        return pos[y_ok[pos]] if need_y else pos

    train_pos = positions(0, v0 - horizon)
    val_pos = positions(v0, t0 - horizon)
    test_pos = positions(t0, t1, need_y=False)

    # 標準化：只用 val 起點之前的特徵 (不含任何 target 資訊)
    mean = X.iloc[:v0].mean()
    std = X.iloc[:v0].std() + 1e-8
    X_scaled = (X - mean) / std

    return {'test_start': test_start, 'test_end': test_end,
            'train_pos': train_pos, 'val_pos': val_pos, 'test_pos': test_pos,
            'mean': mean, 'std': std, 'X_scaled': X_scaled}


def describe_fold(name, fold, X):
    """一行摘要：各 segment 的樣本數與日期範圍。"""
    def rng(pos):
        return f"{len(pos):4d} 筆 {X.index[pos[0]].date()}~{X.index[pos[-1]].date()}" if len(pos) else "   0 筆"
    return f"{name}: train {rng(fold['train_pos'])} | val {rng(fold['val_pos'])} | test {rng(fold['test_pos'])}"


# ==========================================
# 3. PyTorch 資料集模組 (Custom Dataset)
# ==========================================
class WindowDataset(Dataset):
    """
    以「視窗結束位置」定義的滾動視窗資料集。
    第 i 個樣本 = X_scaled[end_i - seq_length + 1 : end_i + 1] 與 Y[end_i]。
    """
    def __init__(self, X_scaled, Y, end_positions, seq_length=_config.SEQ_LENGTH):
        self.X = torch.tensor(X_scaled.values, dtype=torch.float32)
        self.Y = torch.tensor(Y.values, dtype=torch.float32)
        self.ends = np.asarray(end_positions)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, i):
        end = self.ends[i]
        return self.X[end - self.seq_length + 1: end + 1], self.Y[end]


def fold_dataloaders(X, Y, fold, batch_size=_config.BATCH_SIZE, seq_length=_config.SEQ_LENGTH):
    """為一個 fold 建立 train / val DataLoader (train 可 shuffle，因為每筆樣本內部的時間順序固定)。"""
    Xs = fold['X_scaled']
    train_loader = DataLoader(WindowDataset(Xs, Y, fold['train_pos'], seq_length), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(Xs, Y, fold['val_pos'], seq_length), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def month_end_positions(X, positions):
    """從一組位置中挑出每個日曆月的最後一個交易日 (作為決策日)。"""
    dates = X.index[positions]
    is_last = pd.Series(positions, index=dates).groupby(dates.to_period('M')).last()
    return is_last.values


if __name__ == "__main__":
    df = pd.read_csv("etf_adj_close_clean.csv", index_col="Date", parse_dates=True)
    X, Y = generate_ml_features(df)

    print(f"\n✂️ Walk-forward folds (val {_config.WF_VAL_MONTHS} 個月, purge {_config.TARGET_HORIZON} 個樣本):")
    for name, ts, te in fold_specs(X):
        fold = make_fold(X, Y, ts, te)
        flag = "" if len(fold['train_pos']) >= _config.WF_MIN_TRAIN_SAMPLES else "  ⚠️ train 樣本不足，將跳過"
        print("  " + describe_fold(name, fold, X) + flag)

    # 檢視單一 batch 的形狀
    name, ts, te = fold_specs(X)[-1]
    train_loader, val_loader = fold_dataloaders(X, Y, make_fold(X, Y, ts, te))
    for x_batch, y_batch in train_loader:
        print("\n🔍 檢視單一 Batch 的 Tensor 形狀:")
        print(f"X_batch shape: {tuple(x_batch.shape)} --> (Batch Size, Sequence Length, Features)")
        print(f"Y_batch shape: {tuple(y_batch.shape)} --> (Batch Size, Target ETFs)")
        break
