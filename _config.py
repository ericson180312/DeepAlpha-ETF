# ==========================================
# config.py - 量化專案全域設定檔
# ==========================================

# 1. 資料切分比例 (確保相加為 1.0)
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# 2. 深度學習模型參數
SEQ_LENGTH = 60      # 滾動視窗天數
BATCH_SIZE = 64      # 批次大小
HIDDEN_SIZE = 64     # LSTM 隱藏層大小

# 3. 交易策略參數
TOP_N = 5            # 每月選取前 N 強標的

# 4. 診斷參數 (_6_diagnostics.py)
RF_TICKER = 'BIL'                    # 計算超額 Sharpe 時扣除的無風險利率代理 (須在標的池內)
N_NULL_SIMS = 2000                   # 隨機選股 null 的模擬次數
N_BOOTSTRAP = 2000                   # Sharpe 信賴區間的 bootstrap 次數
COST_GRID_BPS = [0, 5, 10, 20, 50]   # 交易成本敏感度 (每單位交易金額, bps)
DIAG_SEED = 0                        # 診斷用亂數種子 (與模型訓練的 seed 無關)
