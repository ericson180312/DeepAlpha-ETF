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
HUBER_DELTA = 0.05   # Huber 損失的線性區起點。target (20 日超額報酬) 的 σ 約 0.04；原本 1.0 從未觸發、等同 MSE

# 3. 交易策略參數
TOP_N = 5            # 每月選取前 N 強標的
COST_BPS = 10        # 交易成本：每單位成交金額 (bps)。理由見 claude-notes/concepts/trading-costs.md
BASE_MIN_SCORE = 0.0 # 基準動能：分數 <= 0 (絕對動能為負) 的槽位不持有
BASE_FALLBACK = None # 基準動能：被剔除的槽位留作現金
ML_MIN_SCORE = 0.0   # ML：預測 alpha <= 0 的槽位不持有
ML_FALLBACK = 'SPY'  # ML：target 是「相對 SPY 的 alpha」，alpha <= 0 的一致做法是改持 SPY，不是空手

# 4. 診斷參數 (_6_diagnostics.py)
RF_TICKER = 'BIL'                    # 計算超額 Sharpe 時扣除的無風險利率代理 (須在標的池內)
N_NULL_SIMS = 2000                   # 隨機選股 null 的模擬次數
N_BOOTSTRAP = 2000                   # Sharpe 信賴區間的 bootstrap 次數
COST_GRID_BPS = [0, 5, 10, 20, 50]   # 交易成本敏感度 (每單位交易金額, bps)
DIAG_SEED = 0                        # 診斷用亂數種子 (與模型訓練的 seed 無關)
