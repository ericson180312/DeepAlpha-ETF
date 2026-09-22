# ==========================================
# config.py - 量化專案全域設定檔
# ==========================================
# 原則：程式碼會依賴的「數值」放這裡；「為什麼是這個數字」寫在 claude-notes/concepts/。

# 0. 資料
ETF_POOL = ['SCHG', 'SPMO', 'AVUV', 'SPHQ', 'IDMO', 'AVDV', 'FRDM', 'DBMF', 'PDBC', 'BIL', 'IEF', 'TLT', 'GLD', 'SPY']
START_DATE = '2019-09-26'   # 標的池中最晚上市者 (AVUV/AVDV) 的首個交易日；要更早就得用 proxy 並標記品質
END_DATE = '2026-09-22'     # 不含。釘死而非用 today()，重跑才會得到同一份資料與同一組切分日

# 1. 樣本切分：walk-forward (expanding window)
#   每個 fold 以 WF_TEST_STARTS[i] 為 test 起點、下一個起點為終點 (最後一個 fold 到資料尾端)；
#   起點之前所有已實現 target 的樣本為 train+val，val 取最後 WF_VAL_MONTHS 個月供 early stopping。
#   train/val 與 val/test 交界各 purge TARGET_HORIZON 個樣本，避免重疊的 target 把下一段的報酬洩漏給前一段。
WF_TEST_STARTS = ['2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01', '2026-01-01']
WF_VAL_MONTHS = 6
WF_MIN_TRAIN_SAMPLES = 100  # train 樣本少於此數的 fold 跳過並警告
TARGET_HORIZON = 20         # 預測目標 = 未來 20 個交易日相對 SPY 的超額報酬；同時是 purge 長度

# 2. 深度學習模型參數
SEQ_LENGTH = 60      # 滾動視窗天數
BATCH_SIZE = 64      # 批次大小
HIDDEN_SIZE = 64     # LSTM 隱藏層大小
DROPOUT = 0.3
EPOCHS = 60
LEARNING_RATE = 0.002
WEIGHT_DECAY = 1e-4
HUBER_DELTA = 0.05   # Huber 損失的線性區起點。target 的 σ 約 0.04；原本 1.0 從未觸發、等同 MSE
N_MODELS = 20        # ensemble 成員數
SEED_START = 42      # 成員 seed = SEED_START + i

# 2b. 線性基準 (同一組特徵的 ridge 回歸；LSTM 要證明比它好才有存在理由)
RIDGE_ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]   # 每個 fold 以 val MSE 選一個 (網格上界要夠大，避免卡在邊緣)

# 3. 交易策略參數
MOM_WINDOWS = [63, 126, 252]   # 動能分數的回看窗口 (交易日)；是否坐在高原上由 _7_momentum_robustness.py 檢驗
TOP_N = 5            # 每月選取前 N 強標的
COST_BPS = 10        # 交易成本：每單位成交金額 (bps)。理由見 claude-notes/concepts/trading-costs.md
RF_TICKER = 'BIL'    # 計算超額 Sharpe 時扣除的無風險利率代理 (須在標的池內)
BASE_MIN_SCORE = 0.0 # 基準動能：分數 <= 0 (絕對動能為負) 的槽位不持有
BASE_FALLBACK = None # 基準動能：被剔除的槽位留作現金
ML_MIN_SCORE = 0.0   # ML / ridge：預測 alpha <= 0 的槽位不持有
ML_FALLBACK = 'SPY'  # ML / ridge：target 是「相對 SPY 的 alpha」，alpha <= 0 的一致做法是改持 SPY，不是空手

# 4. 診斷參數 (_6_diagnostics.py)
N_NULL_SIMS = 2000                   # 隨機選股 null 的模擬次數
N_BOOTSTRAP = 2000                   # Sharpe 信賴區間的 bootstrap 次數
COST_GRID_BPS = [0, 5, 10, 20, 50]   # 交易成本敏感度 (每單位交易金額, bps)
DIAG_SEED = 0                        # 診斷用亂數種子 (與模型訓練的 seed 無關)

# 4b. 動能穩健性檢驗 (_7_momentum_robustness.py)
#   網格：把基準窗口三元組 (MOM_WINDOWS) 整體縮放 k 倍 × TOP_N。目的是檢驗基準值是否坐在高原上。
MOM_SCALE_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
MOM_TOPN_GRID = [3, 4, 5, 6, 7]

# 5. Pre-registration：判定規則在跑出任何結果之前寫定，由 _6_diagnostics.py 機械化計算並原文印在報告開頭。
#    改動這段文字會進 git diff。
PRE_REGISTRATION = """\
本輪目的：驗證 Ensemble LSTM 在 walk-forward 樣本外「有沒有」排序能力，不是找出最佳模型。
所有設計服務於前者；任何導向後者的做法 (調 fold 邊界、超參數、標的池) 都是本輪的失敗。

判準 A (預測力)：OOS 全期逐月 rank-IC 的均值 > 0 且 t 值 > 2
   → 成立：「LSTM 有樣本外排序能力」；不成立：「無證據顯示有排序能力」。
判準 B (相對線性基準)：LSTM 與同特徵 ridge 的逐月 IC 差，配對 t 值 > 2
   → 成立：「LSTM 優於線性」；不成立：「LSTM 不具存在理由」(舉證責任在複雜度一方)。
判準 C (策略層)：淨成本、扣 rf 後的 OOS Sharpe，(LSTM − 傳統動能) ≥ 1 × SE(LSTM)，
   且 LSTM 的 Sharpe 在隨機選股 null 的百分位 ≥ 95%
   → 兩者皆成立才報「勝出」；否則報「與 baseline / 隨機分不開」。
判準 D (一致性)：逐 fold 結論與匯總不一致時兩者並列，不得省略不一致的 fold。
禁止：本輪不調整 SEQ_LENGTH / HIDDEN_SIZE / EPOCHS / LR / fold 邊界；不得因結果增減標的；不得以單一 fold 下結論。
"""

# 5b. 動能穩健性的 pre-registration。由 _7_momentum_robustness.py 原文印出並機械化判定。
PRE_REGISTRATION_MOMENTUM = """本輪目的：驗證基準動能的 (63,126,252) × Top 5 是否坐在高原上，
**不是**找出最佳窗口或最佳 N。所有設計服務於前者；任何導向後者的做法都是本輪的失敗。
背景：上一輪判定 LSTM 無樣本外排序能力後，動能成為唯一支撐結論的訊號，而它本身從未被檢驗過。

網格：MOM_SCALE_GRID (6) × MOM_TOPN_GRID (5) = 30 格，同一段 OOS 月份、同一套會計 (淨成本、扣 rf)。
材料性單位一律用 SE：本輪的 SE(Sharpe) 約 0.5，因此「差距小於 0.5 × SE」判為無實質差異。

判準 A (高原)：基準格的 Sharpe 落在網格分佈的 25~75 百分位之間
   → 判為高原，**維持 (63,126,252) × Top 5**。不得因為某一格數字較高就改用它。
判準 B (尖峰否決)：基準格的 Sharpe 減去其四個相鄰格 (scale ±1 階、TOP_N ±1 階) 的平均，
   若 > 1 × SE → 判為過擬合產物，**上一輪「動能是唯一有效訊號」的結論作廢**，須重新檢討。
   「贏太多」和「輸」一樣是壞消息。
判準 C (劣位)：基準格落在後 25% → 判為劣位；**此時且僅此時**才討論替代，
   且替代者必須是「平原中心」(其鄰域平均最高且鄰域全距最小者)，不是 argmax。
判準 D (逐期間一致性)：逐 fold 重複判準 A。匯總與逐 fold 不一致時兩者並列，不得只報匯總。
判準 E (量化陷阱交叉檢查)：百分位是對 30 格的離散分佈算的，解析度僅 1/30。
   因此同時報「未量化」讀數：(基準格 Sharpe − 網格中位數) ÷ SE。
   兩者相左時**以未量化者為準**，並把矛盾本身寫進報告。

誠實聲明 (必須隨結果一起報)：30 格共用同一段 55 個月，彼此高度相關，
   網格的離散程度**低估**真實不確定性。因此本輪只能回答「基準值在網格內是否異常」，
   不能回答「動能的 Sharpe 有多可靠」——後者由上一輪的 SE 與 null 百分位回答。
禁止：不得輸出「建議改用 X」除非判準 C 觸發；不得改動 LSTM/ridge 任何設定；
   不得改標的池、fold 邊界、成本假設；不得事後放寬門檻；不得以單一 fold 或單一軸下結論。
"""
