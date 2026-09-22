# ==========================================
# config.py - 量化專案全域設定檔
# ==========================================
# 原則：程式碼會依賴的「數值」放這裡；「為什麼是這個數字」寫在 claude-notes/concepts/。

# 0. 資料
ETF_POOL = ['SCHG', 'SPMO', 'AVUV', 'SPHQ', 'IDMO', 'AVDV', 'FRDM', 'DBMF', 'PDBC', 'BIL', 'IEF', 'TLT', 'GLD', 'SPY']
START_DATE = '2019-09-26'   # 標的池中最晚上市者 (AVUV/AVDV) 的首個交易日；要更早就得用 proxy 並標記品質
END_DATE = '2026-09-22'     # 不含。釘死而非用 today()，重跑才會得到同一份資料與同一組切分日

# 0b. Point-in-time 標的池（第四輪；與上面的固定池設定並存，不覆蓋，兩者都要可重現）
#   每個決策日只對「當時已有足夠歷史」的標的排名——這才是當年真的面對的選擇集。
#   不使用任何代理：DBMF 的代理 R² 在別的專案量到 0.52，FRDM 的賣點就是與市值加權 EM 不同，
#   兩者代理誤差都會落在故事最敏感的地方。能代理的 AVUV/AVDV 又不是瓶頸，所以整輪不代理。
PIT_DATA_FILE = 'etf_adj_close_pit.csv'
PIT_START_DATE = '2002-07-01'    # 涵蓋 IEF/TLT 上市（2002-07-30）起的所有歷史
PIT_LOOKBACK_DAYS = 252          # 入池門檻：需有此數量的交易日歷史（= max(MOM_WINDOWS)）
PIT_MIN_ASSETS = 6               # 可排名標的少於此數的月份不交易；此值決定起始月 2011-01
TOP_FRACTION = 0.385             # = 5/13，K=13 時剛好還原現行的 Top 5
PIT_MIN_TOP_N = 2                # 每月至少持有幾檔
PIT_WF_TEST_STARTS = ['2013-01-01', '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01',
                      '2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01',
                      '2023-01-01', '2024-01-01', '2025-01-01', '2026-01-01']
PIT_NARROW_MAX_K = 8             # K <= 此值為「窄橫斷面」段，分段報告用
PIT_CRISIS_YEARS = [2011, 2015, 2018, 2020, 2022]   # 壓力年，逐 fold 之外另行標記

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

# 2c. Pooled 橫斷面 GBDT (_4c_pooled_gbdt.py)
#   把資料攤成 (日期, 標的) 的列，模型看不到「我是哪一檔」，只能學通用規則。
#   超參數在本輪開跑前寫定，不做任何搜尋：淺樹 + 重正則化，對應小樣本情境。
POOLED_LGB_PARAMS = {
    'objective': 'huber', 'alpha': 0.05,     # 與 LSTM 的 HUBER_DELTA 同尺度
    'num_leaves': 7,                          # 深度約 3
    'min_child_samples': 200,
    'learning_rate': 0.03,
    'n_estimators': 500,                      # 上限，實際由 early stopping 決定
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7, 'bagging_freq': 1,
    'lambda_l2': 10.0,
    'verbose': -1,
}
POOLED_EARLY_STOPPING = 50                    # val 連續幾輪未改善即停
POOLED_RIDGE_ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]   # pooled 線性基準，每 fold 以 val MSE 選

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


# 5c. Pooled GBDT 的 pre-registration。由 _8_pooled_diagnostics.py 原文印出並機械化判定。
PRE_REGISTRATION_POOLED = """本輪目的：檢驗「把問題改成 pooled 橫斷面」是否讓 ML 產生**優於傳統動能**的樣本外排序能力。
不是找最佳模型，也不是救 LSTM。判準服務於前者；任何導向調參的做法都是本輪的失敗。

背景與本輪要競爭的上限（§5b，開跑前已量定）：
  現行動能 Sharpe 1.06 的拆解為 0.66（多頭部位）+0.17（標的池）+0.19（動能排序）+0.04（現金濾網）。
  **「選股」這個成分只值 +0.19 Sharpe = 0.40 × SE，且逐 fold 變號（+0.13/−0.33/+0.44/−0.06/−0.65）。**
  本輪競爭的就是這 0.19。任何宣稱的改善都必須以 SE 為單位陳述，並與這個上限並列。

表徵（開跑前寫定，不得事後增減）：
  列 = (日期, 標的)。每列的特徵只有三類，**不含任何標的身分**（無 one-hot、無 embedding）：
   (a) 該標的自身：Ret、Vol_20d、Mom_20d、Mom_60d、Mom_120d
   (b) 同一日橫斷面內，(a) 各欄的 z-score
   (c) 共用總經：Global_SPY_Trend_200、Global_SPY_Vol_20d
  目標與 LSTM 完全相同（未來 20 日相對 SPY 的超額報酬），不做橫斷面去均值，以保可比。
  Fold、purge、標準化範圍、決策日、回測會計一律沿用既有設定。

模型：LightGBM，超參數固定於 POOLED_LGB_PARAMS，early stopping 於同一個 val 尾段。
  **本輪不搜尋任何超參數。** 數值是依小樣本情境事前選定（淺樹、重正則化），非調出來的。

判準 A (預測力)：OOS 逐月 rank-IC 均值 > 0 且 t > 2
   → 成立：「pooled GBDT 有樣本外排序能力」；不成立：「無證據」。
判準 B (相對 pooled 線性)：GBDT 與同表徵 pooled ridge 的逐月 IC 差，配對 t > 2
   → 不成立則「非線性不具存在理由」（舉證責任在複雜度一方）。
判準 C (相對傳統動能，本輪真正的門檻)：GBDT 與動能的逐月 IC 差，配對 t > 2
   → 不成立則「ML 沒有勝過一條三行的規則」。動能才是對手，不是已被否決的 LSTM。
判準 D (策略層)：淨成本、扣 rf 的 OOS Sharpe，(GBDT − 動能) ≥ 1 × SE(GBDT)
   且 GBDT 的 Sharpe 在隨機選股 null 的百分位 ≥ 95%。
判準 E (一致性)：逐 fold 與匯總並列；符號不一致的 fold 不得省略。
判準 F (材料性，即使 A~D 全過也適用)：改善量 < 0.5 × SE 一律報成「無實質改善」，
   並與上面 +0.19（0.40 SE）的上限並列。過門檻一絲一毫不算達成目標。

禁止：不得搜尋超參數；不得在看到結果後增減特徵或改表徵；不得改標的池、fold 邊界、成本假設、目標定義；
   不得事後放寬門檻；不得以單一 fold 下結論；
   **若判準不成立，不得以「換一組超參數再試一次」回應**——那正是本專案已列入「不要再試」的行為。
"""


# 5d. Point-in-time 長樣本重測的 pre-registration。由 _9_pit_diagnostics.py 原文印出並機械化判定。
PRE_REGISTRATION_PIT = """本輪目的：**在約 3 倍的樣本上重測前三輪的結論**，不是找新模型、新特徵或新參數。
唯一改變的是樣本（point-in-time 標的池，決策月 2011-01 起，OOS 約 163 個月，SE 由 0.47 降到約 0.27）。
所有模型設定、特徵、目標、成本、purge、會計一律沿用，逐字不動。
任何導向「順便調一下」的做法都是本輪的失敗。

設計（開跑前寫定，看到結果後不得更動）：
  入池規則：標的在擁有 PIT_LOOKBACK_DAYS 個交易日歷史之後才可被排名；SPY 僅作基準與 fallback。
  最小橫斷面 PIT_MIN_ASSETS = 6（決定起始月 2011-01）。不使用任何代理序列。
  每月持股數 = max(PIT_MIN_TOP_N, round(TOP_FRACTION × K_t))，K_t 為當月可排名檔數；K=13 時還原 Top 5。
  參賽者：傳統動能、pooled ridge、pooled GBDT。**LSTM 本輪不參賽**——它的 13 個輸出 head 假設固定池，
  要接受變動池就得改架構，而改架構是本輪明文禁止的事。這是結構性排除，不是因為它上一輪的成績。

橫斷面寬度的可比口徑（必須在量測前寫定，否則跨期 IC 不可比）：
  Spearman IC 在虛無假設下的變異約為 1/(K−1)，K 越窄 IC 越跳。因此：
  - 檢定統計量用寬度標準化後的 z_t = IC_t × sqrt(K_t − 1)（虛無下變異約為 1，與寬度無關），
    對 z 序列做 t 檢定。
  - 效果量用以 (K_t − 1) 為權重的加權平均 IC。
  - 未加權平均 IC 一併報出供對照；兩者相左時以加權/標準化者為準，並把矛盾寫進報告。

判準 A (動能存續)：動能的 z 序列 t > 2 且加權平均 IC > 0
   → 成立：「動能的排序能力在長樣本、多 regime 下站得住」；
   → 不成立：**前三輪「動能是唯一有效訊號」的結論作廢**，須重新檢討。
判準 B (動能策略層)：動能的淨成本、扣 rf 的 OOS Sharpe 在隨機選股 null 的百分位 >= 95%
   （null 必須使用同一套 PIT 池與逐月持股數，否則不可比）。
判準 C (ML 勝出)：各 ML 模型與動能的逐月 IC 差，配對 t > 2 才算勝出；
   且策略層 ΔSharpe >= 1 × SE。兩者皆成立才報「勝出」。
判準 D (regime 一致性)：逐 fold 全部並列；另單獨報 PIT_CRISIS_YEARS 子集。
   匯總與逐 fold 不一致時兩者都要寫，不得只報匯總。
判準 E (窄/寬分段)：K <= PIT_NARROW_MAX_K 與 K > PIT_NARROW_MAX_K 兩段各報一次。
   兩段方向不一致時，結論即為「不一致」，不得以匯總掩蓋。
判準 F (材料性)：任何差異 < 0.5 × SE 一律報成「無實質差異」，並以 SE 為單位陳述。

必須隨結果一起報的混淆揭露：
  長樣本的早期橫斷面較窄（2011 年只有 6 檔：IEF/TLT/GLD/SPHQ/BIL/SCHG，沒有商品、國際、小型價值、管理期貨），
  **資產構成與後期不同**。因此結論若與前三輪不同，可能來自樣本量，也可能來自池組成，本輪無法分離這兩者。
  緩解措施（強制）：同時報「2022-02 起、13 檔全真實」子期作為對照軌，即前三輪的口徑。

禁止：更動任何模型超參數、特徵、目標定義、成本假設、purge 長度、fold 邊界；
   看到結果後調整 PIT_MIN_ASSETS、TOP_FRACTION、PIT_LOOKBACK_DAYS 或危機年清單；
   加入任何代理序列；把 LSTM 改成可變池後納入；事後放寬門檻；以單一 fold 或單一分段下結論。
"""
