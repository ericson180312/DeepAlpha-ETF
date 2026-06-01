import subprocess
import sys
import time
import os

def run_pipeline():
    # 定義要依序執行的腳本清單
    # 這裡確保按照資料抓取 -> 基準回測 -> 特徵管線(測試) -> 模型訓練 -> AI回測 的順序
    scripts = [
        "_1_fetch_and_clean_data.py",
        "_2_baseline_performance.py",
        "_3_ml_data_pipeline.py",     # 測試特徵工程是否正常運作
        "_4_lstm_model.py",           # 執行 Ensemble Learning 模型訓練
        "_5_strategy_backtest.py"     # 進行終極回測與繪圖
    ]

    print("🚀 開始執行量化交易全自動化管線 (Auto-Trading Pipeline)...")
    print("=" * 60)

    total_start_time = time.time()

    for script in scripts:
        # 檢查檔案是否存在
        if not os.path.exists(script):
            print(f"\n❌ 找不到檔案: {script}，請確認腳本名稱是否正確或是否在同一目錄下。")
            sys.exit(1)

        print(f"\n▶️ [{time.strftime('%H:%M:%S')}] 正在啟動: {script} ...")
        start_time = time.time()
        
        try:
            # sys.executable 會自動抓取你目前執行這個腳本所使用的 Python 環境 (例如你的 conda 虛擬環境)
            # check=True 代表如果該腳本發生 Error (Return code 不為 0)，就會拋出 Exception
            subprocess.run([sys.executable, script], check=True)
            
            elapsed_time = time.time() - start_time
            print(f"✅ {script} 執行完成！ (耗時: {elapsed_time:.2f} 秒)")
            
        except subprocess.CalledProcessError as e:
            # 如果任何一個腳本中斷或報錯，整個 Pipeline 就會安全停止，避免產生錯誤的後續回測
            print(f"\n❌ 執行 {script} 時發生錯誤！自動化管線已強制中斷。")
            print(f"錯誤代碼 (Return Code): {e.returncode}")
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n⚠️ 收到手動中斷指令 (Ctrl+C)！正在停止 {script} ...")
            sys.exit(1)

    total_elapsed_time = time.time() - total_start_time
    print("\n" + "=" * 60)
    print(f"🎉 專案全線執行完畢！總耗時: {total_elapsed_time / 60:.2f} 分鐘")
    print("📊 請檢查資料夾內生成的圖表 (baseline_vs_spy_performance.png, ml_vs_baseline_performance.png) 與 模型權重檔 (*.pth)。")
    print("=" * 60)

if __name__ == "__main__":
    run_pipeline()