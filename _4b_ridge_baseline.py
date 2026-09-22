"""
線性基準模組：同一組特徵、同一個 target、同一套 walk-forward fold 的 ridge 回歸。
- 舉證責任在複雜度一方：LSTM 要證明比這個線性模型好，才有存在理由。
- 輸入是決策日當天的特徵向量 (沒有序列)，輸出是 13 檔的預測 alpha。
- 正則化強度每個 fold 以 val MSE 在 RIDGE_ALPHAS 裡選，不看 test。
- 不存權重：訓練只需幾毫秒，_5 直接呼叫 ridge_scores() 重算。
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import _config
from _3_ml_data_pipeline import month_end_positions


def fit_ridge_fold(X_scaled, Y, fold, alphas=_config.RIDGE_ALPHAS):
    """在 train 上擬合、以 val MSE 選 alpha。回傳 (model, val_mse, alpha, zero_predictor_val_mse)。"""
    Xtr, Ytr = X_scaled.values[fold['train_pos']], Y.values[fold['train_pos']]
    Xva, Yva = X_scaled.values[fold['val_pos']], Y.values[fold['val_pos']]
    best = None
    for a in alphas:
        model = Ridge(alpha=a).fit(Xtr, Ytr)
        mse = float(((model.predict(Xva) - Yva) ** 2).mean())
        if best is None or mse < best[1]:
            best = (model, mse, a)
    zero_mse = float((Yva ** 2).mean())   # 「永遠預測 0」的 val MSE，作為有沒有訊號的地板
    return (*best, zero_mse)


def ridge_scores(X, Y, folds, asset_columns):
    """
    對每個 fold 擬合 ridge，並在該 fold test 期的每個月底決策日輸出預測 alpha。
    folds: [(name, fold_dict), ...]。回傳 (分數 DataFrame [決策日 x 標的], 各 fold 摘要 DataFrame)。
    """
    pieces, info = [], []
    for name, fold in folds:
        model, val_mse, alpha, zero_mse = fit_ridge_fold(fold['X_scaled'], Y, fold)
        pos = month_end_positions(X, fold['test_pos'])
        pred = model.predict(fold['X_scaled'].values[pos])
        pieces.append(pd.DataFrame(pred, index=X.index[pos], columns=asset_columns))
        info.append({'fold': name, 'alpha': alpha, 'val_mse': val_mse, 'zero_pred_val_mse': zero_mse,
                     'n_train': len(fold['train_pos']), 'n_test_months': len(pos)})
    return pd.concat(pieces), pd.DataFrame(info).set_index('fold')


if __name__ == "__main__":
    from _3_ml_data_pipeline import generate_ml_features, fold_specs, make_fold

    df = pd.read_csv("etf_adj_close_clean.csv", index_col="Date", parse_dates=True)
    X, Y = generate_ml_features(df)
    folds = [(name, make_fold(X, Y, ts, te)) for name, ts, te in fold_specs(X)]
    folds = [(n, f) for n, f in folds if len(f['train_pos']) >= _config.WF_MIN_TRAIN_SAMPLES]
    assets = [c.replace('_Target', '') for c in Y.columns]

    scores, info = ridge_scores(X, Y, folds, assets)
    print("\n📐 Ridge 線性基準 (每個 fold 以 val MSE 選 alpha)：")
    print(info.to_string(float_format=lambda v: f"{v:.6f}"))
    print("   zero_pred_val_mse = 永遠預測 0 的 val MSE。ridge 的 val_mse 沒有明顯低於它，代表線性可辨識的訊號很弱。")
    print(f"\n✅ 產生 {len(scores)} 個月底決策日的預測分數 ({scores.index[0].date()} ~ {scores.index[-1].date()})")
