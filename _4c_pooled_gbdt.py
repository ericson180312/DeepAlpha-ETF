"""
Pooled 橫斷面模型：把資料攤成 (日期, 標的) 的列，讓模型看不到「我是哪一檔」。

為什麼要這樣做（本輪要修的結構性缺陷）：
  現行 LSTM 的輸出層有 13 個 head，等於每檔 ETF 各有自己的參數，所以模型可以記住
  「AVUV 大概長這樣」。逐 fold 的 in-sample IC 0.77~0.90、OOS 約 0，就是這件事的長相。
  Pooled 之後每一列只帶「這檔標的此刻的樣子」，身分被拿掉，模型只能學一條通用規則；
  列數 ×13（但同一日的 13 列高度相關，有效樣本大約只增加 3~5 倍，不是 13 倍）。

表徵（寫定於 _config.PRE_REGISTRATION_POOLED，不得事後增減）：
  (a) 該標的自身：Ret、Vol_20d、Mom_20d、Mom_60d、Mom_120d
  (b) 同一日橫斷面內，(a) 各欄的 z-score
  (c) 共用總經：Global_SPY_Trend_200、Global_SPY_Vol_20d
  不含任何標的身分（無 one-hot、無 embedding）。目標與 LSTM 完全相同，以保可比。

本模組提供 LightGBM 與同表徵的 pooled ridge（baseline-first：非線性要證明自己值得存在）。
"""

import io
import contextlib

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

import _config
from _3_ml_data_pipeline import month_end_positions

ASSET_FEATURES = ['Ret', 'Vol_20d', 'Mom_20d', 'Mom_60d', 'Mom_120d']
MACRO_FEATURES = ['Global_SPY_Trend_200', 'Global_SPY_Vol_20d']


# ==========================================
# 1. 寬表 -> 長表
# ==========================================
def to_panel(X, assets):
    """
    X: 寬表 (日期 × 所有欄位, 欄名為 '{asset}_{feature}' 與全域總經欄)
    回傳 (panel, feature_names)：
      panel 為 MultiIndex (date, asset) 的長表，欄位順序固定 = 自身特徵 + 橫斷面 z-score + 總經。
    """
    frames = []
    for f in ASSET_FEATURES:
        cols = [f'{a}_{f}' for a in assets]
        sub = X[cols].copy()
        sub.columns = assets
        frames.append(sub.stack().rename(f))
    panel = pd.concat(frames, axis=1)
    panel.index.names = ['date', 'asset']

    # 同一日橫斷面的 z-score：模型要比較的是「這檔相對於今天其他檔」，不是絕對水準
    grouped = panel.groupby(level='date')
    z = (panel - grouped.transform('mean')) / (grouped.transform('std') + 1e-8)
    z.columns = [f'{c}_cs_z' for c in z.columns]

    macro = X[MACRO_FEATURES].reindex(panel.index.get_level_values('date'))
    macro.index = panel.index

    out = pd.concat([panel, z, macro], axis=1)
    return out, list(out.columns)


def panel_target(Y, assets):
    """目標攤成同樣的 (date, asset) 長表。欄名 '{asset}_Target' -> 值。"""
    sub = Y[[f'{a}_Target' for a in assets]].copy()
    sub.columns = assets
    y = sub.stack(future_stack=True)
    y.index.names = ['date', 'asset']
    return y


def rows_for(panel, y, X, positions, require_target=True):
    """取出對應某組「日期位置」的長表列（每個位置展開成 len(assets) 列）。"""
    dates = X.index[positions]
    sel = panel.loc[panel.index.get_level_values('date').isin(dates)]
    target = y.reindex(sel.index)
    if require_target:
        ok = target.notna().values
        sel, target = sel[ok], target[ok]
    return sel, target


# ==========================================
# 2. 每個 fold 的訓練
# ==========================================
def fit_fold(panel, y, X, fold, lgb_params=None, early_stopping=None, ridge_alphas=None):
    """
    在一個 fold 上訓練 pooled LightGBM 與 pooled ridge。
    兩者用完全相同的列與特徵；ridge 的 alpha 以 val MSE 選，GBDT 以 val MSE early stopping。
    回傳 (gbdt, ridge, info)。
    """
    lgb_params = dict(_config.POOLED_LGB_PARAMS if lgb_params is None else lgb_params)
    early_stopping = _config.POOLED_EARLY_STOPPING if early_stopping is None else early_stopping
    ridge_alphas = _config.POOLED_RIDGE_ALPHAS if ridge_alphas is None else ridge_alphas

    Xtr, ytr = rows_for(panel, y, X, fold['train_pos'])
    Xva, yva = rows_for(panel, y, X, fold['val_pos'])

    # 標準化沿用 fold 的範圍精神：只用 train 的統計量，不碰 val/test
    mean, std = Xtr.mean(), Xtr.std() + 1e-8
    Xtr_s, Xva_s = (Xtr - mean) / std, (Xva - mean) / std

    gbdt = lgb.LGBMRegressor(**lgb_params)
    gbdt.fit(Xtr_s, ytr, eval_set=[(Xva_s, yva)], eval_metric='l2',
             callbacks=[lgb.early_stopping(early_stopping, verbose=False), lgb.log_evaluation(0)])

    best = None
    for a in ridge_alphas:
        m = Ridge(alpha=a).fit(Xtr_s, ytr)
        mse = float(((m.predict(Xva_s) - yva) ** 2).mean())
        if best is None or mse < best[1]:
            best = (m, mse, a)
    ridge, ridge_val_mse, ridge_alpha = best

    gbdt_val_mse = float(((gbdt.predict(Xva_s) - yva) ** 2).mean())
    info = {'n_train_rows': len(Xtr), 'n_val_rows': len(Xva),
            'gbdt_best_iter': gbdt.best_iteration_, 'gbdt_val_mse': gbdt_val_mse,
            'ridge_alpha': ridge_alpha, 'ridge_val_mse': ridge_val_mse,
            'zero_pred_val_mse': float((yva ** 2).mean())}
    return gbdt, ridge, (mean, std), info


def predict_month_ends(model, panel, X, fold, scaler, assets):
    """對該 fold test 期的每個月底決策日推論，回傳 [決策日 × 標的] 的分數表。"""
    pos = month_end_positions(X, fold['test_pos'])
    if len(pos) == 0:
        return pd.DataFrame(columns=assets)
    rows, _ = rows_for(panel, pd.Series(np.nan, index=panel.index), X, pos, require_target=False)
    mean, std = scaler
    pred = model.predict((rows - mean) / std)
    out = pd.Series(pred, index=rows.index).unstack('asset')
    return out.reindex(columns=assets)


def pooled_scores(X, Y, folds, assets):
    """
    每個 fold 各自訓練，對自己的 test 期推論，接成一條樣本外分數序列。
    回傳 (gbdt_scores, ridge_scores, info_table, importance_table)。
    """
    panel, feature_names = to_panel(X, assets)
    y = panel_target(Y, assets)

    gb_pieces, rg_pieces, infos, imps = [], [], [], {}
    for name, fold in folds:
        gbdt, ridge, scaler, info = fit_fold(panel, y, X, fold)
        gb_pieces.append(predict_month_ends(gbdt, panel, X, fold, scaler, assets))
        rg_pieces.append(predict_month_ends(ridge, panel, X, fold, scaler, assets))
        infos.append({'fold': name, **info})
        imps[name] = pd.Series(gbdt.feature_importances_, index=feature_names)

    return (pd.concat(gb_pieces).sort_index(),
            pd.concat(rg_pieces).sort_index(),
            pd.DataFrame(infos).set_index('fold'),
            pd.DataFrame(imps))


if __name__ == "__main__":
    from _3_ml_data_pipeline import generate_ml_features
    from _5_strategy_backtest import usable_folds

    df = pd.read_csv("etf_adj_close_clean.csv", index_col="Date", parse_dates=True)
    with contextlib.redirect_stdout(io.StringIO()):
        X, Y = generate_ml_features(df)
        folds = usable_folds(X, Y)
    assets = [c.replace('_Target', '') for c in Y.columns]

    panel, feature_names = to_panel(X, assets)
    print(f"📦 Pooled 長表: {panel.shape[0]:,} 列 × {panel.shape[1]} 個特徵 "
          f"(寬表為 {X.shape[0]:,} 列 × {X.shape[1]})")
    print(f"   特徵: {', '.join(feature_names)}")
    print(f"   ⚠️ 同一日的 {len(assets)} 列高度相關，有效樣本增幅遠小於 {len(assets)}×。")

    gb, rg, info, imp = pooled_scores(X, Y, folds, assets)
    print("\n📐 各 fold 的訓練摘要 (zero_pred = 永遠預測 0 的 val MSE，是有無訊號的地板)：")
    print(info.to_string(formatters={'n_train_rows': '{:,.0f}'.format, 'n_val_rows': '{:,.0f}'.format,
                                     'gbdt_best_iter': '{:.0f}'.format, 'ridge_alpha': '{:.0f}'.format},
                         float_format=lambda v: f"{v:.6f}"))
    print(f"\n✅ 產生 {len(gb)} 個月底決策日的分數 ({gb.index[0].date()} ~ {gb.index[-1].date()})")
    print("\n🔍 GBDT 特徵重要度 (各 fold 平均, 前 8 名)：")
    print(imp.mean(axis=1).sort_values(ascending=False).head(8).to_string(float_format=lambda v: f"{v:.0f}"))
