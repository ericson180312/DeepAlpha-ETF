"""
特徵工程與 walk-forward 切分的單元測試。

重點在兩件無聲的錯誤：
  (1) 特徵用到未來資料（look-ahead）
  (2) purge 不足，使 early stopping 的驗證集看到與測試期重疊的報酬（leakage）
兩者都不會報錯，只會讓樣本外績效變好看。
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _config
from _3_ml_data_pipeline import (generate_ml_features, fold_specs, make_fold,
                                 month_end_positions, WindowDataset)


@pytest.fixture(scope='module')
def synthetic_df():
    """三年的合成日價格，足以讓 200 日均線與 252 日動能成立。"""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range('2020-01-01', periods=800)
    cols = ['AAA', 'BBB', 'CCC', 'SPY']
    steps = rng.normal(0.0004, 0.01, size=(len(idx), len(cols)))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx, columns=cols)


@pytest.fixture(scope='module')
def features(synthetic_df):
    return generate_ml_features(synthetic_df)


# ==========================================
# 特徵 / target
# ==========================================
def test_features_have_no_nan_and_targets_tail_is_nan(features):
    X, Y = features
    assert X.notna().all().all()
    # 尾端 horizon 天的 target 尚未實現
    assert Y.tail(_config.TARGET_HORIZON).isna().all().all()
    assert Y.iloc[:-_config.TARGET_HORIZON].notna().all().all()


def test_features_extend_to_the_last_price_day(synthetic_df, features):
    """
    舊版對 concat(features, targets) 做 dropna，連最後 20 天的「特徵」都被丟掉，
    於是永遠算不出最近一個月的訊號，實盤無法使用。
    """
    X, _ = features
    assert X.index[-1] == synthetic_df.index[-1]


def test_target_is_excess_return_over_spy(synthetic_df, features):
    X, Y = features
    h = _config.TARGET_HORIZON
    d = X.index[300]
    i = synthetic_df.index.get_loc(d)
    fut = synthetic_df.iloc[i + h] / synthetic_df.iloc[i] - 1
    expected = fut['AAA'] - fut['SPY']
    assert Y.loc[d, 'AAA_Target'] == pytest.approx(expected)


def test_features_do_not_use_future_prices(synthetic_df):
    """
    把某個日期之後的價格全部改掉，該日期之前的特徵必須一模一樣。
    這是 look-ahead 最直接的檢驗方式。
    """
    cut = synthetic_df.index[500]
    tampered = synthetic_df.copy()
    tampered.loc[tampered.index > cut] *= 3.0

    X_orig, _ = generate_ml_features(synthetic_df)
    X_tamp, _ = generate_ml_features(tampered)
    common = X_orig.index[X_orig.index <= cut]
    pd.testing.assert_frame_equal(X_orig.loc[common], X_tamp.loc[common])


def test_targets_do_use_future_prices(synthetic_df):
    """對照組：target 本來就該是前瞻的，否則上面那個測試沒有鑑別力。"""
    cut = synthetic_df.index[500]
    tampered = synthetic_df.copy()
    tampered.loc[tampered.index > cut] *= 3.0
    _, Y_orig = generate_ml_features(synthetic_df)
    _, Y_tamp = generate_ml_features(tampered)
    before = Y_orig.index[Y_orig.index <= cut][-1]
    assert not np.isclose(Y_orig.loc[before, 'AAA_Target'], Y_tamp.loc[before, 'AAA_Target'])


# ==========================================
# Walk-forward 切分
# ==========================================
@pytest.fixture(scope='module')
def fold(features):
    X, Y = features
    test_start = X.index[int(len(X) * 0.75)]
    test_end = X.index[-1] + pd.Timedelta(days=1)
    return X, Y, make_fold(X, Y, test_start, test_end, val_months=3)


def test_segments_are_ordered_and_disjoint(fold):
    _, _, f = fold
    assert len(f['train_pos']) and len(f['val_pos']) and len(f['test_pos'])
    assert f['train_pos'].max() < f['val_pos'].min() < f['test_pos'].min()
    assert not set(f['train_pos']) & set(f['val_pos'])
    assert not set(f['val_pos']) & set(f['test_pos'])


def test_purge_gap_between_segments(fold):
    """
    target 橫跨 TARGET_HORIZON 天，所以相鄰 segment 之間至少要空出這麼多樣本，
    否則 train/val 的 target 會包含下一段的已實現報酬。
    """
    _, _, f = fold
    h = _config.TARGET_HORIZON
    assert f['val_pos'].min() - f['train_pos'].max() >= h
    assert f['test_pos'].min() - f['val_pos'].max() >= h


def test_val_targets_do_not_reach_into_test_period(fold):
    """把 purge 講成日期的版本：val 最後一筆 target 的結束日必須早於 test 起點。"""
    X, _, f = fold
    h = _config.TARGET_HORIZON
    last_val_target_end = f['val_pos'].max() + h
    assert X.index[last_val_target_end] < f['test_start']


def test_windows_never_start_before_the_data(fold):
    _, _, f = fold
    for key in ('train_pos', 'val_pos', 'test_pos'):
        assert f[key].min() >= _config.SEQ_LENGTH - 1


def test_train_and_val_samples_have_realised_targets(fold):
    _, Y, f = fold
    y_ok = Y.notna().all(axis=1).values
    assert y_ok[f['train_pos']].all()
    assert y_ok[f['val_pos']].all()


def test_scaler_uses_only_pre_val_data(fold):
    """標準化的均值/標準差只能由 val 起點之前的特徵算出，否則測試期的分佈資訊會外洩。"""
    X, Y, f = fold
    v0 = int(X.index.searchsorted(f['test_start'] - pd.DateOffset(months=3)))
    pd.testing.assert_series_equal(f['mean'], X.iloc[:v0].mean())
    pd.testing.assert_series_equal(f['std'], X.iloc[:v0].std() + 1e-8)


def test_folds_have_disjoint_test_periods(features):
    X, Y = features
    specs = fold_specs(X, test_starts=['2021-07-01', '2022-01-01'])
    assert specs[0][2] == pd.Timestamp('2022-01-01')        # 前一折的終點 = 下一折的起點
    positions = [set(make_fold(X, Y, ts, te)['test_pos']) for _, ts, te in specs]
    assert not positions[0] & positions[1]


# ==========================================
# Dataset 與決策日
# ==========================================
def test_window_dataset_alignment(fold):
    X, Y, f = fold
    ds = WindowDataset(f['X_scaled'], Y, f['train_pos'], seq_length=_config.SEQ_LENGTH)
    x, y = ds[0]
    end = f['train_pos'][0]
    assert x.shape == (_config.SEQ_LENGTH, X.shape[1])
    # 視窗最後一列 = 結束位置當天的特徵；標的 = 同一天的 target
    assert x[-1].numpy() == pytest.approx(f['X_scaled'].values[end], rel=1e-5)
    assert y.numpy() == pytest.approx(Y.values[end], rel=1e-5)


def test_month_end_positions_picks_last_trading_day(features):
    X, _ = features
    pos = month_end_positions(X, np.arange(len(X)))
    dates = X.index[pos]
    assert dates.is_unique
    assert len(dates.to_period('M').unique()) == len(dates)      # 每月剛好一個
    for d in dates[:-1]:
        later_same_month = X.index[(X.index > d) & (X.index.to_period('M') == d.to_period('M'))]
        assert len(later_same_month) == 0                        # 而且是該月最後一個
