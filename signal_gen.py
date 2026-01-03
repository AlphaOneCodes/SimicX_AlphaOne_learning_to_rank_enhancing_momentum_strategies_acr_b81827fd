"""Signal generation module using LightGBM LambdaRank for momentum strategy.

Implements learning-to-rank signal generation with volatility-scaled position sizing.
"""

from __future__ import annotations

import os
import subprocess
import shutil
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import numpy as np


def _get_pandas():
    """Lazy import pandas."""
    import pandas
    return pandas


def _get_numpy():
    """Lazy import numpy."""
    import numpy
    return numpy


def _get_lightgbm():
    """Lazy import lightgbm."""
    import lightgbm
    return lightgbm


def _get_engineer_features():
    """Lazy import engineer_features from features module."""
    from features import engineer_features
    return engineer_features


def _get_feature_columns():
    """Lazy import get_feature_columns from features module."""
    from features import get_feature_columns
    return get_feature_columns


def _detect_gpu() -> bool:
    """Detect if GPU is available for LightGBM.
    
    Returns:
        True if GPU is available, False otherwise.
    """
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible == '-1':
        return False
    
    nvidia_smi_path = shutil.which('nvidia-smi')
    if nvidia_smi_path is None:
        return False
    
    result = subprocess.run(
        [nvidia_smi_path], 
        capture_output=True, 
        timeout=5
    )
    return result.returncode == 0


def signal_gen(
    ohlcv_df: 'pd.DataFrame',
    split_date: Optional[str] = None,
    learning_rate: float = 0.05,
    num_boost_round: int = 100,
    max_depth: int = 6,
    reg_alpha: float = 0.1,
    reg_lambda: float = 0.1,
) -> 'pd.DataFrame':
    """Generate trading signals using LightGBM LambdaRank.
    
    Args:
        ohlcv_df: DataFrame with OHLCV data. Expected columns: 
                  ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']
        split_date: Date string for train/predict split. Default '2025-01-01'.
        learning_rate: LightGBM learning rate.
        num_boost_round: Number of boosting rounds.
        max_depth: Maximum tree depth.
        reg_alpha: L1 regularization.
        reg_lambda: L2 regularization.
    
    Returns:
        DataFrame with columns [time, ticker, action, quantity].
    """
    pd = _get_pandas()
    np = _get_numpy()
    lgb = _get_lightgbm()
    engineer_features = _get_engineer_features()
    get_feature_columns = _get_feature_columns()
    
    if split_date is None:
        split_date = '2025-01-01'
    
    df = ohlcv_df.copy()
    if 'date' not in df.columns and 'time' in df.columns:
        df['date'] = df['time']
    df['date'] = pd.to_datetime(df['date'])
    
    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
    
    features_df = engineer_features(df, drop_warmup=True, warmup_days=252)
    
    feature_cols = get_feature_columns()
    
    if isinstance(features_df.index, pd.MultiIndex):
        features_df = features_df.reset_index()
    
    if 'date' not in features_df.columns:
        if 'time' in features_df.columns:
            features_df['date'] = features_df['time']
    
    features_df = features_df.sort_values(['ticker', 'date'])
    
    features_df['next_close'] = features_df.groupby('ticker')['close'].shift(-1)
    features_df['target'] = (features_df['next_close'] - features_df['close']) / features_df['close']
    
    features_df = features_df.dropna(subset=['target'])
    
    split_dt = pd.to_datetime(split_date)
    train_df = features_df[features_df['date'] < split_dt].copy()
    predict_df = features_df[features_df['date'] >= split_dt].copy()
    
    if len(train_df) == 0:
        raise ValueError(f"No training data before split date {split_date}")
    
    if len(predict_df) == 0:
        raise ValueError(f"No prediction data on or after split date {split_date}")
    
    available_features = [col for col in feature_cols if col in features_df.columns]
    if len(available_features) == 0:
        exclude_cols = {'date', 'ticker', 'target', 'next_close', 'open', 'high', 'low', 'close', 'volume', 'time'}
        available_features = [
            col for col in features_df.columns 
            if col not in exclude_cols 
            and features_df[col].dtype in ['float64', 'float32', 'int64', 'int32']
        ]
    
    X_train = train_df[available_features].values
    y_train = train_df['target'].values
    
    X_predict = predict_df[available_features].values
    
    train_groups = train_df.groupby('date').size().values
    
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'reg_alpha': reg_alpha,
        'reg_lambda': reg_lambda,
        'verbosity': -1,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
    }
    
    if _detect_gpu():
        params['device'] = 'gpu'
    
    train_data = lgb.Dataset(
        X_train, 
        label=y_train,
        group=train_groups,
        feature_name=available_features,
    )
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
    )
    
    predict_df = predict_df.copy()
    predict_df['score'] = model.predict(X_predict)
    
    def assign_signals(group: 'pd.DataFrame') -> 'pd.DataFrame':
        """Assign signals based on daily ranking."""
        n = len(group)
        if n < 6:
            group = group.copy()
            group['signal'] = 0
            return group
        
        group = group.copy()
        group['rank'] = group['score'].rank(ascending=False, method='first')
        
        group['signal'] = 0
        group.loc[group['rank'] <= 3, 'signal'] = 1
        group.loc[group['rank'] > n - 3, 'signal'] = -1
        
        return group
    
    predict_df = predict_df.groupby('date', group_keys=False).apply(assign_signals)
    
    target_vol_annual = 0.15
    capital = 1_000_000
    allocated_capital = capital / 6
    
    def calculate_position_size(row: 'pd.Series') -> float:
        """Calculate position quantity with volatility scaling."""
        if row['signal'] == 0:
            return 0.0
        
        price = row['close']
        if price <= 0:
            return 0.0
        
        if 'ex_ante_vol' in row.index and not pd.isna(row['ex_ante_vol']):
            daily_vol = row['ex_ante_vol']
        else:
            daily_vol = 0.20 / np.sqrt(252)
        
        annual_vol = daily_vol * np.sqrt(252)
        
        if annual_vol < 0.01:
            annual_vol = 0.01
        
        quantity = row['signal'] * (allocated_capital / price) * (target_vol_annual / annual_vol)
        
        return quantity
    
    predict_df['quantity'] = predict_df.apply(calculate_position_size, axis=1)
    
    def get_action(signal: int) -> str:
        """Get action string from signal."""
        if signal > 0:
            return 'BUY'
        elif signal < 0:
            return 'SELL'
        else:
            return 'HOLD'
    
    predict_df['action'] = predict_df['signal'].apply(get_action)
    
    output_df = predict_df[['date', 'ticker', 'action', 'quantity']].copy()
    output_df = output_df.rename(columns={'date': 'time'})
    
    output_df = output_df[output_df['quantity'] != 0].copy()
    
    output_df['quantity'] = output_df['quantity'].abs()
    
    output_df['quantity'] = output_df['quantity'].round().astype(int)
    
    output_df = output_df[output_df['quantity'] > 0]
    
    return output_df.reset_index(drop=True)


def simicx_test_gpu_detection():
    """Test GPU detection function."""
    result = _detect_gpu()
    assert isinstance(result, bool), "GPU detection should return boolean"
    print(f"GPU detection result: {result}")


def simicx_test_signal_gen_basic():
    """Test basic signal generation with synthetic data."""
    try:
        pd = _get_pandas()
        np = _get_numpy()
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Skipping simicx_test_signal_gen_basic: {e}")
        return

    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2025-06-01', freq='B')
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM']

    records = []
    for ticker in tickers:
        base_price = np.random.uniform(100, 500)
        prices = [base_price]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0.0005, 0.02)
            prices.append(prices[-1] * (1 + change))

        for i, date in enumerate(dates):
            price = prices[i]
            records.append({
                'date': date,
                'ticker': ticker,
                'open': price * (1 + np.random.uniform(-0.01, 0.01)),
                'high': price * (1 + np.random.uniform(0, 0.02)),
                'low': price * (1 - np.random.uniform(0, 0.02)),
                'close': price,
                'volume': np.random.randint(1000000, 10000000),
            })

    ohlcv_df = pd.DataFrame(records)

    signals = signal_gen(
        ohlcv_df,
        split_date='2025-01-01',
        learning_rate=0.05,
        num_boost_round=50,
        max_depth=4,
    )

    assert isinstance(signals, pd.DataFrame), "Output should be DataFrame"
    assert 'time' in signals.columns, "Should have 'time' column"
    assert 'ticker' in signals.columns, "Should have 'ticker' column"
    assert 'action' in signals.columns, "Should have 'action' column"
    assert 'quantity' in signals.columns, "Should have 'quantity' column"

    valid_actions = {'BUY', 'SELL'}
    assert signals['action'].isin(valid_actions).all(), "Actions should be BUY or SELL"

    assert (signals['quantity'] > 0).all(), "Quantities should be positive"

    print(f"Basic test passed. Generated {len(signals)} signals")
    print(f"  Actions: {signals['action'].value_counts().to_dict()}")


def simicx_test_integration_with_features():
    """Test integration with features module."""
    try:
        pd = _get_pandas()
        np = _get_numpy()
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Skipping simicx_test_integration_with_features: {e}")
        return

    try:
        get_feature_columns = _get_feature_columns()
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Skipping simicx_test_integration_with_features: {e}")
        return

    np.random.seed(123)
    dates = pd.date_range('2023-01-01', '2025-06-01', freq='B')
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA']

    records = []
    for ticker in tickers:
        base_price = np.random.uniform(100, 500)
        prices = [base_price]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0.0003, 0.015)
            prices.append(prices[-1] * (1 + change))

        for i, date in enumerate(dates):
            price = prices[i]
            records.append({
                'date': date,
                'ticker': ticker,
                'open': price * 0.99,
                'high': price * 1.01,
                'low': price * 0.98,
                'close': price,
                'volume': np.random.randint(1000000, 5000000),
            })

    ohlcv_df = pd.DataFrame(records)

    feature_cols = get_feature_columns()

    signals = signal_gen(
        ohlcv_df,
        split_date='2025-01-01',
        learning_rate=0.03,
        num_boost_round=100,
        max_depth=5,
        reg_alpha=0.05,
        reg_lambda=0.05,
    )

    assert len(signals) > 0, "Should generate some signals"
    assert signals['quantity'].max() < 100000, "Quantities seem unreasonably large"

    print(f"Integration test passed. Generated {len(signals)} signals")
    print(f"  Features: {len(feature_cols)} columns")
    print(f"  Quantity range: [{signals['quantity'].min():.2f}, {signals['quantity'].max():.2f}]")


if __name__ == '__main__':
    simicx_test_gpu_detection()
    simicx_test_signal_gen_basic()
    simicx_test_integration_with_features()
    print("\nAll tests passed!")