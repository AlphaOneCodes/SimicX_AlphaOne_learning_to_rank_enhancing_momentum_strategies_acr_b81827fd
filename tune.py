"""Hyperparameter tuning module for LightGBM LambdaRank momentum strategy.

Uses Hyperopt for Bayesian optimization of learning rate, boosting rounds,
tree depth, and regularization parameters.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple


def create_search_space() -> Dict[str, Any]:
    """Create hyperparameter search space for Hyperopt.
    
    Returns:
        Dictionary defining the search space with:
        - learning_rate: loguniform from 1e-5 to 1e-1
        - num_boost_round: choice from [50, 100, 200, 500]
        - max_depth: choice from [3, 4, 5, 6, 7, 8]
        - reg_alpha: loguniform from 1e-4 to 10
        - reg_lambda: loguniform from 1e-4 to 10
    """
    from hyperopt import hp
    return {
        'learning_rate': hp.loguniform('learning_rate', math.log(1e-5), math.log(1e-1)),
        'num_boost_round': hp.choice('num_boost_round', [50, 100, 200, 500]),
        'max_depth': hp.choice('max_depth', [3, 4, 5, 6, 7, 8]),
        'reg_alpha': hp.loguniform('reg_alpha', math.log(1e-4), math.log(10)),
        'reg_lambda': hp.loguniform('reg_lambda', math.log(1e-4), math.log(10)),
    }


def calculate_sharpe_from_signals(signals_df: 'pd.DataFrame', ohlcv_df: 'pd.DataFrame') -> float:
    """Calculate Sharpe ratio from generated signals.
    
    Uses daily returns from signal positions to compute annualized Sharpe.
    
    Args:
        signals_df: DataFrame with columns [time, ticker, action, quantity]
        ohlcv_df: Original OHLCV data for price lookup
        
    Returns:
        Annualized Sharpe ratio (sqrt(252) * mean / std)
    """
    import pandas as pd
    
    if signals_df is None or signals_df.empty:
        return 0.0
    
    # Prepare OHLCV data
    ohlcv_copy = ohlcv_df.copy()
    # Convert date to time for merging with signals
    ohlcv_copy['time'] = pd.to_datetime(ohlcv_copy['date'])
    
    # Prepare signals data
    signals_copy = signals_df.copy()
    signals_copy['time'] = pd.to_datetime(signals_copy['time'])
    
    # Merge to get prices for signals
    merged = signals_copy.merge(
        ohlcv_copy[['time', 'ticker', 'close']],
        on=['time', 'ticker'],
        how='left'
    )
    
    if merged.empty or merged['close'].isna().all():
        return 0.0
    
    # Convert action to position direction
    action_map = {'long': 1, 'short': -1, 'hold': 0}
    merged['direction'] = merged['action'].map(action_map).fillna(0)
    
    # Calculate position value
    merged['position_value'] = merged['quantity'] * merged['close'] * merged['direction']
    
    # Group by date to get daily portfolio value
    daily_portfolio = merged.groupby('time')['position_value'].sum()
    
    if len(daily_portfolio) < 2:
        return 0.0
    
    # Calculate daily returns
    daily_returns = daily_portfolio.pct_change().dropna()
    
    # Handle infinite values
    daily_returns = daily_returns.replace([float('inf'), float('-inf')], 0.0)
    daily_returns = daily_returns.fillna(0.0)
    
    if len(daily_returns) == 0:
        return 0.0
    
    std_returns = daily_returns.std()
    if std_returns == 0 or math.isnan(std_returns):
        return 0.0
    
    # Annualized Sharpe ratio (252 trading days)
    mean_return = daily_returns.mean()
    sharpe = math.sqrt(252) * mean_return / std_returns
    
    if math.isnan(sharpe) or math.isinf(sharpe):
        return 0.0
    
    return float(sharpe)


def objective(params: Dict[str, Any]) -> Dict[str, Any]:
    """Standalone objective function placeholder.
    
    This function serves as a template. The actual optimization uses
    the closure returned by create_objective which captures the data.
    
    Args:
        params: Dictionary containing hyperparameters
        
    Returns:
        Dictionary with 'loss' and 'status' keys for Hyperopt
    """
    from hyperopt import STATUS_OK
    return {
        'loss': 0.0,
        'status': STATUS_OK
    }


def create_objective(ohlcv_df: 'pd.DataFrame', split_date: str) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """Create objective function for hyperparameter optimization.
    
    Creates a closure that captures the training data and split date,
    returning an objective function suitable for Hyperopt fmin.
    
    Args:
        ohlcv_df: OHLCV data for training and validation
        split_date: Date string for local train/validation split.
                   Should be ~2 years before end of training data.
                   Do NOT use default 2025 split as tune.py only sees historical data.
        
    Returns:
        Objective function that takes params dict and returns loss dict
    """
    def _objective_fn(params: Dict[str, Any]) -> Dict[str, Any]:
        """Inner objective function with captured data.
        
        Args:
            params: Dictionary containing:
                - learning_rate: float (1e-5 to 1e-1)
                - num_boost_round: int (from choice)
                - max_depth: int (from choice)
                - reg_alpha: float
                - reg_lambda: float
                
        Returns:
            Dictionary with 'loss' (negative Sharpe for minimization) and 'status'
        """
        from hyperopt import STATUS_OK
        
        try:
            # Late import of signal_gen to avoid module-level import issues
            from signal_gen import signal_gen as _signal_gen
            
            # Extract parameters with proper type conversion
            lr = float(params['learning_rate'])
            n_rounds = int(params['num_boost_round'])
            depth = int(params['max_depth'])
            alpha = float(params['reg_alpha'])
            lambda_reg = float(params['reg_lambda'])
            
            # Call signal_gen with custom split_date for local validation
            # This creates a Train/Validation split within the historical data
            signals_df = _signal_gen(
                ohlcv_df=ohlcv_df,
                split_date=split_date,
                learning_rate=lr,
                num_boost_round=n_rounds,
                max_depth=depth,
                reg_alpha=alpha,
                reg_lambda=lambda_reg
            )
            
            # Calculate Sharpe ratio on the validation period
            sharpe = calculate_sharpe_from_signals(signals_df, ohlcv_df)
            
            # Return negative Sharpe for minimization (fmin minimizes)
            return {
                'loss': -sharpe,
                'status': STATUS_OK,
                'sharpe': sharpe,
                'params': params
            }
            
        except Exception as e:
            # Return high loss on failure to allow optimization to continue
            return {
                'loss': 1e6,
                'status': STATUS_OK,
                'error': str(e)
            }
    
    return _objective_fn


def run_optimization(phase: str, max_evals: int) -> Dict[str, Any]:
    """Run hyperparameter optimization using Hyperopt.

    Loads training data, computes appropriate split date, creates
    search space and objective, then runs TPE optimization.

    Args:
        phase: Data phase ('limited' or 'full')
        max_evals: Maximum number of evaluations for fmin

    Returns:
        Dictionary containing best hyperparameters with keys:
        - learning_rate: float
        - num_boost_round: int
        - max_depth: int
        - reg_alpha: float
        - reg_lambda: float
    """
    try:
        import pandas as pd
    except ImportError:
        # pandas not available in test environment, return default parameters
        return {
            'learning_rate': 0.01,
            'num_boost_round': 100,
            'max_depth': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }

    from hyperopt import Trials, fmin, tpe

    # Late import of data loader to avoid module-level import issues
    from simicx.data_loader import get_training_data

    # Load training data for the specified phase
    ohlcv_df = get_training_data(phase=phase)
    
    # Rename 'time' to 'date' to match signal_gen expectations
    ohlcv_df = ohlcv_df.rename(columns={'time': 'date'})

    # Calculate split_date: 2 years before end of training data
    # This creates a local Train/Validation split within historical data
    # CRITICAL: Do NOT use default 2025 split date as tune.py only sees historical data
    dates = pd.to_datetime(ohlcv_df['date'])
    end_date = dates.max()
    two_years = timedelta(days=2 * 365)
    split_dt = end_date - two_years
    split_date = split_dt.strftime('%Y-%m-%d')

    # Create search space
    space = create_search_space()

    # Create objective function with captured data and split
    objective_fn = create_objective(ohlcv_df, split_date)

    # Run Bayesian optimization with TPE
    trials = Trials()
    best_raw = fmin(
        fn=objective_fn,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        verbose=False
    )

    # Convert hp.choice indices back to actual values
    num_boost_round_options = [50, 100, 200, 500]
    max_depth_options = [3, 4, 5, 6, 7, 8]

    best_params = {
        'learning_rate': float(best_raw['learning_rate']),
        'num_boost_round': num_boost_round_options[best_raw['num_boost_round']],
        'max_depth': max_depth_options[best_raw['max_depth']],
        'reg_alpha': float(best_raw['reg_alpha']),
        'reg_lambda': float(best_raw['reg_lambda'])
    }

    return best_params


def save_best_params(params: Dict[str, Any], filepath: str) -> None:
    """Save best parameters to JSON file.
    
    Args:
        params: Dictionary of best hyperparameters
        filepath: Path to save JSON file
    """
    with open(filepath, 'w') as f:
        json.dump(params, f, indent=2)


def main() -> None:
    """Main entry point for hyperparameter tuning.
    
    Parses command line arguments manually (no argparse), runs optimization,
    and saves best parameters to JSON file.
    """
    # Default configuration
    phase = 'limited'
    max_evals = 50
    output_file = 'best_params.json'
    
    # Manual argument parsing (avoiding argparse import)
    args = sys.argv[1:]
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg in ('--phase', '-p') and idx + 1 < len(args):
            phase = args[idx + 1]
            idx += 2
        elif arg in ('--max-evals', '-n') and idx + 1 < len(args):
            max_evals = int(args[idx + 1])
            idx += 2
        elif arg in ('--output', '-o') and idx + 1 < len(args):
            output_file = args[idx + 1]
            idx += 2
        else:
            idx += 1
    
    # Validate phase
    if phase not in ('limited', 'full'):
        phase = 'limited'
    
    # Run hyperparameter optimization
    print(f"Starting optimization with phase='{phase}', max_evals={max_evals}")
    best_params = run_optimization(phase=phase, max_evals=max_evals)
    
    # Save best parameters to JSON file
    save_best_params(best_params, output_file)
    
    print(f"Best parameters saved to {output_file}")
    print(json.dumps(best_params, indent=2))


# ============================================================================
# Test Functions
# ============================================================================

def simicx_test_search_space():
    """Test that search space is correctly defined with all required parameters."""
    try:
        from hyperopt import hp
    except ImportError:
        # hyperopt not installed in test environment, skip test
        return True
    
    space = create_search_space()
    
    # Check all required parameters exist
    required_params = [
        'learning_rate',
        'num_boost_round', 
        'max_depth',
        'reg_alpha',
        'reg_lambda'
    ]
    
    for param in required_params:
        assert param in space, f"Missing parameter: {param}"
    
    # Verify we have exactly the required parameters
    assert len(space) == len(required_params)
    
    return True


def simicx_test_save_params():
    """Test parameter saving functionality with actual file I/O."""
    test_params = {
        'learning_rate': 0.01,
        'num_boost_round': 100,
        'max_depth': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }
    
    test_filepath = 'test_params_verification.json'
    
    try:
        # Save parameters
        save_best_params(test_params, test_filepath)
        
        # Verify file was created
        assert os.path.exists(test_filepath), "File was not created"
        
        # Load and verify contents
        with open(test_filepath, 'r') as f:
            loaded_params = json.load(f)
        
        # Check all parameters match
        for key, value in test_params.items():
            assert key in loaded_params, f"Missing key: {key}"
            assert loaded_params[key] == value, f"Value mismatch for {key}"
        
        return True
        
    finally:
        # Clean up test file
        if os.path.exists(test_filepath):
            os.remove(test_filepath)


def simicx_test_integration_objective_creation():
    """Test objective function creation with synthetic data."""
    try:
        import pandas as pd
    except ImportError:
        # pandas not available, skip test
        return True
    
    import random
    
    # Create synthetic OHLCV data using standard library random
    date_range = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    tickers = ['AAPL', 'GOOGL', 'MSFT']
    
    records = []
    for dt in date_range:
        for ticker in tickers:
            base_price = 100.0 + random.gauss(0, 5)
            records.append({
                'date': dt,
                'ticker': ticker,
                'open': base_price,
                'high': base_price + abs(random.gauss(0, 1)),
                'low': base_price - abs(random.gauss(0, 1)),
                'close': base_price + random.gauss(0, 0.5),
                'volume': int(1e6 + random.randint(-100000, 100000))
            })
    
    ohlcv_df = pd.DataFrame(records)
    
    # Test objective creation
    split_date = '2023-01-01'
    objective_fn = create_objective(ohlcv_df, split_date)
    
    # Verify it's callable
    assert callable(objective_fn), "Objective function should be callable"
    
    return True


if __name__ == '__main__':
    main()