import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from typing import Dict, Any, Tuple, List
from strategies import get_all_strategies, BaseStrategy
from backtest import run_backtest

class StrategyOptimizer:
    def __init__(self, strategy: BaseStrategy, param_space: Dict[str, Any], metric: str = 'sharpe_ratio'):
        self.strategy = strategy
        self.param_space = param_space
        self.metric = metric
        self.best_params = None
        self.best_score = None

    def _objective(self, params, param_names, df):
        # Set parameters
        for name, value in zip(param_names, params):
            setattr(self.strategy, name, value)
        # Run backtest
        result = run_backtest(df)
        # Score
        if self.metric == 'sharpe_ratio':
            score = -result.sharpe_ratio  # minimize negative Sharpe
        elif self.metric == 'total_return':
            score = -result.total_return
        elif self.metric == 'max_drawdown':
            score = result.max_drawdown
        else:
            score = -result.total_return
        return score

    def optimize(self, df: pd.DataFrame, n_calls: int = 30) -> Tuple[Dict[str, Any], float]:
        param_names = list(self.param_space.keys())
        dimensions = [self.param_space[name] for name in param_names]
        
        def obj(params):
            return self._objective(params, param_names, df)
        
        res = gp_minimize(obj, dimensions, n_calls=n_calls, random_state=42)
        self.best_params = dict(zip(param_names, res.x))
        self.best_score = -res.fun
        # Set best params to strategy
        for name, value in self.best_params.items():
            setattr(self.strategy, name, value)
        return self.best_params, self.best_score

# Example param space for TrendFollowingStrategy
trend_param_space = {
    'short_window': Integer(5, 30),
    'long_window': Integer(15, 60)
}

rsi_param_space = {
    'rsi_period': Integer(7, 21),
    'oversold': Integer(10, 40),
    'overbought': Integer(60, 90)
}

macd_param_space = {
    'fast_period': Integer(5, 20),
    'slow_period': Integer(15, 40),
    'signal_period': Integer(5, 15)
}

# Add more param spaces as needed

def auto_optimize_all_strategies(df: pd.DataFrame, metric: str = 'sharpe_ratio', n_calls: int = 30):
    from strategies import TrendFollowingStrategy, RSIStrategy, MACDStrategy
    results = {}
    for strat_cls, param_space in [
        (TrendFollowingStrategy, trend_param_space),
        (RSIStrategy, rsi_param_space),
        (MACDStrategy, macd_param_space)
    ]:
        print(f"Optimizing {strat_cls.__name__}...")
        strat = strat_cls()
        optimizer = StrategyOptimizer(strat, param_space, metric)
        best_params, best_score = optimizer.optimize(df, n_calls=n_calls)
        print(f"  Best params: {best_params}")
        print(f"  Best {metric}: {best_score:.4f}")
        results[strat_cls.__name__] = (best_params, best_score)
    return results 