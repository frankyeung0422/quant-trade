import numpy as np
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from improved_backtest import run_improved_backtest
import pandas as pd

def optimize_improved_backtest(df: pd.DataFrame, n_calls=30, maximize_metric='total_return'):
    """
    Bayesian optimization for improved backtest parameters.
    """
    # Parameter space
    space = [
        Real(0.005, 0.05, name='risk_per_trade'),
        Real(1.0, 3.0, name='stop_loss_atr_mult'),
        Real(1.0, 4.0, name='take_profit_rr'),
        Integer(5, 10, name='min_confirmations'),
        Categorical([True, False], name='allow_short')
    ]
    def objective(params):
        risk_per_trade, stop_loss_atr_mult, take_profit_rr, min_confirmations, allow_short = params
        result = run_improved_backtest(
            df,
            risk_per_trade=risk_per_trade,
            stop_loss_atr_mult=stop_loss_atr_mult,
            take_profit_rr=take_profit_rr,
            min_confirmations=min_confirmations,
            allow_short=allow_short
        )
        if maximize_metric == 'total_return':
            return -result.total_return
        elif maximize_metric == 'sharpe_ratio':
            return -result.sharpe_ratio
        elif maximize_metric == 'profit_factor':
            return -result.profit_factor
        else:
            return -result.total_return
    res = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
    best_params = {
        'risk_per_trade': res.x[0],
        'stop_loss_atr_mult': res.x[1],
        'take_profit_rr': res.x[2],
        'min_confirmations': res.x[3],
        'allow_short': res.x[4]
    }
    best_score = -res.fun
    return best_params, best_score, res 