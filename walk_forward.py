import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from improved_backtest import run_improved_backtest
from improved_optimizer import optimize_improved_backtest
from ensemble_strategy import run_ensemble_backtest

class WalkForwardValidator:
    """Walk-forward validation for robust strategy testing"""
    
    def __init__(self, train_window=126, test_window=42, step_size=21):  # Reduced for shorter periods
        """
        Initialize walk-forward validator
        
        Args:
            train_window: Number of days for training (default: 1 year)
            test_window: Number of days for testing (default: 3 months)
            step_size: Number of days to step forward (default: 1 month)
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step_size = step_size
        self.results = []
        
    def run_walk_forward(self, df: pd.DataFrame, strategy_type='improved', optimize_params=True):
        """
        Run walk-forward validation
        
        Args:
            df: DataFrame with OHLCV data
            strategy_type: 'improved', 'ensemble', or 'optimized'
            optimize_params: Whether to optimize parameters in each training window
        """
        self.results = []
        
        # Calculate number of windows
        total_days = len(df)
        start_idx = self.train_window
        
        while start_idx + self.test_window <= total_days:
            # Define train and test periods
            train_end = start_idx
            test_start = start_idx
            test_end = start_idx + self.test_window
            
            # Split data
            train_data = df.iloc[:train_end].copy()
            test_data = df.iloc[test_start:test_end].copy()
            
            print(f"Training on {train_data.index[0].date()} to {train_data.index[-1].date()}")
            print(f"Testing on {test_data.index[0].date()} to {test_data.index[-1].date()}")
            
            # Run strategy based on type
            if strategy_type == 'ensemble':
                result = self._run_ensemble_strategy(test_data)
            elif strategy_type == 'optimized' and optimize_params:
                result = self._run_optimized_strategy(train_data, test_data)
            else:
                result = self._run_improved_strategy(test_data)
            
            # Store results
            self.results.append({
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'result': result
            })
            
            # Move to next window
            start_idx += self.step_size
            
        return self._aggregate_results()
    
    def _run_improved_strategy(self, test_data: pd.DataFrame):
        """Run improved backtest on test data"""
        return run_improved_backtest(test_data, allow_short=True)
    
    def _run_ensemble_strategy(self, test_data: pd.DataFrame):
        """Run ensemble strategy on test data"""
        return run_ensemble_backtest(test_data)
    
    def _run_optimized_strategy(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Optimize parameters on train data and test on test data"""
        # Optimize parameters on training data
        best_params, best_score = optimize_improved_backtest(
            train_data, 
            n_calls=20,  # Fewer calls for faster execution
            maximize_metric='total_return'
        )
        
        print(f"Optimized parameters: {best_params}")
        print(f"Training score: {best_score:.2f}%")
        
        # Apply optimized parameters to test data
        return run_improved_backtest(test_data, **best_params)
    
    def _aggregate_results(self) -> Dict:
        """Aggregate results from all walk-forward windows"""
        if not self.results:
            return {}
        
        # Calculate aggregate metrics
        total_returns = [r['result'].total_return for r in self.results]
        win_rates = [r['result'].win_rate for r in self.results]
        sharpe_ratios = [r['result'].sharpe_ratio for r in self.results]
        max_drawdowns = [r['result'].max_drawdown for r in self.results]
        profit_factors = [r['result'].profit_factor for r in self.results]
        
        # Calculate statistics
        aggregate_results = {
            'num_windows': len(self.results),
            'avg_total_return': np.mean(total_returns),
            'std_total_return': np.std(total_returns),
            'min_total_return': np.min(total_returns),
            'max_total_return': np.max(total_returns),
            'avg_win_rate': np.mean(win_rates),
            'avg_sharpe_ratio': np.mean(sharpe_ratios),
            'avg_max_drawdown': np.mean(max_drawdowns),
            'avg_profit_factor': np.mean(profit_factors),
            'positive_windows': sum(1 for r in total_returns if r > 0),
            'negative_windows': sum(1 for r in total_returns if r <= 0),
            'consistency_score': sum(1 for r in total_returns if r > 0) / len(total_returns)
        }
        
        return aggregate_results
    
    def print_results(self):
        """Print walk-forward validation results"""
        if not self.results:
            print("No results to display")
            return
        
        aggregate = self._aggregate_results()
        
        print("\n" + "="*80)
        print("WALK-FORWARD VALIDATION RESULTS")
        print("="*80)
        print(f"Number of windows: {aggregate['num_windows']}")
        print(f"Training window: {self.train_window} days")
        print(f"Testing window: {self.test_window} days")
        print(f"Step size: {self.step_size} days")
        print()
        
        print("PERFORMANCE METRICS:")
        print(f"Average Total Return: {aggregate['avg_total_return']:.2f}% ± {aggregate['std_total_return']:.2f}%")
        print(f"Return Range: {aggregate['min_total_return']:.2f}% to {aggregate['max_total_return']:.2f}%")
        print(f"Average Win Rate: {aggregate['avg_win_rate']:.2f}%")
        print(f"Average Sharpe Ratio: {aggregate['avg_sharpe_ratio']:.2f}")
        print(f"Average Max Drawdown: {aggregate['avg_max_drawdown']:.2f}%")
        print(f"Average Profit Factor: {aggregate['avg_profit_factor']:.2f}")
        print()
        
        print("ROBUSTNESS METRICS:")
        print(f"Positive Windows: {aggregate['positive_windows']}/{aggregate['num_windows']}")
        print(f"Negative Windows: {aggregate['negative_windows']}/{aggregate['num_windows']}")
        print(f"Consistency Score: {aggregate['consistency_score']:.2%}")
        print()
        
        print("INDIVIDUAL WINDOW RESULTS:")
        print("-" * 80)
        for i, result in enumerate(self.results):
            r = result['result']
            print(f"Window {i+1}: {result['test_start'].date()} to {result['test_end'].date()}")
            print(f"  Return: {r.total_return:.2f}%, Win Rate: {r.win_rate:.1f}%, "
                  f"Sharpe: {r.sharpe_ratio:.2f}, Max DD: {r.max_drawdown:.2f}%")
        print("="*80)

def run_walk_forward_validation(df: pd.DataFrame, strategy_type='ensemble', optimize_params=True):
    """Convenience function to run walk-forward validation"""
    validator = WalkForwardValidator()
    results = validator.run_walk_forward(df, strategy_type, optimize_params)
    validator.print_results()
    return results 