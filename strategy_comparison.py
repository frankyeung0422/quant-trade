import pandas as pd
import numpy as np
from typing import Dict, List
from strategies import get_all_strategies, BaseStrategy
from backtest import run_backtest, BacktestResult

class StrategyComparison:
    """Compare multiple trading strategies"""
    
    def __init__(self, strategies: List[BaseStrategy] = None):
        self.strategies = strategies or get_all_strategies()
        self.results = {}
    
    def run_comparison(self, df: pd.DataFrame) -> Dict[str, BacktestResult]:
        """Run backtest for all strategies"""
        print("Running strategy comparison...")
        print("="*80)
        
        for strategy in self.strategies:
            print(f"\nTesting {strategy.name}...")
            
            # Create a copy of the dataframe for this strategy
            strategy_df = df.copy()
            
            # Add strategy signals to dataframe
            strategy_df['Strategy_Signal'] = [
                strategy.generate_signal(strategy_df, i) 
                for i in range(len(strategy_df))
            ]
            
            # Run backtest with strategy signals
            result = self._run_strategy_backtest(strategy_df, strategy)
            self.results[strategy.name] = result
            
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Win Rate: {result.win_rate:.2f}%")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Max Drawdown: {result.max_drawdown:.2f}%")
        
        return self.results
    
    def _run_strategy_backtest(self, df: pd.DataFrame, strategy: BaseStrategy) -> BacktestResult:
        """Run backtest using strategy signals instead of analysis module"""
        capital = 10000
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        direction = None
        trades = []
        equity_curve = []
        transaction_cost = 0.001
        
        # Calculate ATR for dynamic stop losses
        df['ATR'] = self._calculate_atr(df, window=14)
        
        for i in range(20, len(df)):
            current_price = df.iloc[i]['Close']
            current_atr = df.iloc[i]['ATR']
            signal = df.iloc[i]['Strategy_Signal']
            
            # Check stop loss and take profit
            if position > 0:
                if direction == 'long':
                    if current_price <= stop_loss or current_price >= take_profit:
                        # Close position
                        capital = position * current_price * (1 - transaction_cost)
                        profit_pct = (current_price - entry_price) / entry_price
                        trades.append({
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'profit_pct': profit_pct,
                            'exit_reason': 'stop_loss' if current_price <= stop_loss else 'take_profit'
                        })
                        position = 0
                        entry_price = 0
                        stop_loss = 0
                        take_profit = 0
                        direction = None
            
            # Execute strategy signal
            if signal == 'Buy' and position == 0:
                direction = 'long'
                entry_price = current_price
                position_size = self._calculate_position_size(capital, current_price)
                position = position_size
                capital -= position_size * current_price * (1 + transaction_cost)
                
                # Set stop loss and take profit
                stop_loss = entry_price - (current_atr * 2)
                take_profit = entry_price + (entry_price * 0.05 * 2)  # 2:1 risk-reward
            
            elif signal == 'Sell' and position > 0:
                # Close position on sell signal
                capital = position * current_price * (1 - transaction_cost)
                profit_pct = (current_price - entry_price) / entry_price
                trades.append({
                    'entry_price': entry_price,
                    'exit_price': current_price,
                    'profit_pct': profit_pct,
                    'exit_reason': 'signal'
                })
                position = 0
                entry_price = 0
                stop_loss = 0
                take_profit = 0
                direction = None
            
            # Track equity curve
            current_equity = capital + (position * current_price if position > 0 else 0)
            equity_curve.append(current_equity)
        
        # Close final position
        if position > 0:
            final_price = df.iloc[-1]['Close']
            capital = position * final_price * (1 - transaction_cost)
            profit_pct = (final_price - entry_price) / entry_price
            trades.append({
                'entry_price': entry_price,
                'exit_price': final_price,
                'profit_pct': profit_pct,
                'exit_reason': 'final'
            })
        
        # Calculate performance metrics
        result = BacktestResult()
        result.total_return = (capital - 10000) / 10000 * 100
        result.num_trades = len(trades)
        result.trades = trades
        
        if trades:
            wins = [t for t in trades if t['profit_pct'] > 0]
            losses = [t for t in trades if t['profit_pct'] <= 0]
            
            result.win_rate = len(wins) / len(trades) * 100
            result.avg_win = np.mean([t['profit_pct'] for t in wins]) * 100 if wins else 0
            result.avg_loss = np.mean([t['profit_pct'] for t in losses]) * 100 if losses else 0
            
            if losses:
                result.profit_factor = abs(sum([t['profit_pct'] for t in wins])) / abs(sum([t['profit_pct'] for t in losses]))
        
        # Calculate max drawdown
        if equity_curve:
            peak = equity_curve[0]
            max_dd = 0
            for equity in equity_curve:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)
            result.max_drawdown = max_dd * 100
        
        # Calculate Sharpe ratio
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            if len(returns) > 0 and np.std(returns) > 0:
                result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        return result
    
    def _calculate_atr(self, df, window=14):
        """Calculate Average True Range"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        return atr
    
    def _calculate_position_size(self, capital, price, risk_per_trade=0.02, stop_loss_pct=0.05):
        """Calculate position size based on risk management"""
        risk_amount = capital * risk_per_trade
        stop_loss_amount = price * stop_loss_pct
        position_size = risk_amount / stop_loss_amount
        return min(position_size, capital * 0.95 / price)
    
    def print_comparison_table(self):
        """Print comparison table of all strategies"""
        if not self.results:
            print("No results to compare. Run comparison first.")
            return
        
        print("\n" + "="*100)
        print("STRATEGY COMPARISON RESULTS")
        print("="*100)
        
        # Create comparison table
        table_data = []
        for strategy_name, result in self.results.items():
            table_data.append({
                'Strategy': strategy_name,
                'Total Return (%)': f"{result.total_return:.2f}",
                'Win Rate (%)': f"{result.win_rate:.2f}",
                'Num Trades': result.num_trades,
                'Avg Win (%)': f"{result.avg_win:.2f}",
                'Avg Loss (%)': f"{result.avg_loss:.2f}",
                'Profit Factor': f"{result.profit_factor:.2f}",
                'Max Drawdown (%)': f"{result.max_drawdown:.2f}",
                'Sharpe Ratio': f"{result.sharpe_ratio:.2f}"
            })
        
        # Sort by total return
        table_data.sort(key=lambda x: float(x['Total Return (%)']), reverse=True)
        
        # Print table
        headers = list(table_data[0].keys())
        col_widths = [max(len(str(row[col])) for row in table_data + [dict(zip(headers, headers))]) for col in headers]
        
        # Print header
        header_str = " | ".join(f"{h:<{w}}" for h, w in zip(headers, col_widths))
        print(header_str)
        print("-" * len(header_str))
        
        # Print rows
        for row in table_data:
            row_str = " | ".join(f"{str(row[col]):<{w}}" for col, w in zip(headers, col_widths))
            print(row_str)
        
        print("="*100)
        
        # Find best strategy
        best_strategy = max(self.results.items(), key=lambda x: x[1].total_return)
        print(f"\nBest Strategy: {best_strategy[0]} ({best_strategy[1].total_return:.2f}% return)")
        
        # Find best Sharpe ratio
        best_sharpe = max(self.results.items(), key=lambda x: x[1].sharpe_ratio)
        print(f"Best Risk-Adjusted: {best_sharpe[0]} (Sharpe: {best_sharpe[1].sharpe_ratio:.2f})")
    
    def get_best_strategy(self, metric='total_return') -> str:
        """Get the best strategy based on specified metric"""
        if not self.results:
            return None
        
        if metric == 'total_return':
            return max(self.results.items(), key=lambda x: x[1].total_return)[0]
        elif metric == 'sharpe_ratio':
            return max(self.results.items(), key=lambda x: x[1].sharpe_ratio)[0]
        elif metric == 'win_rate':
            return max(self.results.items(), key=lambda x: x[1].win_rate)[0]
        elif metric == 'profit_factor':
            return max(self.results.items(), key=lambda x: x[1].profit_factor)[0]
        else:
            return max(self.results.items(), key=lambda x: x[1].total_return)[0]

def run_strategy_comparison(df: pd.DataFrame):
    """Convenience function to run strategy comparison"""
    comparison = StrategyComparison()
    results = comparison.run_comparison(df)
    comparison.print_comparison_table()
    return comparison 