import pandas as pd
import numpy as np
from analysis import generate_analysis

class BacktestResult:
    def __init__(self):
        self.total_return = 0
        self.win_rate = 0
        self.num_trades = 0
        self.max_drawdown = 0
        self.sharpe_ratio = 0
        self.profit_factor = 0
        self.avg_win = 0
        self.avg_loss = 0
        self.trades = []

def calculate_position_size(capital, price, risk_per_trade=0.02, stop_loss_pct=0.05):
    """Calculate position size based on risk management"""
    risk_amount = capital * risk_per_trade
    stop_loss_amount = price * stop_loss_pct
    position_size = risk_amount / stop_loss_amount
    return min(position_size, capital * 0.95 / price)  # Max 95% of capital

def calculate_stop_loss(entry_price, direction, atr, multiplier=2):
    """Calculate dynamic stop loss based on ATR"""
    if direction == 'long':
        return entry_price - (atr * multiplier)
    else:
        return entry_price + (atr * multiplier)

def calculate_take_profit(entry_price, direction, risk_reward_ratio=2, stop_loss_pct=0.05):
    """Calculate take profit based on risk-reward ratio"""
    risk_amount = entry_price * stop_loss_pct
    if direction == 'long':
        return entry_price + (risk_amount * risk_reward_ratio)
    else:
        return entry_price - (risk_amount * risk_reward_ratio)

def run_backtest(df: pd.DataFrame, initial_capital=10000, transaction_cost=0.001):
    """
    Advanced backtest with position sizing, stop losses, take profits, and transaction costs.
    """
    capital = initial_capital
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    direction = None
    trades = []
    equity_curve = []
    
    # Calculate ATR for dynamic stop losses
    df['ATR'] = calculate_atr(df, window=14)
    
    for i in range(20, len(df)):
        current_price = df.iloc[i]['Close']
        current_atr = df.iloc[i]['ATR']
        
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
        
        # Generate trading signal
        sub_df = df.iloc[:i+1]
        _, rec = generate_analysis(sub_df)
        
        # Enhanced entry criteria
        if rec == 'Buy' and position == 0:
            # Additional confirmation signals
            rsi = df.iloc[i]['RSI_14']
            macd = df.iloc[i]['MACD']
            macd_signal = df.iloc[i]['MACD_Signal']
            volume = df.iloc[i]['Volume']
            avg_volume = df.iloc[i-20:i]['Volume'].mean()
            
            # Volume confirmation
            volume_confirmed = volume > avg_volume * 1.2
            
            # MACD confirmation
            macd_confirmed = macd > macd_signal and macd > 0
            
            # RSI confirmation (not overbought)
            rsi_confirmed = 30 < rsi < 70
            
            if volume_confirmed and macd_confirmed and rsi_confirmed:
                direction = 'long'
                entry_price = current_price
                position_size = calculate_position_size(capital, current_price)
                position = position_size
                capital -= position_size * current_price * (1 + transaction_cost)
                
                # Set stop loss and take profit
                stop_loss = calculate_stop_loss(entry_price, direction, current_atr)
                take_profit = calculate_take_profit(entry_price, direction)
        
        elif rec == 'Sell' and position > 0:
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
    result.total_return = (capital - initial_capital) / initial_capital * 100
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
    
    # Calculate Sharpe ratio (simplified)
    if len(equity_curve) > 1:
        returns = np.diff(equity_curve) / equity_curve[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)  # Annualized
    
    return result

def calculate_atr(df, window=14):
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

def print_backtest_results(result: BacktestResult):
    """Print comprehensive backtest results"""
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total Return: {result.total_return:.2f}%")
    print(f"Number of Trades: {result.num_trades}")
    print(f"Win Rate: {result.win_rate:.2f}%")
    print(f"Average Win: {result.avg_win:.2f}%")
    print(f"Average Loss: {result.avg_loss:.2f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print("="*60)
    
    if result.trades:
        print("\nTRADE DETAILS:")
        print("-"*60)
        for i, trade in enumerate(result.trades, 1):
            print(f"Trade {i}: Entry: ${trade['entry_price']:.2f}, "
                  f"Exit: ${trade['exit_price']:.2f}, "
                  f"P&L: {trade['profit_pct']*100:.2f}%, "
                  f"Reason: {trade['exit_reason']}") 