import pandas as pd
import numpy as np
from analysis import generate_analysis

class ImprovedBacktestResult:
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
        self.equity_curve = []
        self.daily_returns = []

def calculate_improved_position_size(capital, price, risk_per_trade=0.01, stop_loss_pct=0.03):
    """Calculate position size with better risk management"""
    risk_amount = capital * risk_per_trade
    stop_loss_amount = price * stop_loss_pct
    position_size = risk_amount / stop_loss_amount
    # Limit position size to max 10% of capital
    max_position = capital * 0.1 / price
    return min(position_size, max_position)

def calculate_dynamic_stop_loss(entry_price, direction, atr, multiplier=1.5):
    """Calculate dynamic stop loss based on ATR"""
    if direction == 'long':
        return entry_price - (atr * multiplier)
    else:
        return entry_price + (atr * multiplier)

def calculate_dynamic_take_profit(entry_price, direction, atr, risk_reward_ratio=2):
    """Calculate dynamic take profit based on ATR"""
    if direction == 'long':
        return entry_price + (atr * risk_reward_ratio)
    else:
        return entry_price - (atr * risk_reward_ratio)

def run_improved_backtest(
    df: pd.DataFrame,
    initial_capital=10000,
    transaction_cost=0.0005,
    max_concurrent_trades=5,
    risk_per_trade=0.01,
    stop_loss_atr_mult=1.5,
    take_profit_rr=2.0,
    min_confirmations=7,
    allow_short=False
):
    """
    Improved backtest with parameterization and short selling support.
    """
    capital = initial_capital
    open_trades = []
    trades = []
    equity_curve = [initial_capital]
    daily_returns = []
    
    df['ATR'] = calculate_atr(df, window=14)
    df['Price_Change'] = df['Close'].pct_change()
    df['Volatility'] = df['Price_Change'].rolling(20).std()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    df['RSI_MA'] = df['RSI_14'].rolling(10).mean()

    for i in range(50, len(df)):
        current_price = df.iloc[i]['Close']
        current_atr = df.iloc[i]['ATR']
        current_volatility = df.iloc[i]['Volatility']
        
        # 1. Check all open trades for exit conditions
        closed_indices = []
        for idx, trade in enumerate(open_trades):
            if trade['direction'] == 'long':
                if current_price <= trade['stop_loss']:
                    exit_reason = 'stop_loss'
                elif current_price >= trade['take_profit']:
                    exit_reason = 'take_profit'
                elif 'exit_signal' in trade and trade['exit_signal']:
                    exit_reason = 'signal'
                else:
                    continue
                exit_value = trade['size'] * current_price * (1 - transaction_cost)
                capital += exit_value
                profit_pct = (current_price - trade['entry_price']) / trade['entry_price']
                trades.append({
                    'entry_price': trade['entry_price'],
                    'exit_price': current_price,
                    'profit_pct': profit_pct,
                    'exit_reason': exit_reason,
                    'hold_days': i - trade['entry_index'] + 1,
                    'direction': 'long'
                })
                closed_indices.append(idx)
            elif trade['direction'] == 'short':
                if current_price >= trade['stop_loss']:
                    exit_reason = 'stop_loss'
                elif current_price <= trade['take_profit']:
                    exit_reason = 'take_profit'
                elif 'exit_signal' in trade and trade['exit_signal']:
                    exit_reason = 'signal'
                else:
                    continue
                exit_value = trade['size'] * (2 * trade['entry_price'] - current_price) * (1 - transaction_cost)
                capital += exit_value
                profit_pct = (trade['entry_price'] - current_price) / trade['entry_price']
                trades.append({
                    'entry_price': trade['entry_price'],
                    'exit_price': current_price,
                    'profit_pct': profit_pct,
                    'exit_reason': exit_reason,
                    'hold_days': i - trade['entry_index'] + 1,
                    'direction': 'short'
                })
                closed_indices.append(idx)
        for idx in reversed(closed_indices):
            del open_trades[idx]
        
        # 2. Generate trading signal
        if 'Strategy_Signal' in df.columns:
            signal = df.iloc[i]['Strategy_Signal']
        else:
            sub_df = df.iloc[:i+1]
            _, signal = generate_analysis(sub_df)
        
        # 3. Enhanced entry criteria
        rsi = df.iloc[i]['RSI_14']
        rsi_ma = df.iloc[i]['RSI_MA']
        macd = df.iloc[i]['MACD']
        macd_signal = df.iloc[i]['MACD_Signal']
        volume = df.iloc[i]['Volume']
        avg_volume = df.iloc[i-20:i]['Volume'].mean()
        sma_20 = df.iloc[i]['SMA_20']
        sma_50 = df.iloc[i]['SMA_50']
        ema_12 = df.iloc[i]['EMA_12']
        ema_26 = df.iloc[i]['EMA_26']
        volume_confirmed = volume > avg_volume * 1.05
        macd_confirmed = macd > macd_signal and macd > 0
        rsi_confirmed = 40 < rsi < 60
        rsi_trending_up = rsi > rsi_ma
        volatility_ok = current_volatility < 0.025
        price_above_sma20 = current_price > sma_20 * 1.005
        price_above_sma50 = current_price > sma_50 * 1.005
        ema_bullish = ema_12 > ema_26
        sma_bullish = sma_20 > sma_50
        momentum_positive = df.iloc[i]['Price_Change'] > 0
        confirmations = sum([
            volume_confirmed,
            macd_confirmed,
            rsi_confirmed,
            rsi_trending_up,
            volatility_ok,
            price_above_sma20,
            price_above_sma50,
            ema_bullish,
            sma_bullish,
            momentum_positive
        ])
        if signal == 'Buy' and len(open_trades) < max_concurrent_trades and capital > 0:
            if confirmations >= min_confirmations:
                direction = 'long'
                entry_price = current_price
                position_size = calculate_improved_position_size(capital, current_price, risk_per_trade, stop_loss_atr_mult / 10)
                if position_size * current_price * (1 + transaction_cost) > capital:
                    position_size = capital / (current_price * (1 + transaction_cost))
                if position_size > 0:
                    capital -= position_size * current_price * (1 + transaction_cost)
                    stop_loss = calculate_dynamic_stop_loss(entry_price, direction, current_atr, stop_loss_atr_mult)
                    take_profit = calculate_dynamic_take_profit(entry_price, direction, current_atr, take_profit_rr)
                    open_trades.append({
                        'entry_price': entry_price,
                        'size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'direction': direction,
                        'entry_index': i
                    })
        if allow_short and signal == 'Sell' and len(open_trades) < max_concurrent_trades and capital > 0:
            if confirmations >= min_confirmations:
                direction = 'short'
                entry_price = current_price
                position_size = calculate_improved_position_size(capital, current_price, risk_per_trade, stop_loss_atr_mult / 10)
                if position_size * current_price * (1 + transaction_cost) > capital:
                    position_size = capital / (current_price * (1 + transaction_cost))
                if position_size > 0:
                    capital -= position_size * current_price * (1 + transaction_cost)
                    stop_loss = calculate_dynamic_stop_loss(entry_price, direction, current_atr, stop_loss_atr_mult)
                    take_profit = calculate_dynamic_take_profit(entry_price, direction, current_atr, take_profit_rr)
                    open_trades.append({
                        'entry_price': entry_price,
                        'size': position_size,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'direction': direction,
                        'entry_index': i
                    })
        # Set exit_signal for all open trades if signal == 'Sell' (for long) or 'Buy' (for short)
        if signal == 'Sell':
            for trade in open_trades:
                if trade['direction'] == 'long':
                    trade['exit_signal'] = True
        if allow_short and signal == 'Buy':
            for trade in open_trades:
                if trade['direction'] == 'short':
                    trade['exit_signal'] = True
        # 5. Track equity curve
        current_equity = capital + sum([
            t['size'] * current_price if t['direction'] == 'long' else t['size'] * (2 * t['entry_price'] - current_price)
            for t in open_trades
        ])
        equity_curve.append(current_equity)
        if len(equity_curve) > 1:
            daily_return = (current_equity - equity_curve[-2]) / equity_curve[-2]
            daily_returns.append(daily_return)
    # 6. Close all open trades at the end
    for trade in open_trades:
        final_price = df.iloc[-1]['Close']
        if trade['direction'] == 'long':
            exit_value = trade['size'] * final_price * (1 - transaction_cost)
            capital += exit_value
            profit_pct = (final_price - trade['entry_price']) / trade['entry_price']
        else:
            exit_value = trade['size'] * (2 * trade['entry_price'] - final_price) * (1 - transaction_cost)
            capital += exit_value
            profit_pct = (trade['entry_price'] - final_price) / trade['entry_price']
        trades.append({
            'entry_price': trade['entry_price'],
            'exit_price': final_price,
            'profit_pct': profit_pct,
            'exit_reason': 'final',
            'hold_days': len(df) - trade['entry_index'],
            'direction': trade['direction']
        })
    # 7. Calculate performance metrics (same as before)
    result = ImprovedBacktestResult()
    result.total_return = (capital - initial_capital) / initial_capital * 100
    result.num_trades = len(trades)
    result.trades = trades
    result.equity_curve = equity_curve
    result.daily_returns = daily_returns
    if trades:
        wins = [t for t in trades if t['profit_pct'] > 0]
        losses = [t for t in trades if t['profit_pct'] <= 0]
        result.win_rate = len(wins) / len(trades) * 100
        result.avg_win = np.mean([t['profit_pct'] for t in wins]) * 100 if wins else 0
        result.avg_loss = np.mean([t['profit_pct'] for t in losses]) * 100 if losses else 0
        if losses:
            result.profit_factor = abs(sum([t['profit_pct'] for t in wins])) / abs(sum([t['profit_pct'] for t in losses]))
    if equity_curve:
        peak = equity_curve[0]
        max_dd = 0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        result.max_drawdown = max_dd * 100
    if len(daily_returns) > 0 and np.std(daily_returns) > 0:
        result.sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
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

def print_improved_backtest_results(result: ImprovedBacktestResult):
    """Print comprehensive improved backtest results"""
    print("\n" + "="*70)
    print("IMPROVED BACKTEST RESULTS")
    print("="*70)
    print(f"Total Return: {result.total_return:.2f}%")
    print(f"Number of Trades: {result.num_trades}")
    print(f"Win Rate: {result.win_rate:.2f}%")
    print(f"Average Win: {result.avg_win:.2f}%")
    print(f"Average Loss: {result.avg_loss:.2f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    
    if result.trades:
        avg_hold_days = np.mean([t.get('hold_days', 1) for t in result.trades])
        print(f"Average Hold Days: {avg_hold_days:.1f}")
    
    print("="*70)
    
    if result.trades:
        print("\nTRADE DETAILS:")
        print("-"*70)
        for i, trade in enumerate(result.trades, 1):
            hold_days = trade.get('hold_days', 'N/A')
            print(f"Trade {i}: Entry: ${trade['entry_price']:.2f}, "
                  f"Exit: ${trade['exit_price']:.2f}, "
                  f"P&L: {trade['profit_pct']*100:.2f}%, "
                  f"Hold: {hold_days} days, "
                  f"Reason: {trade['exit_reason']}")

def run_strategy_backtest(df: pd.DataFrame, strategy_signal_column='Strategy_Signal'):
    """Run backtest using strategy signals from dataframe"""
    df_copy = df.copy()
    if strategy_signal_column not in df_copy.columns:
        # Generate signals using analysis module
        signals = []
        for i in range(len(df_copy)):
            if i < 50:  # Need enough data for analysis
                signals.append('Hold')
            else:
                sub_df = df_copy.iloc[:i+1]
                _, signal = generate_analysis(sub_df)
                signals.append(signal)
        df_copy[strategy_signal_column] = signals
    
    return run_improved_backtest(df_copy) 