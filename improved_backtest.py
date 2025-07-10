import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ImprovedBacktestResult:
    def __init__(self):
        self.total_return = 0
        self.annualized_return = 0
        self.win_rate = 0
        self.num_trades = 0
        self.max_drawdown = 0
        self.sharpe_ratio = 0
        self.profit_factor = 0
        self.avg_win = 0
        self.avg_loss = 0
        self.max_consecutive_losses = 0
        self.trades = []
        self.equity_curve = []
        self.strategy_name = ""

def calculate_advanced_indicators(df):
    """Calculate advanced technical indicators"""
    # Basic indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50).mean()
    
    # RSI
    df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['Close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid']
    
    # Stochastic
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['STOCH_K'] = stoch.stoch()
    df['STOCH_D'] = stoch.stoch_signal()
    
    # Williams %R
    df['WILLR'] = ta.momentum.WilliamsRIndicator(df['High'], df['Low'], df['Close']).williams_r()
    
    # CCI
    df['CCI'] = ta.trend.CCIIndicator(df['High'], df['Low'], df['Close']).cci()
    
    # ADX
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'])
    df['ADX'] = adx.adx()
    df['DI_Plus'] = adx.adx_pos()
    df['DI_Minus'] = adx.adx_neg()
    
    # ATR
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
    
    # Volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Price momentum
    df['Price_Change'] = df['Close'].pct_change()
    df['Price_Change_5'] = df['Close'].pct_change(5)
    df['Price_Change_10'] = df['Close'].pct_change(10)
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Support and Resistance
    df['Support'] = df['Low'].rolling(window=20).min()
    df['Resistance'] = df['High'].rolling(window=20).max()
    
    return df

def generate_ml_signals(df, lookback=60):
    """Generate machine learning based signals"""
    # Prepare features
    features = ['RSI_14', 'MACD', 'STOCH_K', 'WILLR', 'CCI', 'Volume_Ratio', 
                'Price_Change', 'Price_Change_5', 'Price_Change_10', 'BB_Width']
    
    # Create target (1 if price increases by 2% in next 5 days, 0 otherwise)
    df['Target'] = (df['Close'].shift(-5) / df['Close'] - 1) > 0.02
    
    # Prepare data for ML
    X = df[features].fillna(0)
    y = df['Target'].fillna(False)
    
    # Train model on historical data
    signals = np.zeros(len(df))
    
    for i in range(lookback, len(df) - 5):
        # Train on past data
        X_train = X.iloc[i-lookback:i]
        y_train = y.iloc[i-lookback:i]
        
        # Skip if not enough data
        if len(X_train) < 30 or y_train.sum() < 5:
            continue
            
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict current signal
        X_current = X.iloc[i:i+1]
        prediction = model.predict_proba(X_current)[0][1]  # Probability of positive outcome
        
        if prediction > 0.6:  # High confidence buy signal
            signals[i] = 1
        elif prediction < 0.3:  # High confidence sell signal
            signals[i] = -1
    
    return signals

def generate_momentum_signals(df):
    """Generate momentum-based signals"""
    signals = np.zeros(len(df))
    
    for i in range(20, len(df)):
        # Multiple momentum confirmations
        rsi = df.iloc[i]['RSI_14']
        macd = df.iloc[i]['MACD']
        macd_signal = df.iloc[i]['MACD_Signal']
        stoch_k = df.iloc[i]['STOCH_K']
        stoch_d = df.iloc[i]['STOCH_D']
        volume_ratio = df.iloc[i]['Volume_Ratio']
        
        # Strong buy conditions
        buy_conditions = [
            rsi > 30 and rsi < 70,  # RSI not extreme
            macd > macd_signal,     # MACD bullish crossover
            macd > 0,               # MACD above zero
            stoch_k > stoch_d,      # Stochastic bullish
            stoch_k < 80,           # Not overbought
            volume_ratio > 1.2,     # Above average volume
            df.iloc[i]['Close'] > df.iloc[i]['SMA_20'],  # Price above SMA
            df.iloc[i]['SMA_20'] > df.iloc[i]['SMA_50']  # Golden cross
        ]
        
        # Strong sell conditions
        sell_conditions = [
            rsi > 70,               # RSI overbought
            macd < macd_signal,     # MACD bearish crossover
            macd < 0,               # MACD below zero
            stoch_k < stoch_d,      # Stochastic bearish
            stoch_k > 20,           # Not oversold
            df.iloc[i]['Close'] < df.iloc[i]['SMA_20'],  # Price below SMA
            df.iloc[i]['SMA_20'] < df.iloc[i]['SMA_50']  # Death cross
        ]
        
        if sum(buy_conditions) >= 6:  # Need majority of conditions
            signals[i] = 1
        elif sum(sell_conditions) >= 5:
            signals[i] = -1
    
    return signals

def generate_mean_reversion_signals(df):
    """Generate mean reversion signals"""
    signals = np.zeros(len(df))
    
    for i in range(20, len(df)):
        current_price = df.iloc[i]['Close']
        bb_high = df.iloc[i]['BB_High']
        bb_low = df.iloc[i]['BB_Low']
        bb_mid = df.iloc[i]['BB_Mid']
        rsi = df.iloc[i]['RSI_14']
        willr = df.iloc[i]['WILLR']
        
        # Oversold conditions (buy signal)
        oversold_conditions = [
            current_price <= bb_low * 1.01,  # Price at or below lower BB
            rsi < 30,                         # RSI oversold
            willr < -80,                      # Williams %R oversold
            df.iloc[i]['Volume_Ratio'] > 1.5  # High volume confirmation
        ]
        
        # Overbought conditions (sell signal)
        overbought_conditions = [
            current_price >= bb_high * 0.99,  # Price at or above upper BB
            rsi > 70,                          # RSI overbought
            willr > -20,                       # Williams %R overbought
            df.iloc[i]['Volume_Ratio'] > 1.5   # High volume confirmation
        ]
        
        if sum(oversold_conditions) >= 3:
            signals[i] = 1
        elif sum(overbought_conditions) >= 3:
            signals[i] = -1
    
    return signals

def generate_sma_crossover_signals(df, short_window=20, long_window=50):
    signals = np.zeros(len(df))
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()
    prev_signal = 0
    for i in range(long_window, len(df)):
        if df['SMA_short'].iloc[i] > df['SMA_long'].iloc[i] and df['SMA_short'].iloc[i-1] <= df['SMA_long'].iloc[i-1]:
            signals[i] = 1  # Buy
        elif df['SMA_short'].iloc[i] < df['SMA_long'].iloc[i] and df['SMA_short'].iloc[i-1] >= df['SMA_long'].iloc[i-1]:
            signals[i] = -1  # Sell
        else:
            signals[i] = prev_signal
        prev_signal = signals[i]
    return signals

def generate_rsi_macd_signals(df):
    signals = np.zeros(len(df))
    for i in range(1, len(df)):
        rsi = df['RSI_14'].iloc[i]
        macd = df['MACD'].iloc[i]
        macd_prev = df['MACD'].iloc[i-1]
        macd_signal = df['MACD_Signal'].iloc[i]
        macd_signal_prev = df['MACD_Signal'].iloc[i-1]
        # Buy: RSI < 35 and MACD crosses above signal
        if rsi < 35 and macd > macd_signal and macd_prev <= macd_signal_prev:
            signals[i] = 1
        # Sell: RSI > 65 and MACD crosses below signal
        elif rsi > 65 and macd < macd_signal and macd_prev >= macd_signal_prev:
            signals[i] = -1
    return signals

def generate_volatility_breakout_signals(df):
    signals = np.zeros(len(df))
    for i in range(1, len(df)):
        close = df['Close'].iloc[i]
        upper = df['BB_High'].iloc[i]
        lower = df['BB_Low'].iloc[i]
        prev_close = df['Close'].iloc[i-1]
        # Buy breakout
        if prev_close <= upper and close > upper:
            signals[i] = 1
        # Sell breakdown
        elif prev_close >= lower and close < lower:
            signals[i] = -1
    return signals

def run_buy_and_hold(df, initial_capital=10000):
    entry_price = df['Close'].iloc[0]
    exit_price = df['Close'].iloc[-1]
    total_return = (exit_price - entry_price) / entry_price * 100
    result = ImprovedBacktestResult()
    result.total_return = total_return
    result.annualized_return = total_return  # Approximate for 1y
    result.num_trades = 1
    result.win_rate = 100 if total_return > 0 else 0
    result.avg_win = total_return if total_return > 0 else 0
    result.avg_loss = total_return if total_return < 0 else 0
    result.max_drawdown = 0  # Not calculated
    result.sharpe_ratio = 0  # Not calculated
    result.trades = [{
        'entry_date': df.index[0],
        'exit_date': df.index[-1],
        'entry_price': entry_price,
        'exit_price': exit_price,
        'profit_pct': total_return,
        'exit_reason': 'hold'
    }]
    result.strategy_name = 'buy_and_hold'
    return result

def run_improved_backtest(df, strategy='combined', initial_capital=10000, transaction_cost=0.001):
    """
    Run improved backtest with multiple strategies
    """
    # Calculate all indicators
    df = calculate_advanced_indicators(df)
    
    # Generate signals based on strategy
    if strategy == 'ml':
        signals = generate_ml_signals(df)
    elif strategy == 'momentum':
        signals = generate_momentum_signals(df)
    elif strategy == 'mean_reversion':
        signals = generate_mean_reversion_signals(df)
    elif strategy == 'sma_crossover':
        signals = generate_sma_crossover_signals(df)
    elif strategy == 'rsi_macd':
        signals = generate_rsi_macd_signals(df)
    elif strategy == 'volatility_breakout':
        signals = generate_volatility_breakout_signals(df)
    elif strategy == 'buy_and_hold':
        return run_buy_and_hold(df, initial_capital)
    elif strategy == 'combined':
        # Combine all strategies
        ml_signals = generate_ml_signals(df)
        momentum_signals = generate_momentum_signals(df)
        mean_rev_signals = generate_mean_reversion_signals(df)
        sma_crossover_signals = generate_sma_crossover_signals(df)
        rsi_macd_signals = generate_rsi_macd_signals(df)
        volatility_breakout_signals = generate_volatility_breakout_signals(df)
        
        # Weighted combination
        signals = (ml_signals * 0.3 + momentum_signals * 0.3 + mean_rev_signals * 0.2)
        signals += sma_crossover_signals * 0.1
        signals += rsi_macd_signals * 0.05
        signals += volatility_breakout_signals * 0.05
        # Convert to discrete signals
        signals = np.where(signals > 0.5, 1, np.where(signals < -0.5, -1, 0))
    
    # Initialize backtest variables
    capital = initial_capital
    position = 0
    entry_price = 0
    trades = []
    equity_curve = [initial_capital]
    
    # Risk management parameters
    max_position_size = 0.2  # Max 20% of capital per trade
    stop_loss_pct = 0.05     # 5% stop loss
    take_profit_pct = 0.10   # 10% take profit
    
    for i in range(20, len(df)):
        current_price = df.iloc[i]['Close']
        current_signal = signals[i]
        
        # Check stop loss and take profit for existing position
        if position > 0:
            profit_pct = (current_price - entry_price) / entry_price
            
            if profit_pct <= -stop_loss_pct or profit_pct >= take_profit_pct:
                # Close position
                exit_price = current_price
                capital = position * exit_price * (1 - transaction_cost)
                profit_pct = (exit_price - entry_price) / entry_price
                
                # Find entry date (approximate)
                entry_idx = max(0, i - 30)  # Look back up to 30 days
                trades.append({
                    'entry_date': df.index[entry_idx],
                    'exit_date': df.index[i],
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit_pct': profit_pct,
                    'exit_reason': 'stop_loss' if profit_pct <= -stop_loss_pct else 'take_profit'
                })
                
                position = 0
                entry_price = 0
        
        # Open new position based on signal
        if current_signal == 1 and position == 0:
            # Buy signal
            position_size = capital * max_position_size / current_price
            position = position_size
            entry_price = current_price
            capital -= position_size * current_price * (1 + transaction_cost)
            
        elif current_signal == -1 and position > 0:
            # Sell signal
            exit_price = current_price
            capital = position * exit_price * (1 - transaction_cost)
            profit_pct = (exit_price - entry_price) / entry_price
            
            # Find entry date (approximate)
            entry_idx = max(0, i - 30)  # Look back up to 30 days
            trades.append({
                'entry_date': df.index[entry_idx],
                'exit_date': df.index[i],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'profit_pct': profit_pct,
                'exit_reason': 'signal'
            })
            
            position = 0
            entry_price = 0
        
        # Track equity curve
        current_equity = capital + (position * current_price if position > 0 else 0)
        equity_curve.append(current_equity)
    
    # Close final position
    if position > 0:
        final_price = df.iloc[-1]['Close']
        capital = position * final_price * (1 - transaction_cost)
        profit_pct = (final_price - entry_price) / entry_price
        
        # Find entry date (approximate)
        entry_idx = max(0, len(df) - 30)  # Look back up to 30 days
        trades.append({
            'entry_date': df.index[entry_idx],
            'exit_date': df.index[-1],
            'entry_price': entry_price,
            'exit_price': final_price,
            'profit_pct': profit_pct,
            'exit_reason': 'final'
        })
    
    # Calculate performance metrics
    result = ImprovedBacktestResult()
    result.total_return = (capital - initial_capital) / initial_capital * 100
    result.num_trades = len(trades)
    result.trades = trades
    result.equity_curve = equity_curve
    result.strategy_name = strategy
    
    # Calculate annualized return
    days = (df.index[-1] - df.index[0]).days
    if days > 0:
        result.annualized_return = ((capital / initial_capital) ** (365 / days) - 1) * 100
    
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
    
    # Calculate max consecutive losses
    consecutive_losses = 0
    max_consecutive_losses = 0
    for trade in trades:
        if trade['profit_pct'] <= 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0
    result.max_consecutive_losses = max_consecutive_losses
    
    return result

def print_improved_results(result):
    """Print comprehensive backtest results"""
    print("\n" + "="*70)
    print(f"IMPROVED BACKTEST RESULTS - {result.strategy_name.upper()} STRATEGY")
    print("="*70)
    print(f"Total Return: {result.total_return:.2f}%")
    print(f"Annualized Return: {result.annualized_return:.2f}%")
    print(f"Number of Trades: {result.num_trades}")
    print(f"Win Rate: {result.win_rate:.1f}%")
    print(f"Average Win: {result.avg_win:.2f}%")
    print(f"Average Loss: {result.avg_loss:.2f}%")
    print(f"Profit Factor: {result.profit_factor:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2f}%")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Max Consecutive Losses: {result.max_consecutive_losses}")
    
    if result.trades:
        print(f"\nRecent Trades:")
        for i, trade in enumerate(result.trades[-5:]):
            print(f"  {i+1}. {trade['entry_date'].strftime('%Y-%m-%d')} → {trade['exit_date'].strftime('%Y-%m-%d')}: "
                  f"{trade['profit_pct']:.2f}% ({trade['exit_reason']})")

def run_strategy_comparison(df, initial_capital=10000):
    """Compare different strategies"""
    strategies = ['buy_and_hold', 'ml', 'momentum', 'mean_reversion', 'sma_crossover', 'rsi_macd', 'volatility_breakout', 'combined']
    results = {}
    
    print("Running strategy comparison...")
    for strategy in strategies:
        print(f"Testing {strategy} strategy...")
        result = run_improved_backtest(df, strategy, initial_capital)
        results[strategy] = result
        print_improved_results(result)
    
    # Find best strategy
    best_strategy = max(results.keys(), key=lambda x: results[x].total_return)
    print(f"\n🏆 BEST STRATEGY: {best_strategy.upper()}")
    print(f"Total Return: {results[best_strategy].total_return:.2f}%")
    
    return results 