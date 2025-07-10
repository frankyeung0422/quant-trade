import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BaseStrategy:
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    def generate_signal(self, df: pd.DataFrame, index: int) -> str:
        """Generate trading signal (Buy/Sell/Hold)"""
        raise NotImplementedError
    
    def get_parameters(self) -> Dict:
        """Get strategy parameters for optimization"""
        return {}

class TrendFollowingStrategy(BaseStrategy):
    """Simple trend following strategy using moving averages"""
    
    def __init__(self, short_window=10, long_window=20):
        super().__init__("Trend Following")
        self.short_window = short_window
        self.long_window = long_window
    
    def generate_signal(self, df: pd.DataFrame, index: int) -> str:
        if index < self.long_window:
            return 'Hold'
        
        short_ma = df['Close'].iloc[index-self.short_window:index].mean()
        long_ma = df['Close'].iloc[index-self.long_window:index].mean()
        
        if short_ma > long_ma:
            return 'Buy'
        elif short_ma < long_ma:
            return 'Sell'
        else:
            return 'Hold'
    
    def get_parameters(self) -> Dict:
        return {
            'short_window': self.short_window,
            'long_window': self.long_window
        }

class RSIStrategy(BaseStrategy):
    """RSI-based mean reversion strategy"""
    
    def __init__(self, rsi_period=14, oversold=30, overbought=70):
        super().__init__("RSI Mean Reversion")
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signal(self, df: pd.DataFrame, index: int) -> str:
        if index < self.rsi_period:
            return 'Hold'
        
        rsi = df['RSI_14'].iloc[index]
        
        if rsi < self.oversold:
            return 'Buy'
        elif rsi > self.overbought:
            return 'Sell'
        else:
            return 'Hold'
    
    def get_parameters(self) -> Dict:
        return {
            'rsi_period': self.rsi_period,
            'oversold': self.oversold,
            'overbought': self.overbought
        }

class MACDStrategy(BaseStrategy):
    """MACD crossover strategy"""
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        super().__init__("MACD Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def generate_signal(self, df: pd.DataFrame, index: int) -> str:
        if index < self.slow_period + self.signal_period:
            return 'Hold'
        
        macd = df['MACD'].iloc[index]
        macd_signal = df['MACD_Signal'].iloc[index]
        macd_prev = df['MACD'].iloc[index-1]
        macd_signal_prev = df['MACD_Signal'].iloc[index-1]
        
        # MACD crosses above signal line
        if macd > macd_signal and macd_prev <= macd_signal_prev:
            return 'Buy'
        # MACD crosses below signal line
        elif macd < macd_signal and macd_prev >= macd_signal_prev:
            return 'Sell'
        else:
            return 'Hold'
    
    def get_parameters(self) -> Dict:
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period
        }

class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands mean reversion strategy"""
    
    def __init__(self, window=20, std_dev=2):
        super().__init__("Bollinger Bands")
        self.window = window
        self.std_dev = std_dev
    
    def generate_signal(self, df: pd.DataFrame, index: int) -> str:
        if index < self.window:
            return 'Hold'
        
        close = df['Close'].iloc[index]
        bb_high = df['BB_High'].iloc[index]
        bb_low = df['BB_Low'].iloc[index]
        
        # Price touches lower band
        if close <= bb_low:
            return 'Buy'
        # Price touches upper band
        elif close >= bb_high:
            return 'Sell'
        else:
            return 'Hold'
    
    def get_parameters(self) -> Dict:
        return {
            'window': self.window,
            'std_dev': self.std_dev
        }

class VolumePriceStrategy(BaseStrategy):
    """Volume and price action strategy"""
    
    def __init__(self, volume_threshold=1.5, price_change_threshold=0.02):
        super().__init__("Volume Price Action")
        self.volume_threshold = volume_threshold
        self.price_change_threshold = price_change_threshold
    
    def generate_signal(self, df: pd.DataFrame, index: int) -> str:
        if index < 20:
            return 'Hold'
        
        current_volume = df['Volume'].iloc[index]
        avg_volume = df['Volume'].iloc[index-20:index].mean()
        volume_ratio = current_volume / avg_volume
        
        current_price = df['Close'].iloc[index]
        prev_price = df['Close'].iloc[index-1]
        price_change = (current_price - prev_price) / prev_price
        
        # High volume with price increase
        if volume_ratio > self.volume_threshold and price_change > self.price_change_threshold:
            return 'Buy'
        # High volume with price decrease
        elif volume_ratio > self.volume_threshold and price_change < -self.price_change_threshold:
            return 'Sell'
        else:
            return 'Hold'
    
    def get_parameters(self) -> Dict:
        return {
            'volume_threshold': self.volume_threshold,
            'price_change_threshold': self.price_change_threshold
        }

class MultiSignalStrategy(BaseStrategy):
    """Combines multiple signals for confirmation"""
    
    def __init__(self):
        super().__init__("Multi-Signal")
        self.trend_strategy = TrendFollowingStrategy()
        self.rsi_strategy = RSIStrategy()
        self.macd_strategy = MACDStrategy()
    
    def generate_signal(self, df: pd.DataFrame, index: int) -> str:
        signals = []
        
        # Get signals from all strategies
        trend_signal = self.trend_strategy.generate_signal(df, index)
        rsi_signal = self.rsi_strategy.generate_signal(df, index)
        macd_signal = self.macd_strategy.generate_signal(df, index)
        
        signals.extend([trend_signal, rsi_signal, macd_signal])
        
        # Count buy and sell signals
        buy_count = signals.count('Buy')
        sell_count = signals.count('Sell')
        
        # Require at least 2 confirming signals
        if buy_count >= 2:
            return 'Buy'
        elif sell_count >= 2:
            return 'Sell'
        else:
            return 'Hold'
    
    def get_parameters(self) -> Dict:
        return {
            'trend_params': self.trend_strategy.get_parameters(),
            'rsi_params': self.rsi_strategy.get_parameters(),
            'macd_params': self.macd_strategy.get_parameters()
        }

class MLStrategy(BaseStrategy):
    """Machine learning-based strategy using Random Forest"""
    def __init__(self):
        super().__init__("ML Random Forest")
        self.model = None
        self.scaler = None
        self.features = [
            'SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_High', 'BB_Low', 'STOCH_K', 'STOCH_D', 'CCI', 'ADX', 'WILLR', 'OBV', 'Volume'
        ]
        self.is_trained = False

    def train(self, df: pd.DataFrame):
        # Prepare features and target
        df = df.copy().dropna()
        X = df[self.features]
        y = (df['Close'].shift(-1) > df['Close']).astype(int)  # 1=Buy, 0=Sell/Hold
        y = y[:-1]
        X = X[:-1]
        # Standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def generate_signal(self, df: pd.DataFrame, index: int) -> str:
        if not self.is_trained:
            self.train(df)
        if index < 1 or index >= len(df):
            return 'Hold'
        row = df.iloc[index][self.features].values.reshape(1, -1)
        row_scaled = self.scaler.transform(row)
        pred = self.model.predict(row_scaled)[0]
        # Optionally, use predict_proba for confidence threshold
        if pred == 1:
            return 'Buy'
        else:
            return 'Sell'

def get_all_strategies() -> List[BaseStrategy]:
    """Get all available strategies"""
    return [
        TrendFollowingStrategy(),
        RSIStrategy(),
        MACDStrategy(),
        BollingerBandsStrategy(),
        VolumePriceStrategy(),
        MultiSignalStrategy(),
        MLStrategy()
    ]

def optimize_strategy_parameters(strategy: BaseStrategy, df: pd.DataFrame, 
                                param_ranges: Dict, metric='sharpe_ratio') -> Tuple[Dict, float]:
    """Optimize strategy parameters using grid search"""
    best_params = {}
    best_score = float('-inf')
    
    # Generate parameter combinations
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    from itertools import product
    for param_combo in product(*param_values):
        # Set parameters
        for name, value in zip(param_names, param_combo):
            setattr(strategy, name, value)
        
        # Run backtest
        from backtest import run_backtest
        result = run_backtest(df)
        
        # Calculate score
        if metric == 'sharpe_ratio':
            score = result.sharpe_ratio
        elif metric == 'total_return':
            score = result.total_return
        elif metric == 'profit_factor':
            score = result.profit_factor
        else:
            score = result.total_return
        
        if score > best_score:
            best_score = score
            best_params = dict(zip(param_names, param_combo))
    
    return best_params, best_score 