import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class AdaptiveStrategy:
    """Adaptive strategy that adjusts parameters based on market conditions"""
    
    def __init__(self):
        self.name = 'AdaptiveStrategy'
        self.regime = 'unknown'
        self.volatility_regime = 'normal'
        self.trend_strength = 0
        
    def detect_market_regime(self, df: pd.DataFrame, index: int) -> Dict:
        """Detect current market regime"""
        if index < 50:
            return {'regime': 'unknown', 'volatility': 'normal', 'trend_strength': 0}
        
        # Calculate volatility
        returns = df['Close'].pct_change()
        volatility = returns.iloc[index-20:index].std()
        avg_volatility = returns.iloc[index-50:index].std()
        
        # Calculate trend strength
        sma_20 = df['SMA_20'].iloc[index]
        sma_50 = df['SMA_50'].iloc[index] if 'SMA_50' in df.columns else sma_20
        current_price = df['Close'].iloc[index]
        
        # ADX for trend strength
        adx = df['ADX'].iloc[index] if 'ADX' in df.columns else 25
        
        # Determine volatility regime
        if volatility > avg_volatility * 1.5:
            volatility_regime = 'high'
        elif volatility < avg_volatility * 0.7:
            volatility_regime = 'low'
        else:
            volatility_regime = 'normal'
        
        # Determine market regime
        trend_strength = 0
        if adx > 25:
            if current_price > sma_20 and sma_20 > sma_50:
                regime = 'strong_bull'
                trend_strength = 2
            elif current_price < sma_20 and sma_20 < sma_50:
                regime = 'strong_bear'
                trend_strength = -2
            else:
                regime = 'trending'
                trend_strength = 1 if current_price > sma_20 else -1
        else:
            if volatility > avg_volatility * 1.2:
                regime = 'volatile_sideways'
            else:
                regime = 'sideways'
        
        return {
            'regime': regime,
            'volatility': volatility_regime,
            'trend_strength': trend_strength,
            'adx': adx,
            'volatility': volatility
        }
    
    def get_adaptive_parameters(self, regime_info: Dict) -> Dict:
        """Get adaptive parameters based on market regime"""
        regime = regime_info['regime']
        volatility = regime_info['volatility']
        trend_strength = regime_info['trend_strength']
        
        # Base parameters
        params = {
            'risk_per_trade': 0.01,
            'stop_loss_multiplier': 1.5,
            'take_profit_multiplier': 2.0,
            'position_size_multiplier': 1.0,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_threshold': 1.05,
            'min_confirmations': 4
        }
        
        # Adjust based on regime
        if regime == 'strong_bull':
            params['risk_per_trade'] = 0.015
            params['position_size_multiplier'] = 1.2
            params['rsi_oversold'] = 35
            params['rsi_overbought'] = 75
            params['min_confirmations'] = 3
        elif regime == 'strong_bear':
            params['risk_per_trade'] = 0.008
            params['position_size_multiplier'] = 0.8
            params['rsi_oversold'] = 25
            params['rsi_overbought'] = 65
            params['min_confirmations'] = 5
        elif regime == 'volatile_sideways':
            params['risk_per_trade'] = 0.006
            params['stop_loss_multiplier'] = 2.0
            params['take_profit_multiplier'] = 1.5
            params['position_size_multiplier'] = 0.7
            params['min_confirmations'] = 6
        elif regime == 'sideways':
            params['risk_per_trade'] = 0.01
            params['stop_loss_multiplier'] = 1.8
            params['take_profit_multiplier'] = 1.8
            params['min_confirmations'] = 5
        
        # Adjust based on volatility
        if volatility == 'high':
            params['stop_loss_multiplier'] *= 1.2
            params['position_size_multiplier'] *= 0.8
        elif volatility == 'low':
            params['stop_loss_multiplier'] *= 0.8
            params['position_size_multiplier'] *= 1.1
        
        return params
    
    def generate_adaptive_signal(self, df: pd.DataFrame, index: int) -> str:
        """Generate adaptive trading signal"""
        if index < 50:
            return 'Hold'
        
        # Detect market regime
        regime_info = self.detect_market_regime(df, index)
        params = self.get_adaptive_parameters(regime_info)
        
        # Get current indicators
        current_price = df['Close'].iloc[index]
        rsi = df['RSI_14'].iloc[index]
        macd = df['MACD'].iloc[index]
        macd_signal = df['MACD_Signal'].iloc[index]
        volume = df['Volume'].iloc[index]
        avg_volume = df['Volume'].iloc[index-20:index].mean()
        
        # Calculate confirmations based on regime
        confirmations = []
        
        # Volume confirmation
        volume_confirmed = volume > avg_volume * params['volume_threshold']
        confirmations.append(volume_confirmed)
        
        # MACD confirmation
        macd_confirmed = macd > macd_signal and macd > 0
        confirmations.append(macd_confirmed)
        
        # RSI confirmation (adaptive)
        rsi_confirmed = params['rsi_oversold'] < rsi < params['rsi_overbought']
        confirmations.append(rsi_confirmed)
        
        # Trend confirmations
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            sma_20 = df['SMA_20'].iloc[index]
            sma_50 = df['SMA_50'].iloc[index]
            price_above_sma20 = current_price > sma_20
            sma_bullish = sma_20 > sma_50
            confirmations.extend([price_above_sma20, sma_bullish])
        
        # Momentum confirmation
        if index > 0:
            momentum_positive = df['Close'].iloc[index] > df['Close'].iloc[index-1]
            confirmations.append(momentum_positive)
        
        # Count confirmations
        num_confirmations = sum(confirmations)
        
        # Generate signal based on regime and confirmations
        if regime_info['regime'] in ['strong_bull', 'trending'] and num_confirmations >= params['min_confirmations']:
            return 'Buy'
        elif regime_info['regime'] in ['strong_bear'] and num_confirmations >= params['min_confirmations']:
            return 'Sell'
        elif regime_info['regime'] in ['volatile_sideways', 'sideways']:
            # More conservative in sideways markets
            if num_confirmations >= params['min_confirmations'] + 1:
                return 'Buy'
            elif rsi > params['rsi_overbought'] and macd < macd_signal:
                return 'Sell'
        
        return 'Hold'

    def generate_signal(self, df: pd.DataFrame, index: int) -> str:
        return self.generate_adaptive_signal(df, index)

class MultiTimeframeStrategy:
    """Strategy that combines signals from multiple timeframes"""
    
    def __init__(self):
        self.name = 'MultiTimeframeStrategy'
        self.timeframes = ['daily', 'weekly']
        
    def calculate_weekly_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weekly indicators"""
        # Resample to weekly data
        weekly = df.resample('W').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        # Calculate weekly indicators
        weekly['SMA_20_W'] = weekly['Close'].rolling(20).mean()
        weekly['RSI_14_W'] = self._calculate_rsi(weekly['Close'])
        weekly['MACD_W'] = self._calculate_macd(weekly['Close'])
        
        return weekly
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        return macd - signal_line
    
    def generate_multi_timeframe_signal(self, df: pd.DataFrame, index: int) -> str:
        """Generate signal combining daily and weekly timeframes"""
        if index < 50:
            return 'Hold'
        
        # Daily signals
        daily_rsi = df['RSI_14'].iloc[index]
        daily_macd = df['MACD'].iloc[index]
        daily_macd_signal = df['MACD_Signal'].iloc[index]
        
        # Weekly signals (approximate)
        weekly_index = index // 5  # Approximate weekly index
        if weekly_index < len(df):
            weekly_rsi = df['RSI_14'].iloc[weekly_index]
            weekly_macd = df['MACD'].iloc[weekly_index]
        else:
            weekly_rsi = daily_rsi
            weekly_macd = daily_macd
        
        # Combine signals
        daily_bullish = daily_rsi < 70 and daily_macd > daily_macd_signal
        weekly_bullish = weekly_rsi < 70 and weekly_macd > 0
        
        daily_bearish = daily_rsi > 30 and daily_macd < daily_macd_signal
        weekly_bearish = weekly_rsi > 30 and weekly_macd < 0
        
        # Generate combined signal
        if daily_bullish and weekly_bullish:
            return 'Buy'
        elif daily_bearish and weekly_bearish:
            return 'Sell'
        elif daily_bullish or weekly_bullish:
            return 'Buy'  # Weak buy signal
        elif daily_bearish or weekly_bearish:
            return 'Sell'  # Weak sell signal
        
        return 'Hold'

    def generate_signal(self, df: pd.DataFrame, index: int) -> str:
        return self.generate_multi_timeframe_signal(df, index)

def run_adaptive_backtest(df: pd.DataFrame, strategy_type='adaptive'):
    """Run backtest with adaptive strategy"""
    if strategy_type == 'adaptive':
        strategy = AdaptiveStrategy()
        df_copy = df.copy()
        df_copy['Strategy_Signal'] = [
            strategy.generate_adaptive_signal(df_copy, i) for i in range(len(df_copy))
        ]
    elif strategy_type == 'multi_timeframe':
        strategy = MultiTimeframeStrategy()
        df_copy = df.copy()
        df_copy['Strategy_Signal'] = [
            strategy.generate_multi_timeframe_signal(df_copy, i) for i in range(len(df_copy))
        ]
    else:
        # Use default analysis
        from analysis import generate_analysis
        signals = []
        for i in range(len(df)):
            if i < 50:
                signals.append('Hold')
            else:
                sub_df = df.iloc[:i+1]
                _, signal = generate_analysis(sub_df)
                signals.append(signal)
        df_copy = df.copy()
        df_copy['Strategy_Signal'] = signals
    
    # Run improved backtest
    from improved_backtest import run_improved_backtest
    return run_improved_backtest(df_copy) 