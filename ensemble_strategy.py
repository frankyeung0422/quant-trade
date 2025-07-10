import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from strategies import get_all_strategies, BaseStrategy
from adaptive_strategy import AdaptiveStrategy, MultiTimeframeStrategy
from improved_ml import ImprovedMLStrategy

class EnsembleStrategy:
    """Ensemble strategy that combines multiple strategies with dynamic capital allocation"""
    
    def __init__(self, strategies: List[BaseStrategy] = None, weights: List[float] = None):
        self.strategies = strategies or []
        self.weights = weights or []
        self.signal_history = []
        self.performance_history = []
        
    def add_strategy(self, strategy: BaseStrategy, weight: float = 1.0):
        """Add a strategy to the ensemble"""
        self.strategies.append(strategy)
        self.weights.append(weight)
        
    def normalize_weights(self):
        """Normalize weights to sum to 1"""
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
            
    def get_ensemble_signal(self, df: pd.DataFrame, index: int) -> Tuple[str, float]:
        """Get ensemble signal and confidence score"""
        if not self.strategies:
            return 'Hold', 0.0
            
        signals = []
        confidences = []
        
        for strategy in self.strategies:
            try:
                signal = strategy.generate_signal(df, index)
                signals.append(signal)
                
                # Calculate confidence based on strategy type and recent performance
                confidence = self._calculate_signal_confidence(strategy, df, index, signal)
                confidences.append(confidence)
            except Exception as e:
                print(f"Error in strategy {strategy.name}: {e}")
                signals.append('Hold')
                confidences.append(0.0)
        
        # Weighted voting
        buy_score = 0.0
        sell_score = 0.0
        
        for i, signal in enumerate(signals):
            weight = self.weights[i] if i < len(self.weights) else 1.0
            confidence = confidences[i] if i < len(confidences) else 1.0
            
            if signal == 'Buy':
                buy_score += weight * confidence
            elif signal == 'Sell':
                sell_score += weight * confidence
        
        # Determine final signal
        if buy_score > sell_score and buy_score > 0.3:  # Threshold for buy
            return 'Buy', buy_score
        elif sell_score > buy_score and sell_score > 0.3:  # Threshold for sell
            return 'Sell', sell_score
        else:
            return 'Hold', max(buy_score, sell_score)
    
    def _calculate_signal_confidence(self, strategy: BaseStrategy, df: pd.DataFrame, index: int, signal: str) -> float:
        """Calculate confidence score for a strategy's signal"""
        if index < 50:
            return 0.5
            
        # Base confidence
        confidence = 0.5
        
        # Adjust based on recent performance (if available)
        if hasattr(strategy, 'recent_performance'):
            confidence += strategy.recent_performance * 0.3
            
        # Adjust based on market conditions
        if signal == 'Buy':
            # Check if market conditions favor buying
            rsi = df['RSI_14'].iloc[index] if 'RSI_14' in df.columns else 50
            macd = df['MACD'].iloc[index] if 'MACD' in df.columns else 0
            sma_20 = df['SMA_20'].iloc[index] if 'SMA_20' in df.columns else df['Close'].iloc[index]
            current_price = df['Close'].iloc[index]
            
            # Favorable conditions for buying
            if rsi < 70 and macd > 0 and current_price > sma_20:
                confidence += 0.2
            elif rsi > 80 or macd < -1:
                confidence -= 0.2
                
        elif signal == 'Sell':
            # Check if market conditions favor selling
            rsi = df['RSI_14'].iloc[index] if 'RSI_14' in df.columns else 50
            macd = df['MACD'].iloc[index] if 'MACD' in df.columns else 0
            sma_20 = df['SMA_20'].iloc[index] if 'SMA_20' in df.columns else df['Close'].iloc[index]
            current_price = df['Close'].iloc[index]
            
            # Favorable conditions for selling
            if rsi > 30 and macd < 0 and current_price < sma_20:
                confidence += 0.2
            elif rsi < 20 or macd > 1:
                confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def update_performance(self, strategy_name: str, performance: float):
        """Update performance history for a strategy"""
        self.performance_history.append({
            'strategy': strategy_name,
            'performance': performance,
            'timestamp': pd.Timestamp.now()
        })
        
        # Update strategy weights based on recent performance
        self._update_weights()
    
    def _update_weights(self):
        """Update strategy weights based on recent performance"""
        if len(self.performance_history) < 10:
            return
            
        # Calculate recent performance for each strategy
        recent_performance = {}
        for entry in self.performance_history[-10:]:
            strategy = entry['strategy']
            if strategy not in recent_performance:
                recent_performance[strategy] = []
            recent_performance[strategy].append(entry['performance'])
        
        # Update weights based on average performance
        for i, strategy in enumerate(self.strategies):
            if strategy.name in recent_performance:
                avg_performance = np.mean(recent_performance[strategy.name])
                # Softmax-like weight update
                self.weights[i] = max(0.1, min(2.0, 1.0 + avg_performance * 0.1))
        
        self.normalize_weights()

def create_default_ensemble() -> EnsembleStrategy:
    """Create a default ensemble with multiple strategies"""
    ensemble = EnsembleStrategy()
    
    # Add classic strategies
    from strategies import TrendFollowingStrategy, RSIStrategy, MACDStrategy, BollingerBandsStrategy
    ensemble.add_strategy(TrendFollowingStrategy(short_window=10, long_window=20), weight=1.0)
    ensemble.add_strategy(RSIStrategy(rsi_period=14, oversold=30, overbought=70), weight=1.0)
    ensemble.add_strategy(MACDStrategy(fast_period=12, slow_period=26, signal_period=9), weight=1.0)
    ensemble.add_strategy(BollingerBandsStrategy(window=20, std_dev=2), weight=0.8)
    
    # Add adaptive strategy
    ensemble.add_strategy(AdaptiveStrategy(), weight=1.2)
    
    # Add multi-timeframe strategy
    ensemble.add_strategy(MultiTimeframeStrategy(), weight=1.0)
    
    # Add ML strategy
    ensemble.add_strategy(ImprovedMLStrategy(), weight=1.1)
    
    ensemble.normalize_weights()
    return ensemble

def run_ensemble_backtest(df: pd.DataFrame, ensemble: EnsembleStrategy = None):
    """Run backtest with ensemble strategy"""
    if ensemble is None:
        ensemble = create_default_ensemble()
    
    # Generate ensemble signals
    signals = []
    confidences = []
    
    for i in range(len(df)):
        if i < 50:
            signals.append('Hold')
            confidences.append(0.0)
        else:
            signal, confidence = ensemble.get_ensemble_signal(df, i)
            signals.append(signal)
            confidences.append(confidence)
    
    # Add signals to dataframe
    df_copy = df.copy()
    df_copy['Strategy_Signal'] = signals
    df_copy['Signal_Confidence'] = confidences
    
    # Run improved backtest with ensemble signals
    from improved_backtest import run_improved_backtest
    return run_improved_backtest(df_copy, allow_short=True) 