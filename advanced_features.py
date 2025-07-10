import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class MarketRegimeDetector:
    """Detect market regimes (bull/bear/sideways) using volatility and trend"""
    
    def __init__(self, volatility_window=20, trend_window=50):
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.regimes = []
    
    def detect_regime(self, df: pd.DataFrame) -> str:
        """Detect current market regime"""
        if len(df) < self.trend_window:
            return 'unknown'
        
        # Calculate volatility
        returns = df['Close'].pct_change()
        volatility = returns.rolling(self.volatility_window).std().iloc[-1]
        
        # Calculate trend strength
        current_price = df['Close'].iloc[-1]
        trend_start_price = df['Close'].iloc[-self.trend_window]
        trend_change = (current_price - trend_start_price) / trend_start_price
        
        # Calculate ADX for trend strength
        adx = df['ADX'].iloc[-1] if 'ADX' in df.columns else 25
        
        # Regime classification
        if adx > 25 and trend_change > 0.05:  # Strong uptrend
            regime = 'bull'
        elif adx > 25 and trend_change < -0.05:  # Strong downtrend
            regime = 'bear'
        elif volatility > 0.02:  # High volatility
            regime = 'volatile'
        else:
            regime = 'sideways'
        
        self.regimes.append(regime)
        return regime
    
    def get_regime_history(self) -> List[str]:
        return self.regimes

class WalkForwardValidator:
    """Walk-forward validation for robust strategy testing"""
    
    def __init__(self, train_size=0.6, test_size=0.2, step_size=0.1):
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.results = []
    
    def validate_strategy(self, df: pd.DataFrame, strategy_class, param_space: Dict = None):
        """Run walk-forward validation"""
        total_length = len(df)
        train_length = int(total_length * self.train_size)
        test_length = int(total_length * self.test_size)
        step_length = int(total_length * self.step_size)
        
        print(f"Walk-forward validation: {total_length} days, train: {train_length}, test: {test_length}, step: {step_length}")
        
        for start_idx in range(0, total_length - train_length - test_length, step_length):
            # Define windows
            train_end = start_idx + train_length
            test_end = train_end + test_length
            
            # Split data
            train_data = df.iloc[start_idx:train_end].copy()
            test_data = df.iloc[train_end:test_end].copy()
            
            # Optimize on training data if param_space provided
            if param_space:
                from ai_optimizer import StrategyOptimizer
                strategy = strategy_class()
                optimizer = StrategyOptimizer(strategy, param_space, 'sharpe_ratio')
                best_params, _ = optimizer.optimize(train_data, n_calls=20)
                # Apply best params
                for name, value in best_params.items():
                    setattr(strategy, name, value)
            else:
                strategy = strategy_class()
            
            # Test on out-of-sample data
            from backtest import run_backtest
            result = run_backtest(test_data)
            
            self.results.append({
                'start_date': test_data.index[0],
                'end_date': test_data.index[-1],
                'total_return': result.total_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'num_trades': result.num_trades
            })
        
        return self.get_summary()
    
    def get_summary(self) -> Dict:
        """Get summary statistics of walk-forward results"""
        if not self.results:
            return {}
        
        returns = [r['total_return'] for r in self.results]
        sharpes = [r['sharpe_ratio'] for r in self.results]
        drawdowns = [r['max_drawdown'] for r in self.results]
        win_rates = [r['win_rate'] for r in self.results]
        
        summary = {
            'num_periods': len(self.results),
            'avg_return': np.mean(returns),
            'std_return': np.std(returns),
            'avg_sharpe': np.mean(sharpes),
            'avg_drawdown': np.mean(drawdowns),
            'avg_win_rate': np.mean(win_rates),
            'positive_periods': sum(1 for r in returns if r > 0),
            'best_period': max(returns),
            'worst_period': min(returns)
        }
        
        return summary

class EnsembleStrategy:
    """Combine multiple strategies using voting or weighted averaging"""
    
    def __init__(self, strategies: List, weights: List[float] = None, method='voting'):
        self.strategies = strategies
        self.weights = weights or [1.0] * len(strategies)
        self.method = method  # 'voting' or 'weighted'
    
    def generate_signal(self, df: pd.DataFrame, index: int) -> str:
        """Generate ensemble signal"""
        signals = []
        for strategy in self.strategies:
            signal = strategy.generate_signal(df, index)
            signals.append(signal)
        
        if self.method == 'voting':
            # Simple majority voting
            buy_votes = signals.count('Buy')
            sell_votes = signals.count('Sell')
            
            if buy_votes > sell_votes:
                return 'Buy'
            elif sell_votes > buy_votes:
                return 'Sell'
            else:
                return 'Hold'
        
        elif self.method == 'weighted':
            # Weighted signal scoring
            score = 0
            for signal, weight in zip(signals, self.weights):
                if signal == 'Buy':
                    score += weight
                elif signal == 'Sell':
                    score -= weight
            
            if score > 0.5:
                return 'Buy'
            elif score < -0.5:
                return 'Sell'
            else:
                return 'Hold'
        
        return 'Hold'

class AdaptiveStrategy:
    """Strategy that adapts parameters based on market regime"""
    
    def __init__(self, base_strategy, regime_detector: MarketRegimeDetector):
        self.base_strategy = base_strategy
        self.regime_detector = regime_detector
        self.regime_params = {
            'bull': {'risk_multiplier': 1.2, 'position_size_multiplier': 1.1},
            'bear': {'risk_multiplier': 0.8, 'position_size_multiplier': 0.9},
            'volatile': {'risk_multiplier': 0.6, 'position_size_multiplier': 0.7},
            'sideways': {'risk_multiplier': 1.0, 'position_size_multiplier': 1.0}
        }
    
    def generate_signal(self, df: pd.DataFrame, index: int) -> str:
        """Generate signal with regime adaptation"""
        # Detect current regime
        current_regime = self.regime_detector.detect_regime(df.iloc[:index+1])
        
        # Get regime-specific parameters
        params = self.regime_params.get(current_regime, self.regime_params['sideways'])
        
        # Apply regime-specific adjustments
        # (This would modify the strategy's behavior based on regime)
        
        return self.base_strategy.generate_signal(df, index)

class FeatureEngineer:
    """Advanced feature engineering for ML strategies"""
    
    @staticmethod
    def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced technical features"""
        df = df.copy()
        
        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_Change_2d'] = df['Close'].pct_change(2)
        df['Price_Change_5d'] = df['Close'].pct_change(5)
        
        # Volatility features
        df['Volatility_5d'] = df['Price_Change'].rolling(5).std()
        df['Volatility_20d'] = df['Price_Change'].rolling(20).std()
        
        # Momentum features
        df['Momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
        df['Momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Volume features
        df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
        df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio_5'] = df['Volume'] / df['Volume_MA_5']
        df['Volume_Ratio_20'] = df['Volume'] / df['Volume_MA_20']
        
        # Technical indicator features
        if 'RSI_14' in df.columns:
            df['RSI_Change'] = df['RSI_14'].diff()
            df['RSI_MA'] = df['RSI_14'].rolling(10).mean()
        
        if 'MACD' in df.columns:
            df['MACD_Change'] = df['MACD'].diff()
            df['MACD_MA'] = df['MACD'].rolling(10).mean()
        
        # Support/Resistance features
        df['High_20'] = df['High'].rolling(20).max()
        df['Low_20'] = df['Low'].rolling(20).min()
        df['Price_Position'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
        
        # Time-based features
        df['Day_of_Week'] = df.index.dayofweek
        df['Month'] = df.index.month
        df['Quarter'] = df.index.quarter
        
        return df

class PerformanceAnalyzer:
    """Advanced performance analysis and visualization"""
    
    @staticmethod
    def calculate_advanced_metrics(equity_curve: List[float]) -> Dict:
        """Calculate advanced performance metrics"""
        if len(equity_curve) < 2:
            return {}
        
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Basic metrics
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        # Maximum drawdown
        peak = equity_curve[0]
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
        
        # Calmar ratio
        calmar_ratio = total_return / max_dd if max_dd > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = np.mean(returns) / downside_deviation if downside_deviation > 0 else 0
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5)
        
        # Conditional Value at Risk (Expected Shortfall)
        cvar_95 = np.mean(returns[returns <= var_95])
        
        return {
            'total_return': total_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_dd * 100,
            'var_95': var_95 * 100,
            'cvar_95': cvar_95 * 100
        }

def run_advanced_analysis(df: pd.DataFrame, strategy_class, param_space: Dict = None):
    """Run comprehensive advanced analysis"""
    print("="*80)
    print("ADVANCED ANALYSIS")
    print("="*80)
    
    # 1. Feature Engineering
    print("\n1. Adding advanced features...")
    df_enhanced = FeatureEngineer.add_advanced_features(df)
    
    # 2. Market Regime Detection
    print("\n2. Detecting market regimes...")
    regime_detector = MarketRegimeDetector()
    current_regime = regime_detector.detect_regime(df_enhanced)
    print(f"Current market regime: {current_regime}")
    
    # 3. Walk-Forward Validation
    print("\n3. Running walk-forward validation...")
    wf_validator = WalkForwardValidator()
    wf_results = wf_validator.validate_strategy(df_enhanced, strategy_class, param_space)
    
    print("\nWalk-Forward Results:")
    for key, value in wf_results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # 4. Ensemble Strategy
    print("\n4. Testing ensemble strategy...")
    from strategies import TrendFollowingStrategy, RSIStrategy, MACDStrategy
    strategies = [TrendFollowingStrategy(), RSIStrategy(), MACDStrategy()]
    ensemble = EnsembleStrategy(strategies, method='voting')
    
    # Create ensemble signals
    ensemble_data = df_enhanced.copy()
    ensemble_data['Ensemble_Signal'] = [
        ensemble.generate_signal(ensemble_data, i) for i in range(len(ensemble_data))
    ]
    
    # Run ensemble backtest
    from backtest import run_backtest
    ensemble_result = run_backtest(ensemble_data)
    
    print("\nEnsemble Strategy Results:")
    print(f"  Total Return: {ensemble_result.total_return:.2f}%")
    print(f"  Sharpe Ratio: {ensemble_result.sharpe_ratio:.2f}")
    print(f"  Max Drawdown: {ensemble_result.max_drawdown:.2f}%")
    
    return {
        'regime': current_regime,
        'walk_forward': wf_results,
        'ensemble': ensemble_result
    } 