import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ImprovedMLStrategy:
    """Improved ML strategy with better target definition and feature engineering"""
    
    def __init__(self, algorithm='random_forest', feature_selection=True, n_features=25):
        self.algorithm = algorithm
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.is_trained = False
        
        # Enhanced feature set
        self.all_features = [
            # Price-based features
            'SMA_20', 'EMA_20', 'Price_Change', 'Price_Change_2d', 'Price_Change_5d',
            'High_20', 'Low_20', 'Price_Position',
            
            # Momentum features
            'RSI_14', 'RSI_Change', 'RSI_MA', 'STOCH_K', 'STOCH_D', 'WILLR',
            'Momentum_5d', 'Momentum_20d',
            
            # Trend features
            'MACD', 'MACD_Signal', 'MACD_Histogram', 'MACD_Change', 'MACD_MA',
            'ADX', 'CCI',
            
            # Volatility features
            'BB_High', 'BB_Low', 'Volatility_5d', 'Volatility_20d',
            
            # Volume features
            'Volume', 'OBV', 'Volume_MA_5', 'Volume_MA_20', 'Volume_Ratio_5', 'Volume_Ratio_20',
            
            # Time features
            'Day_of_Week', 'Month', 'Quarter',
            
            # Advanced features
            'ATR', 'ATR_Ratio', 'Price_Volatility_Ratio', 'Volume_Price_Trend'
        ]
    
    def _get_model(self):
        """Get the specified ML model with optimized parameters"""
        if self.algorithm == 'random_forest':
            return RandomForestClassifier(n_estimators=200, max_depth=15, 
                                        min_samples_split=5, min_samples_leaf=2,
                                        random_state=42, n_jobs=-1)
        elif self.algorithm == 'gradient_boosting':
            return GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                            max_depth=6, subsample=0.8, random_state=42)
        elif self.algorithm == 'adaboost':
            return AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
        elif self.algorithm == 'svm':
            return SVC(kernel='rbf', probability=True, random_state=42, C=1.0, gamma='scale')
        elif self.algorithm == 'neural_network':
            return MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=1000,
                               alpha=0.001, learning_rate='adaptive', random_state=42)
        else:
            return RandomForestClassifier(n_estimators=200, random_state=42)
    
    def _create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced technical features"""
        df = df.copy()
        
        # Basic features if not present
        if 'Price_Change' not in df.columns:
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_2d'] = df['Close'].pct_change(2)
            df['Price_Change_5d'] = df['Close'].pct_change(5)
            df['Volatility_5d'] = df['Price_Change'].rolling(5).std()
            df['Volatility_20d'] = df['Price_Change'].rolling(20).std()
            df['Momentum_5d'] = df['Close'] / df['Close'].shift(5) - 1
            df['Momentum_20d'] = df['Close'] / df['Close'].shift(20) - 1
            df['Volume_MA_5'] = df['Volume'].rolling(5).mean()
            df['Volume_MA_20'] = df['Volume'].rolling(20).mean()
            df['Volume_Ratio_5'] = df['Volume'] / df['Volume_MA_5']
            df['Volume_Ratio_20'] = df['Volume'] / df['Volume_MA_20']
            
            if 'RSI_14' in df.columns:
                df['RSI_Change'] = df['RSI_14'].diff()
                df['RSI_MA'] = df['RSI_14'].rolling(10).mean()
            
            if 'MACD' in df.columns:
                df['MACD_Change'] = df['MACD'].diff()
                df['MACD_MA'] = df['MACD'].rolling(10).mean()
            
            df['High_20'] = df['High'].rolling(20).max()
            df['Low_20'] = df['Low'].rolling(20).min()
            df['Price_Position'] = (df['Close'] - df['Low_20']) / (df['High_20'] - df['Low_20'])
            
            df['Day_of_Week'] = df.index.dayofweek
            df['Month'] = df.index.month
            df['Quarter'] = df.index.quarter
        
        # Advanced features
        if 'ATR' not in df.columns:
            df['ATR'] = self._calculate_atr(df)
        
        # ATR ratio (current ATR vs average ATR)
        df['ATR_Ratio'] = df['ATR'] / df['ATR'].rolling(20).mean()
        
        # Price volatility ratio
        df['Price_Volatility_Ratio'] = df['Volatility_5d'] / df['Volatility_20d']
        
        # Volume-price trend
        df['Volume_Price_Trend'] = df['Volume'] * df['Price_Change']
        
        # Additional momentum features
        df['RSI_Momentum'] = df['RSI_14'] - df['RSI_14'].shift(5)
        df['MACD_Momentum'] = df['MACD'] - df['MACD'].shift(5)
        
        # Support/Resistance features
        df['Distance_to_High'] = (df['High_20'] - df['Close']) / df['Close']
        df['Distance_to_Low'] = (df['Close'] - df['Low_20']) / df['Close']
        
        return df
    
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
    
    def _create_better_target(self, df: pd.DataFrame, target_type='smart_direction') -> pd.Series:
        """Create better target variables for ML training"""
        if target_type == 'smart_direction':
            # Predict if next 3-day return will be positive and above threshold
            future_3d_return = df['Close'].shift(-3) / df['Close'] - 1
            threshold = 0.02  # 2% threshold
            target = (future_3d_return > threshold).astype(int)
        elif target_type == 'volatility_adjusted':
            # Adjust target based on volatility
            future_return = df['Close'].shift(-1) / df['Close'] - 1
            volatility = df['Volatility_20d']
            threshold = volatility * 0.5  # Dynamic threshold
            target = (future_return > threshold).astype(int)
        elif target_type == 'trend_following':
            # Predict continuation of current trend
            current_trend = (df['Close'] > df['SMA_20']).astype(int)
            future_trend = (df['Close'].shift(-5) > df['SMA_20'].shift(-5)).astype(int)
            target = (current_trend == future_trend).astype(int)
        else:
            # Default: simple direction prediction
            target = (df['Close'].shift(-1) > df['Close']).astype(int)
        
        return target
    
    def train(self, df: pd.DataFrame, target_type='smart_direction'):
        """Train the improved ML model"""
        df = self._create_advanced_features(df)
        
        # Prepare features and target
        available_features = [f for f in self.all_features if f in df.columns]
        X = df[available_features].dropna()
        
        # Create better target
        y = self._create_better_target(df, target_type)
        y = y[X.index]
        y = y[:-3]  # Remove last 3 rows since we need future data
        X = X[:-3]
        
        # Remove rows with NaN values
        valid_indices = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_indices]
        y = y[valid_indices]
        
        if len(X) < 100:
            print("Warning: Insufficient data for training")
            return
        
        # Feature selection
        if self.feature_selection and len(available_features) > self.n_features:
            self.feature_selector = SelectKBest(score_func=f_classif, k=self.n_features)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_features = [available_features[i] for i in self.feature_selector.get_support(indices=True)]
        else:
            X_selected = X
            self.selected_features = available_features
        
        # Scaling
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Hyperparameter tuning with more conservative parameters
        model = self._get_model()
        if self.algorithm == 'random_forest':
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15, None],
                'min_samples_split': [5, 10],
                'min_samples_leaf': [2, 4]
            }
        elif self.algorithm == 'gradient_boosting':
            param_grid = {
                'n_estimators': [100, 150],
                'learning_rate': [0.05, 0.1],
                'max_depth': [4, 6],
                'subsample': [0.8, 0.9]
            }
        else:
            param_grid = {}
        
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=tscv, scoring='f1', n_jobs=-1)
            grid_search.fit(X_scaled, y)
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
        else:
            self.model = model
            self.model.fit(X_scaled, y)
        
        self.is_trained = True
        
        # Print feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.selected_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"\nTop 10 Most Important Features for {self.algorithm}:")
            print(feature_importance.head(10))
    
    def generate_signal(self, df: pd.DataFrame, index: int) -> str:
        """Generate trading signal with confidence threshold"""
        if not self.is_trained:
            self.train(df)
        
        if index < 50 or index >= len(df):
            return 'Hold'
        
        df = self._create_advanced_features(df)
        
        # Get features for current index
        features = df.iloc[index][self.selected_features].values.reshape(1, -1)
        
        # Handle missing values
        if np.isnan(features).any():
            return 'Hold'
        
        features_scaled = self.scaler.transform(features)
        
        # Get prediction and probability
        pred = self.model.predict(features_scaled)[0]
        prob = self.model.predict_proba(features_scaled)[0]
        
        # Use higher confidence threshold for more conservative signals
        confidence_threshold = 0.65
        
        if pred == 1 and prob[1] > confidence_threshold:
            return 'Buy'
        elif pred == 0 and prob[0] > confidence_threshold:
            return 'Sell'
        else:
            return 'Hold'

def run_improved_ml_comparison(df: pd.DataFrame):
    """Compare improved ML algorithms"""
    algorithms = ['random_forest', 'gradient_boosting', 'adaboost']
    results = {}
    
    print("Comparing Improved ML Algorithms...")
    print("="*60)
    
    for algorithm in algorithms:
        print(f"\nTesting {algorithm}...")
        try:
            ml_strategy = ImprovedMLStrategy(algorithm=algorithm)
            ml_strategy.train(df, target_type='smart_direction')
            
            # Create signals
            ml_data = df.copy()
            ml_data['Strategy_Signal'] = [
                ml_strategy.generate_signal(ml_data, i) for i in range(len(ml_data))
            ]
            
            # Run improved backtest
            from improved_backtest import run_improved_backtest
            result = run_improved_backtest(ml_data)
            results[algorithm] = result
            
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Win Rate: {result.win_rate:.2f}%")
            print(f"  Max Drawdown: {result.max_drawdown:.2f}%")
            
        except Exception as e:
            print(f"  Error with {algorithm}: {str(e)}")
            continue
    
    # Find best algorithm
    if results:
        best_algorithm = max(results.items(), key=lambda x: x[1].sharpe_ratio)
        print(f"\nBest Improved ML Algorithm: {best_algorithm[0]} (Sharpe: {best_algorithm[1].sharpe_ratio:.2f})")
    
    return results 