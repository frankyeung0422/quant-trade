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

class EnhancedMLStrategy:
    """Enhanced ML strategy with multiple algorithms and feature selection"""
    
    def __init__(self, algorithm='random_forest', feature_selection=True, n_features=20):
        self.algorithm = algorithm
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.selected_features = None
        self.is_trained = False
        
        # Define all possible features
        self.all_features = [
            'SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_High', 'BB_Low', 'STOCH_K', 'STOCH_D', 'CCI', 'ADX', 'WILLR', 'OBV',
            'Volume', 'Price_Change', 'Price_Change_2d', 'Price_Change_5d',
            'Volatility_5d', 'Volatility_20d', 'Momentum_5d', 'Momentum_20d',
            'Volume_MA_5', 'Volume_MA_20', 'Volume_Ratio_5', 'Volume_Ratio_20',
            'RSI_Change', 'RSI_MA', 'MACD_Change', 'MACD_MA', 'High_20', 'Low_20',
            'Price_Position', 'Day_of_Week', 'Month', 'Quarter'
        ]
    
    def _get_model(self):
        """Get the specified ML model"""
        if self.algorithm == 'random_forest':
            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        elif self.algorithm == 'gradient_boosting':
            return GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif self.algorithm == 'adaboost':
            return AdaBoostClassifier(n_estimators=100, random_state=42)
        elif self.algorithm == 'svm':
            return SVC(kernel='rbf', probability=True, random_state=42)
        elif self.algorithm == 'neural_network':
            return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        else:
            return RandomForestClassifier(n_estimators=100, random_state=42)
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and engineer features"""
        df = df.copy()
        
        # Add advanced features if not present
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
        
        return df
    
    def train(self, df: pd.DataFrame, target_type='direction'):
        """Train the ML model"""
        df = self._prepare_features(df)
        
        # Prepare features and target
        available_features = [f for f in self.all_features if f in df.columns]
        X = df[available_features].dropna()
        
        if target_type == 'direction':
            # Predict next day direction (1=up, 0=down)
            y = (df['Close'].shift(-1) > df['Close']).astype(int)
        elif target_type == 'return':
            # Predict next day return
            y = (df['Close'].shift(-1) / df['Close'] - 1) > 0.01  # 1% threshold
            y = y.astype(int)
        else:
            raise ValueError("target_type must be 'direction' or 'return'")
        
        # Align X and y
        y = y[X.index]
        y = y[:-1]  # Remove last row since we don't have next day data
        X = X[:-1]
        
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
        
        # Hyperparameter tuning
        model = self._get_model()
        if self.algorithm == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        elif self.algorithm == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
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
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10))
    
    def generate_signal(self, df: pd.DataFrame, index: int) -> str:
        """Generate trading signal"""
        if not self.is_trained:
            self.train(df)
        
        if index < 1 or index >= len(df):
            return 'Hold'
        
        df = self._prepare_features(df)
        row = df.iloc[index][self.selected_features].values.reshape(1, -1)
        
        # Handle missing values
        if np.isnan(row).any():
            return 'Hold'
        
        row_scaled = self.scaler.transform(row)
        
        # Get prediction and probability
        pred = self.model.predict(row_scaled)[0]
        prob = self.model.predict_proba(row_scaled)[0]
        
        # Use probability threshold for more conservative signals
        confidence_threshold = 0.6
        if pred == 1 and prob[1] > confidence_threshold:
            return 'Buy'
        elif pred == 0 and prob[0] > confidence_threshold:
            return 'Sell'
        else:
            return 'Hold'

class EnsembleMLStrategy:
    """Ensemble of multiple ML models"""
    
    def __init__(self, algorithms=['random_forest', 'gradient_boosting', 'adaboost']):
        self.algorithms = algorithms
        self.models = {}
        self.is_trained = False
    
    def train(self, df: pd.DataFrame):
        """Train all models"""
        for algorithm in self.algorithms:
            print(f"Training {algorithm}...")
            model = EnhancedMLStrategy(algorithm=algorithm)
            model.train(df)
            self.models[algorithm] = model
        self.is_trained = True
    
    def generate_signal(self, df: pd.DataFrame, index: int) -> str:
        """Generate ensemble signal"""
        if not self.is_trained:
            self.train(df)
        
        signals = []
        for model in self.models.values():
            signal = model.generate_signal(df, index)
            signals.append(signal)
        
        # Majority voting
        buy_votes = signals.count('Buy')
        sell_votes = signals.count('Sell')
        
        if buy_votes > sell_votes:
            return 'Buy'
        elif sell_votes > buy_votes:
            return 'Sell'
        else:
            return 'Hold'

def run_ml_comparison(df: pd.DataFrame):
    """Compare different ML algorithms"""
    algorithms = ['random_forest', 'gradient_boosting', 'adaboost', 'svm', 'neural_network']
    results = {}
    
    print("Comparing ML Algorithms...")
    print("="*60)
    
    for algorithm in algorithms:
        print(f"\nTesting {algorithm}...")
        try:
            ml_strategy = EnhancedMLStrategy(algorithm=algorithm)
            ml_strategy.train(df)
            
            # Create signals
            ml_data = df.copy()
            ml_data['Strategy_Signal'] = [
                ml_strategy.generate_signal(ml_data, i) for i in range(len(ml_data))
            ]
            
            # Run backtest
            from backtest import run_backtest
            result = run_backtest(ml_data)
            results[algorithm] = result
            
            print(f"  Total Return: {result.total_return:.2f}%")
            print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
            print(f"  Win Rate: {result.win_rate:.2f}%")
            
        except Exception as e:
            print(f"  Error with {algorithm}: {str(e)}")
            continue
    
    # Find best algorithm
    if results:
        best_algorithm = max(results.items(), key=lambda x: x[1].sharpe_ratio)
        print(f"\nBest ML Algorithm: {best_algorithm[0]} (Sharpe: {best_algorithm[1].sharpe_ratio:.2f})")
    
    return results 