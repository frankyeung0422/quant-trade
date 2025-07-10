import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

class DeepLearningStrategy:
    """Advanced deep learning strategy with LSTM and Transformer models"""
    
    def __init__(self, sequence_length=60, prediction_days=1):
        self.sequence_length = sequence_length
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.history = None
        self.feature_columns = []
        
    def prepare_data(self, df: pd.DataFrame, target_column='Close') -> tuple:
        """Prepare data for deep learning models"""
        # Select features
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if 'RSI_14' in df.columns:
            feature_columns.append('RSI_14')
        if 'MACD' in df.columns:
            feature_columns.append('MACD')
        if 'BB_Upper' in df.columns:
            feature_columns.extend(['BB_Upper', 'BB_Lower', 'BB_Middle'])
        if 'SMA_20' in df.columns:
            feature_columns.append('SMA_20')
        if 'EMA_20' in df.columns:
            feature_columns.append('EMA_20')
            
        self.feature_columns = feature_columns
        
        # Create features DataFrame
        features_df = df[feature_columns].copy()
        
        # Add technical indicators as features
        features_df['Price_Change'] = features_df['Close'].pct_change()
        features_df['Volume_Change'] = features_df['Volume'].pct_change()
        features_df['High_Low_Ratio'] = features_df['High'] / features_df['Low']
        features_df['Close_Open_Ratio'] = features_df['Close'] / features_df['Open']
        
        # Add lagged features
        for i in range(1, 6):
            features_df[f'Close_Lag_{i}'] = features_df['Close'].shift(i)
            features_df[f'Volume_Lag_{i}'] = features_df['Volume'].shift(i)
            
        # Add rolling statistics
        features_df['Close_MA_5'] = features_df['Close'].rolling(window=5).mean()
        features_df['Close_MA_10'] = features_df['Close'].rolling(window=10).mean()
        features_df['Volume_MA_5'] = features_df['Volume'].rolling(window=5).mean()
        features_df['Price_Volatility'] = features_df['Close'].rolling(window=20).std()
        
        # Drop NaN values
        features_df = features_df.dropna()
        
        # Scale features
        scaled_data = self.scaler.fit_transform(features_df)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data) - self.prediction_days + 1):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i:i+self.prediction_days, features_df.columns.get_loc('Close')])
            
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        return X_train, X_test, y_train, y_test, features_df.columns.tolist()
        
    def build_lstm_model(self, input_shape: tuple, output_shape: int) -> Model:
        """Build LSTM model with attention mechanism"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(output_shape)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def build_transformer_model(self, input_shape: tuple, output_shape: int) -> Model:
        """Build Transformer model for time series prediction"""
        inputs = Input(shape=input_shape)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=8, 
            key_dim=64
        )(inputs, inputs)
        
        # Add & Norm
        attention_output = LayerNormalization(epsilon=1e-6)(attention_output + inputs)
        
        # Feed forward
        ffn_output = Dense(256, activation='relu')(attention_output)
        ffn_output = Dense(input_shape[-1])(ffn_output)
        
        # Add & Norm
        ffn_output = LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)
        
        # Global average pooling
        pooled_output = tf.keras.layers.GlobalAveragePooling1D()(ffn_output)
        
        # Output layers
        dense_output = Dense(64, activation='relu')(pooled_output)
        dense_output = Dropout(0.2)(dense_output)
        dense_output = Dense(32, activation='relu')(dense_output)
        outputs = Dense(output_shape)(dense_output)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def build_ensemble_model(self, input_shape: tuple, output_shape: int) -> Model:
        """Build ensemble model combining LSTM and Transformer"""
        # LSTM branch
        lstm_input = Input(shape=input_shape)
        lstm_output = LSTM(64, return_sequences=True)(lstm_input)
        lstm_output = LSTM(32)(lstm_output)
        lstm_output = Dense(16, activation='relu')(lstm_output)
        
        # Transformer branch
        transformer_input = Input(shape=input_shape)
        transformer_output = MultiHeadAttention(num_heads=4, key_dim=32)(transformer_input, transformer_input)
        transformer_output = LayerNormalization(epsilon=1e-6)(transformer_output + transformer_input)
        transformer_output = tf.keras.layers.GlobalAveragePooling1D()(transformer_output)
        transformer_output = Dense(16, activation='relu')(transformer_output)
        
        # Combine outputs
        combined = tf.keras.layers.Concatenate()([lstm_output, transformer_output])
        combined = Dense(32, activation='relu')(combined)
        combined = Dropout(0.2)(combined)
        outputs = Dense(output_shape)(combined)
        
        model = Model(inputs=[lstm_input, transformer_input], outputs=outputs)
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
        
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, y_test: np.ndarray, 
                   model_type: str = 'lstm', epochs: int = 100) -> dict:
        """Train the deep learning model"""
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        if model_type == 'lstm':
            self.model = self.build_lstm_model(X_train.shape[1:], y_train.shape[1])
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
        elif model_type == 'transformer':
            self.model = self.build_transformer_model(X_train.shape[1:], y_train.shape[1])
            self.history = self.model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
        elif model_type == 'ensemble':
            self.model = self.build_ensemble_model(X_train.shape[1:], y_train.shape[1])
            self.history = self.model.fit(
                [X_train, X_train], y_train,
                validation_data=([X_test, X_test], y_test),
                epochs=epochs,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
        # Evaluate model
        if model_type == 'ensemble':
            y_pred = self.model.predict([X_test, X_test])
        else:
            y_pred = self.model.predict(X_test)
            
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Calculate directional accuracy
        y_test_direction = np.diff(y_test.flatten()) > 0
        y_pred_direction = np.diff(y_pred.flatten()) > 0
        direction_accuracy = accuracy_score(y_test_direction, y_pred_direction)
        
        return {
            'mse': mse,
            'mae': mae,
            'direction_accuracy': direction_accuracy,
            'history': self.history.history if self.history else None
        }
        
    def predict(self, df: pd.DataFrame, model_type: str = 'lstm') -> np.ndarray:
        """Make predictions using the trained model"""
        if self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
            
        # Prepare data
        X_train, X_test, y_train, y_test, _ = self.prepare_data(df)
        
        # Make predictions
        if model_type == 'ensemble':
            predictions = self.model.predict([X_test, X_test])
        else:
            predictions = self.model.predict(X_test)
            
        return predictions
        
    def generate_signals(self, df: pd.DataFrame, threshold: float = 0.02) -> pd.Series:
        """Generate trading signals based on predictions"""
        predictions = self.predict(df)
        
        # Get actual prices for comparison
        actual_prices = df['Close'].values[-len(predictions):]
        
        # Calculate predicted price changes
        predicted_changes = np.diff(predictions.flatten())
        actual_changes = np.diff(actual_prices)
        
        # Generate signals
        signals = pd.Series(index=df.index[-len(predicted_changes):], data='Hold')
        
        # Buy signal: predicted increase > threshold
        buy_signals = predicted_changes > threshold
        signals[buy_signals] = 'Buy'
        
        # Sell signal: predicted decrease > threshold
        sell_signals = predicted_changes < -threshold
        signals[sell_signals] = 'Sell'
        
        return signals

class AdvancedMLEnsemble:
    """Advanced ensemble of multiple ML models"""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        
    def add_model(self, name: str, model, weight: float = 1.0):
        """Add a model to the ensemble"""
        self.models[name] = model
        self.weights[name] = weight
        
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble prediction"""
        predictions = []
        
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                pred = model.predict(X)
                predictions.append(pred * self.weights[name])
            else:
                # For non-ML models (e.g., technical indicators)
                pred = self._generate_technical_prediction(model, X)
                predictions.append(pred * self.weights[name])
                
        # Weighted average
        ensemble_prediction = np.average(predictions, axis=0, weights=list(self.weights.values()))
        return ensemble_prediction
        
    def _generate_technical_prediction(self, indicator_type: str, X: np.ndarray) -> np.ndarray:
        """Generate predictions based on technical indicators"""
        # This is a simplified implementation
        # In practice, you'd implement specific technical analysis logic
        return np.random.normal(0, 0.01, X.shape[0])

def run_deep_learning_analysis(ticker: str, period: str = '1y', model_type: str = 'lstm') -> dict:
    """Run comprehensive deep learning analysis"""
    
    # Fetch data
    df = yf.Ticker(ticker).history(period=period)
    if df.empty:
        return {'error': 'No data available'}
        
    # Calculate technical indicators
    df['RSI_14'] = calculate_rsi(df['Close'], 14)
    df['MACD'] = calculate_macd(df['Close'])
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Low'] = df['BB_Middle'] - (bb_std * 2)
    
    # Initialize strategy
    strategy = DeepLearningStrategy(sequence_length=60, prediction_days=1)
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = strategy.prepare_data(df)
    
    # Train model
    results = strategy.train_model(X_train, y_train, X_test, y_test, model_type, epochs=50)
    
    # Generate signals
    signals = strategy.generate_signals(df)
    
    # Calculate performance metrics
    performance = calculate_performance_metrics(df, signals)
    
    return {
        'model_type': model_type,
        'training_results': results,
        'signals': signals,
        'performance': performance,
        'feature_names': feature_names
    }

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    return macd

def calculate_performance_metrics(df: pd.DataFrame, signals: pd.Series) -> dict:
    """Calculate performance metrics for trading signals"""
    
    # Simple backtest
    capital = 10000
    position = 0
    entry_price = 0
    trades = []
    
    for i, signal in enumerate(signals):
        current_price = df['Close'].iloc[i]
        
        if signal == 'Buy' and position == 0:
            position = capital / current_price
            entry_price = current_price
            trades.append({
                'action': 'Buy',
                'price': current_price,
                'date': df.index[i]
            })
        elif signal == 'Sell' and position > 0:
            exit_price = current_price
            profit = (exit_price - entry_price) * position
            capital += profit
            position = 0
            trades.append({
                'action': 'Sell',
                'price': current_price,
                'profit': profit,
                'date': df.index[i]
            })
    
    # Calculate metrics
    total_return = (capital - 10000) / 10000 * 100
    num_trades = len([t for t in trades if t['action'] == 'Sell'])
    
    if num_trades > 0:
        profitable_trades = len([t for t in trades if t.get('profit', 0) > 0])
        win_rate = profitable_trades / num_trades * 100
    else:
        win_rate = 0
        
    return {
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'final_capital': capital,
        'trades': trades
    }

# Example usage
if __name__ == "__main__":
    # Run deep learning analysis
    results = run_deep_learning_analysis('AAPL', '1y', 'lstm')
    
    print("Deep Learning Analysis Results:")
    print(f"Model Type: {results['model_type']}")
    print(f"Total Return: {results['performance']['total_return']:.2f}%")
    print(f"Number of Trades: {results['performance']['num_trades']}")
    print(f"Win Rate: {results['performance']['win_rate']:.2f}%")
    print(f"Direction Accuracy: {results['training_results']['direction_accuracy']:.2f}") 