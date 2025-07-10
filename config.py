"""
Configuration file for Quant Trader Pro
Centralized settings for the application
"""

import os
from typing import Dict, Any

class Config:
    """Application configuration class."""
    
    # Application settings
    APP_NAME = "Quant Trader Pro"
    APP_VERSION = "1.0.0"
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    
    # Streamlit settings
    STREAMLIT_PORT = int(os.getenv('STREAMLIT_PORT', 8501))
    STREAMLIT_HOST = os.getenv('STREAMLIT_HOST', 'localhost')
    
    # Alpaca API settings
    ALPACA_API_KEY = os.getenv('APCA_API_KEY_ID')
    ALPACA_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
    ALPACA_BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
    
    # Data settings
    DEFAULT_PERIOD = '1y'
    DEFAULT_TICKER = 'AAPL'
    DATA_CACHE_DURATION = 300  # 5 minutes
    
    # Technical indicators settings
    INDICATORS = {
        'SMA': {'window': 20},
        'EMA': {'window': 20},
        'RSI': {'window': 14},
        'MACD': {'fast': 12, 'slow': 26, 'signal': 9},
        'Bollinger_Bands': {'window': 20, 'std': 2},
        'Stochastic': {'window': 14, 'smooth': 3},
        'CCI': {'window': 20},
        'ADX': {'window': 14},
        'Williams_R': {'window': 14}
    }
    
    # Trading parameters
    DEFAULT_INITIAL_CAPITAL = 10000
    DEFAULT_POSITION_SIZE = 100  # percentage
    DEFAULT_STOP_LOSS = 5.0  # percentage
    DEFAULT_TAKE_PROFIT = 10.0  # percentage
    
    # Risk management
    MAX_POSITION_SIZE = 20  # percentage of portfolio
    MAX_DAILY_TRADES = 10
    ENABLE_STOP_LOSS = True
    ENABLE_TAKE_PROFIT = True
    
    # Backtesting settings
    BACKTEST_MIN_DATA_POINTS = 20
    BACKTEST_COMMISSION = 0.001  # 0.1% commission
    
    # Chart settings
    CHART_HEIGHT = 800
    CHART_THEME = 'plotly_white'
    
    # UI settings
    SIDEBAR_STATE = 'expanded'
    LAYOUT = 'wide'
    
    @classmethod
    def get_alpaca_config(cls) -> Dict[str, Any]:
        """Get Alpaca configuration dictionary."""
        return {
            'key_id': cls.ALPACA_API_KEY,
            'secret_key': cls.ALPACA_SECRET_KEY,
            'base_url': cls.ALPACA_BASE_URL
        }
    
    @classmethod
    def is_alpaca_configured(cls) -> bool:
        """Check if Alpaca API is properly configured."""
        return bool(cls.ALPACA_API_KEY and cls.ALPACA_SECRET_KEY)
    
    @classmethod
    def is_paper_trading(cls) -> bool:
        """Check if using paper trading."""
        return 'paper-api.alpaca.markets' in cls.ALPACA_BASE_URL
    
    @classmethod
    def get_trading_params(cls) -> Dict[str, Any]:
        """Get trading parameters dictionary."""
        return {
            'initial_capital': cls.DEFAULT_INITIAL_CAPITAL,
            'position_size': cls.DEFAULT_POSITION_SIZE,
            'stop_loss': cls.DEFAULT_STOP_LOSS,
            'take_profit': cls.DEFAULT_TAKE_PROFIT,
            'max_position_size': cls.MAX_POSITION_SIZE,
            'max_daily_trades': cls.MAX_DAILY_TRADES,
            'enable_stop_loss': cls.ENABLE_STOP_LOSS,
            'enable_take_profit': cls.ENABLE_TAKE_PROFIT
        }

# Color scheme for the application
COLORS = {
    'primary': '#1f77b4',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'info': '#17a2b8',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'white': '#ffffff',
    'black': '#000000'
}

# Chart colors
CHART_COLORS = {
    'candlestick_up': '#26a69a',
    'candlestick_down': '#ef5350',
    'sma': '#ff9800',
    'ema': '#9c27b0',
    'rsi': '#2196f3',
    'macd': '#4caf50',
    'bollinger_upper': '#757575',
    'bollinger_lower': '#757575',
    'volume_up': '#26a69a',
    'volume_down': '#ef5350'
}

# Time periods for data fetching
TIME_PERIODS = [
    ('1 Day', '1d'),
    ('5 Days', '5d'),
    ('1 Month', '1mo'),
    ('3 Months', '3mo'),
    ('6 Months', '6mo'),
    ('1 Year', '1y'),
    ('2 Years', '2y'),
    ('5 Years', '5y'),
    ('10 Years', '10y'),
    ('Year to Date', 'ytd'),
    ('Max', 'max')
]

# Popular stock tickers for quick access
POPULAR_TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    'META', 'NVDA', 'NFLX', 'AMD', 'INTC',
    'SPY', 'QQQ', 'IWM', 'VTI', 'VOO'
]

# Technical indicators for selection
TECHNICAL_INDICATORS = [
    ('RSI', 'RSI'),
    ('MACD', 'MACD'),
    ('Bollinger Bands', 'Bollinger_Bands'),
    ('Stochastic', 'Stochastic'),
    ('CCI', 'CCI'),
    ('ADX', 'ADX'),
    ('Williams %R', 'Williams_R')
] 