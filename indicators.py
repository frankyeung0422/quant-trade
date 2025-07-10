import pandas as pd
import ta

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add common technical indicators to the DataFrame.
    """
    df = df.copy()
    
    # Ensure we have the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # SMA & EMA
    df['SMA_20'] = ta.trend.sma_indicator(close=df['Close'], window=20)
    df['EMA_20'] = ta.trend.ema_indicator(close=df['Close'], window=20)
    
    # RSI
    df['RSI_14'] = ta.momentum.rsi(close=df['Close'], window=14)
    
    # MACD - fix the calculation
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Middle'] = bb.bollinger_mavg()
    
    # Stochastic Oscillator
    df['STOCH_K'] = ta.momentum.stoch(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    df['STOCH_D'] = ta.momentum.stoch_signal(high=df['High'], low=df['Low'], close=df['Close'], window=14, smooth_window=3)
    
    # CCI
    df['CCI'] = ta.trend.cci(high=df['High'], low=df['Low'], close=df['Close'], window=20)
    
    # ADX
    df['ADX'] = ta.trend.adx(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    
    # Williams %R
    df['WILLR'] = ta.momentum.williams_r(high=df['High'], low=df['Low'], close=df['Close'], lbp=14)
    
    # OBV
    df['OBV'] = ta.volume.on_balance_volume(close=df['Close'], volume=df['Volume'])
    
    return df 