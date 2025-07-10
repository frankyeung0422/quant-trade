import yfinance as yf
import pandas as pd
from typing import Optional

def fetch_data(ticker: str, period: str = '1y') -> pd.DataFrame:
    """
    Fetch historical stock data for the given ticker and period.
    Ensures columns are flat and compatible with TA-Lib/ta.
    Supports US, Hong Kong, and other international stocks.
    """
    try:
        # Clean ticker symbol
        ticker = ticker.strip().upper()
        
        # Validate ticker format
        if not ticker:
            raise ValueError("Ticker symbol cannot be empty")
        
        # Download data
        data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        
        # Check if data is empty
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        # If columns are multi-index (from yfinance), flatten them
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns.values]
            # Try to extract the correct columns for a single ticker
            if f'Close_{ticker}' in data.columns:
                data = data.rename(columns={
                    f'Open_{ticker}': 'Open',
                    f'High_{ticker}': 'High',
                    f'Low_{ticker}': 'Low',
                    f'Close_{ticker}': 'Close',
                    f'Volume_{ticker}': 'Volume',
                })
        
        # Ensure required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove any rows with NaN values in critical columns
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Ensure we have enough data
        if len(data) < 20:
            raise ValueError(f"Insufficient data for {ticker}. Need at least 20 data points, got {len(data)}")
        
        return data
        
    except Exception as e:
        # Re-raise with more context
        raise Exception(f"Error fetching data for {ticker}: {str(e)}")

def get_stock_info(ticker: str) -> Optional[dict]:
    """
    Get basic stock information including company name, sector, etc.
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
        return {
            'name': info.get('longName', info.get('shortName', ticker)),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'market_cap': info.get('marketCap'),
            'currency': info.get('currency', 'Unknown'),
            'exchange': info.get('exchange', 'Unknown')
        }
    except:
        return None

def validate_ticker(ticker: str) -> bool:
    """
    Validate if a ticker symbol is properly formatted.
    """
    if not ticker or not isinstance(ticker, str):
        return False
    
    ticker = ticker.strip().upper()
    
    # Basic validation rules
    if len(ticker) < 1 or len(ticker) > 10:
        return False
    
    # Check for common patterns
    if ticker.endswith('.HK'):
        # Hong Kong stocks: 4 digits + .HK
        base = ticker[:-3]
        if not base.isdigit() or len(base) != 4:
            return False
    elif '.' in ticker:
        # Other international stocks
        parts = ticker.split('.')
        if len(parts) != 2:
            return False
    
    return True 