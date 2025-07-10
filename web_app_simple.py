import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import ta
import os
import time

# Try to import optional modules, but don't fail if they're missing
try:
    from alpaca_trade_api.rest import REST, TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    st.warning("⚠️ Alpaca API not available. Trading features will be disabled.")

# Try to import local modules, but provide fallbacks
try:
    from data_fetcher import fetch_data
    DATA_FETCHER_AVAILABLE = True
except ImportError:
    DATA_FETCHER_AVAILABLE = False

try:
    from indicators import calculate_indicators
    INDICATORS_AVAILABLE = True
except ImportError:
    INDICATORS_AVAILABLE = False

try:
    from analysis import generate_analysis
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False

try:
    from backtest import run_backtest
    BACKTEST_AVAILABLE = True
except ImportError:
    BACKTEST_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Quant Trader Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = None
if 'current_period' not in st.session_state:
    st.session_state.current_period = '1y'

def fetch_stock_data(ticker, period='1y'):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            st.error(f"No data found for {ticker}")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None

def calculate_basic_indicators(df):
    """Calculate basic technical indicators"""
    try:
        # Simple Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        
        # RSI
        df['RSI_14'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['Close'])
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
        df['BB_Mid'] = bb.bollinger_mavg()
        
        return df
    except Exception as e:
        st.error(f"Error calculating indicators: {str(e)}")
        return df

def create_candlestick_chart(df, ticker):
    """Create an interactive candlestick chart with indicators"""
    try:
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{ticker} Price & Indicators', 'RSI', 'MACD'),
            row_heights=[0.6, 0.2, 0.2]
        )

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        ), row=1, col=1)

        # Add SMA and EMA
        if 'SMA_20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['SMA_20'],
                mode='lines', name='SMA 20',
                line=dict(color='orange', width=2)
            ), row=1, col=1)

        if 'EMA_20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['EMA_20'],
                mode='lines', name='EMA 20',
                line=dict(color='purple', width=2)
            ), row=1, col=1)

        # Add Bollinger Bands
        if 'BB_High' in df.columns and 'BB_Low' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_High'],
                mode='lines', name='BB Upper',
                line=dict(color='gray', width=1, dash='dash')
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=df.index, y=df['BB_Low'],
                mode='lines', name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty'
            ), row=1, col=1)

        # RSI
        if 'RSI_14' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['RSI_14'],
                mode='lines', name='RSI',
                line=dict(color='blue', width=2)
            ), row=2, col=1)

            # Add RSI overbought/oversold lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # MACD
        if 'MACD' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['MACD'],
                mode='lines', name='MACD',
                line=dict(color='blue', width=2)
            ), row=3, col=1)

            if 'MACD_Signal' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df['MACD_Signal'],
                    mode='lines', name='MACD Signal',
                    line=dict(color='red', width=2)
                ), row=3, col=1)

        fig.update_layout(
            title=f'{ticker} Technical Analysis',
            xaxis_rangeslider_visible=False,
            height=700,
            showlegend=True
        )

        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def create_analysis_dashboard(df, ticker):
    """Create analysis dashboard with key metrics"""
    try:
        latest = df.iloc[-1]
        
        # Calculate key metrics
        current_price = latest['Close']
        price_change = current_price - df.iloc[-2]['Close']
        price_change_pct = (price_change / df.iloc[-2]['Close']) * 100
        
        # Create metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Current Price",
                value=f"${current_price:.2f}",
                delta=f"{price_change:.2f} ({price_change_pct:.2f}%)"
            )
        
        with col2:
            if 'RSI_14' in df.columns and not pd.isna(latest['RSI_14']):
                rsi_status = "Overbought" if latest['RSI_14'] > 70 else "Oversold" if latest['RSI_14'] < 30 else "Neutral"
                st.metric(
                    label="RSI (14)",
                    value=f"{latest['RSI_14']:.1f}",
                    delta=rsi_status
                )
            else:
                st.metric(label="RSI (14)", value="N/A")
        
        with col3:
            if 'MACD' in df.columns and not pd.isna(latest['MACD']):
                macd_status = "Bullish" if latest['MACD'] > 0 else "Bearish"
                st.metric(
                    label="MACD",
                    value=f"{latest['MACD']:.3f}",
                    delta=macd_status
                )
            else:
                st.metric(label="MACD", value="N/A")
        
        with col4:
            if 'BB_High' in df.columns and 'BB_Low' in df.columns:
                bb_position = (current_price - latest['BB_Low']) / (latest['BB_High'] - latest['BB_Low'])
                bb_status = "Upper Band" if bb_position > 0.8 else "Lower Band" if bb_position < 0.2 else "Middle"
                st.metric(
                    label="BB Position",
                    value=f"{bb_position:.1%}",
                    delta=bb_status
                )
            else:
                st.metric(label="BB Position", value="N/A")
    except Exception as e:
        st.error(f"Error creating analysis dashboard: {str(e)}")

def main():
    # Main header
    st.markdown('<h1 class="main-header">📈 Quant Trader Pro</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Stock selection
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL").upper()
    
    # Time period selection
    period_options = {
        "1 Day": "1d",
        "5 Days": "5d", 
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y",
        "2 Years": "2y",
        "5 Years": "5y",
        "Max": "max"
    }
    period = st.sidebar.selectbox("Time Period", list(period_options.keys()), index=5)
    period_value = period_options[period]
    
    # Fetch data button
    if st.sidebar.button("📊 Fetch Data", type="primary"):
        with st.spinner("Fetching data..."):
            data = fetch_stock_data(ticker, period_value)
            if data is not None:
                # Calculate indicators
                data = calculate_basic_indicators(data)
                st.session_state.current_data = data
                st.session_state.current_ticker = ticker
                st.session_state.current_period = period_value
                st.success(f"✅ Data loaded for {ticker}")
    
    # Main content
    if st.session_state.current_data is not None:
        df = st.session_state.current_data
        ticker = st.session_state.current_ticker
        
        # Analysis dashboard
        create_analysis_dashboard(df, ticker)
        
        # Chart
        st.subheader("📈 Technical Analysis Chart")
        chart = create_candlestick_chart(df, ticker)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        
        # Data table
        st.subheader("📋 Data Table")
        st.dataframe(df.tail(20), use_container_width=True)
        
        # Statistics
        st.subheader("📊 Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Return", f"{((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%")
        
        with col2:
            st.metric("Volatility", f"{df['Close'].pct_change().std() * 100:.2f}%")
        
        with col3:
            st.metric("Max Price", f"${df['High'].max():.2f}")
    
    else:
        st.info("👆 Please enter a stock ticker and click 'Fetch Data' to get started!")
        
        # Show some popular stocks
        st.subheader("💡 Popular Stocks")
        popular_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
        cols = st.columns(4)
        for i, stock in enumerate(popular_stocks):
            with cols[i % 4]:
                if st.button(stock):
                    st.session_state.current_ticker = stock
                    st.rerun()

if __name__ == "__main__":
    main() 