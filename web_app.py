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
from alpaca_trade_api.rest import REST, TimeFrame
import time

# Import our existing modules
from data_fetcher import fetch_data
from indicators import calculate_indicators
from analysis import generate_analysis
from backtest import run_backtest
from quant_trader import execute_trades
from real_time_trader import RealTimeTrader
from portfolio_manager import PortfolioManager

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
if 'real_time_trader' not in st.session_state:
    st.session_state.real_time_trader = None
if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = None
if 'is_streaming' not in st.session_state:
    st.session_state.is_streaming = False

def create_candlestick_chart(df, ticker):
    """Create an interactive candlestick chart with indicators"""
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(f'{ticker} Price & Indicators', 'RSI', 'MACD', 'Volume'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
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
    fig.add_trace(go.Scatter(
        x=df.index, y=df['SMA_20'],
        mode='lines', name='SMA 20',
        line=dict(color='orange', width=2)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df.index, y=df['EMA_20'],
        mode='lines', name='EMA 20',
        line=dict(color='purple', width=2)
    ), row=1, col=1)

    # Add Bollinger Bands
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
    fig.add_trace(go.Scatter(
        x=df.index, y=df['RSI_14'],
        mode='lines', name='RSI',
        line=dict(color='blue', width=2)
    ), row=2, col=1)

    # Add RSI overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # MACD
    fig.add_trace(go.Scatter(
        x=df.index, y=df['MACD'],
        mode='lines', name='MACD',
        line=dict(color='blue', width=2)
    ), row=3, col=1)

    # Volume
    colors = ['red' if close < open else 'green' for close, open in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(
        x=df.index, y=df['Volume'],
        name='Volume',
        marker_color=colors
    ), row=4, col=1)

    fig.update_layout(
        title=f'{ticker} Technical Analysis',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True
    )

    return fig

def create_analysis_dashboard(df, ticker):
    """Create analysis dashboard with key metrics"""
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
        rsi_status = "Overbought" if latest['RSI_14'] > 70 else "Oversold" if latest['RSI_14'] < 30 else "Neutral"
        st.metric(
            label="RSI (14)",
            value=f"{latest['RSI_14']:.1f}",
            delta=rsi_status
        )
    
    with col3:
        macd_status = "Bullish" if latest['MACD'] > 0 else "Bearish"
        st.metric(
            label="MACD",
            value=f"{latest['MACD']:.3f}",
            delta=macd_status
        )
    
    with col4:
        bb_position = (current_price - latest['BB_Low']) / (latest['BB_High'] - latest['BB_Low'])
        bb_status = "Upper Band" if bb_position > 0.8 else "Lower Band" if bb_position < 0.2 else "Middle"
        st.metric(
            label="BB Position",
            value=f"{bb_position:.1%}",
            delta=bb_status
        )

def create_real_time_dashboard():
    """Create real-time trading dashboard"""
    st.subheader("🚀 Real-Time Trading Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        ticker = st.text_input("Ticker Symbol", value="AAPL", key="rt_ticker")
        initial_capital = st.number_input("Initial Capital ($)", value=10000, min_value=1000, key="rt_capital")
        
    with col2:
        if st.button("Start Real-Time Trading", key="start_rt"):
            if not st.session_state.is_streaming:
                st.session_state.real_time_trader = RealTimeTrader(ticker, initial_capital)
                st.session_state.real_time_trader.start_streaming()
                st.session_state.is_streaming = True
                st.success(f"Started real-time trading for {ticker}")
                
        if st.button("Stop Real-Time Trading", key="stop_rt"):
            if st.session_state.is_streaming and st.session_state.real_time_trader:
                st.session_state.real_time_trader.stop_streaming()
                st.session_state.is_streaming = False
                st.warning("Stopped real-time trading")
                
    with col3:
        if st.session_state.is_streaming:
            st.success("🟢 Live Trading Active")
        else:
            st.error("🔴 Trading Inactive")
    
    # Display real-time status
    if st.session_state.real_time_trader:
        status = st.session_state.real_time_trader.get_portfolio_status()
        
        st.subheader("📊 Real-Time Portfolio Status")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Value", f"${status['current_value']:,.2f}")
        with col2:
            st.metric("Total Return", f"{status['total_return']:.2f}%")
        with col3:
            st.metric("Cash", f"${status['capital']:,.2f}")
        with col4:
            st.metric("Trades", status['num_trades'])
            
        # Display recent trades
        if st.session_state.real_time_trader.trades:
            st.subheader("📈 Recent Trades")
            trades_df = pd.DataFrame(st.session_state.real_time_trader.trades[-10:])
            st.dataframe(trades_df)
            
        # Display last signal
        if st.session_state.real_time_trader.signals:
            last_signal = st.session_state.real_time_trader.signals[-1]
            st.subheader("🎯 Last Signal")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Signal", last_signal.signal_type.value)
            with col2:
                st.metric("Price", f"${last_signal.price:.2f}")
            with col3:
                st.metric("Confidence", f"{last_signal.confidence:.1%}")

def create_portfolio_dashboard():
    """Create portfolio management dashboard"""
    st.subheader("💼 Portfolio Management")
    
    # Initialize portfolio manager
    if st.session_state.portfolio_manager is None:
        st.session_state.portfolio_manager = PortfolioManager(initial_capital=100000)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Add Position")
        ticker = st.text_input("Ticker", key="add_ticker")
        shares = st.number_input("Shares", min_value=1, key="add_shares")
        price = st.number_input("Price", min_value=0.01, key="add_price")
        
        if st.button("Add Position"):
            try:
                st.session_state.portfolio_manager.add_position(ticker, shares, price)
                st.success(f"Added {shares} shares of {ticker} at ${price:.2f}")
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.subheader("Remove Position")
        if st.session_state.portfolio_manager.positions:
            ticker_to_remove = st.selectbox("Select Ticker", list(st.session_state.portfolio_manager.positions.keys()), key="remove_ticker")
            remove_price = st.number_input("Current Price", min_value=0.01, key="remove_price")
            
            if st.button("Remove Position"):
                try:
                    st.session_state.portfolio_manager.remove_position(ticker_to_remove, remove_price)
                    st.success(f"Removed position in {ticker_to_remove}")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Display portfolio summary
    if st.session_state.portfolio_manager.positions:
        st.subheader("📊 Portfolio Summary")
        summary = st.session_state.portfolio_manager.get_portfolio_summary()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Value", f"${summary['total_value']:,.2f}")
        with col2:
            st.metric("Total Return", f"{summary['total_return']:.2%}")
        with col3:
            st.metric("Cash", f"${summary['cash']:,.2f}")
        with col4:
            st.metric("Positions", summary['num_positions'])
        
        # Display positions
        st.subheader("📈 Current Positions")
        positions_data = []
        for ticker, pos_data in summary['positions'].items():
            positions_data.append({
                'Ticker': ticker,
                'Shares': pos_data['shares'],
                'Entry Price': f"${pos_data['entry_price']:.2f}",
                'Current Price': f"${pos_data['current_price']:.2f}",
                'Current Value': f"${pos_data['current_value']:.2f}",
                'Return': f"{pos_data['return']:.2%}",
                'Weight': f"{pos_data['weight']:.1%}"
            })
        
        positions_df = pd.DataFrame(positions_data)
        st.dataframe(positions_df)
        
        # Portfolio optimization
        st.subheader("🎯 Portfolio Optimization")
        if st.button("Optimize Portfolio"):
            optimization = st.session_state.portfolio_manager.optimize_portfolio(target_volatility=0.15)
            if optimization:
                st.success("Portfolio optimized!")
                st.write("Optimal Weights:")
                for ticker, weight in optimization['weights'].items():
                    st.write(f"{ticker}: {weight:.1%}")
                st.write(f"Expected Return: {optimization['expected_return']:.2%}")
                st.write(f"Expected Volatility: {optimization['expected_volatility']:.2%}")

def run_enhanced_backtest(df):
    """Enhanced backtest with detailed results"""
    capital = 10000
    position = 0
    entry_price = 0
    trades = []
    portfolio_values = []
    
    for i in range(20, len(df)):
        sub_df = df.iloc[:i+1]
        _, rec = generate_analysis(sub_df)
        price = df.iloc[i]['Close']
        date = df.index[i]
        
        if rec == 'Buy' and position == 0:
            position = capital / price
            entry_price = price
            capital = 0
            trades.append({
                'date': date,
                'action': 'BUY',
                'price': price,
                'shares': position
            })
        elif rec == 'Sell' and position > 0:
            capital = position * price
            profit = price - entry_price
            profit_pct = (profit / entry_price) * 100
            trades.append({
                'date': date,
                'action': 'SELL',
                'price': price,
                'shares': position,
                'profit': profit,
                'profit_pct': profit_pct
            })
            position = 0
        
        # Calculate current portfolio value
        current_value = capital + (position * price)
        portfolio_values.append({
            'date': date,
            'value': current_value
        })
    
    # Final liquidation
    if position > 0:
        final_price = df.iloc[-1]['Close']
        capital = position * final_price
        profit = final_price - entry_price
        profit_pct = (profit / entry_price) * 100
        trades.append({
            'date': df.index[-1],
            'action': 'SELL',
            'price': final_price,
            'shares': position,
            'profit': profit,
            'profit_pct': profit_pct
        })
    
    return capital, trades, portfolio_values

def main():
    # Sidebar
    st.sidebar.title("📈 Quant Trader Pro")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Navigation",
        ["Dashboard", "Technical Analysis", "Backtesting", "Live Trading", "Portfolio", "Real-Time Trading", "Portfolio Management", "Settings"]
    )
    
    # Sidebar inputs
    st.sidebar.header("Configuration")
    
    # Market selection
    market = st.sidebar.selectbox(
        "Market",
        ["US Stocks", "Hong Kong Stocks", "Other"],
        help="Select the market for your stock"
    )
    
    # Popular stocks by market with descriptions
    popular_stocks = {
        "US Stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX"],
        "Hong Kong Stocks": ["0700.HK", "0941.HK", "9988.HK", "3690.HK", "9618.HK", "1810.HK", "2269.HK", "2020.HK"],
        "Other": ["TSMC", "ASML", "NFLX", "AMD", "INTC"]
    }
    
    # Stock descriptions for tooltips
    stock_descriptions = {
        "0700.HK": "Tencent Holdings (Technology)",
        "0941.HK": "China Mobile (Telecommunications)",
        "9988.HK": "Alibaba Group (E-commerce)",
        "3690.HK": "Meituan (Food Delivery)",
        "9618.HK": "JD.com (E-commerce)",
        "1810.HK": "Xiaomi (Technology)",
        "2269.HK": "Li Ning (Sportswear)",
        "2020.HK": "Anta Sports (Sportswear)"
    }
    
    # Stock selection
    if market in popular_stocks:
        # Create options with descriptions for Hong Kong stocks
        if market == "Hong Kong Stocks":
            options = ["Custom"]
            for stock in popular_stocks[market]:
                desc = stock_descriptions.get(stock, "")
                options.append(f"{stock} - {desc}")
            
            selected_popular = st.sidebar.selectbox(
                "Popular Stocks",
                options,
                help="Select from popular Hong Kong stocks or enter custom ticker"
            )
        else:
            selected_popular = st.sidebar.selectbox(
                "Popular Stocks",
                ["Custom"] + popular_stocks[market],
                help="Select from popular stocks or enter custom ticker"
            )
        
        if selected_popular == "Custom":
            ticker = st.sidebar.text_input("Stock Ticker", value="").upper()
        else:
            # Extract ticker from selection (remove description for HK stocks)
            if market == "Hong Kong Stocks" and " - " in selected_popular:
                ticker = selected_popular.split(" - ")[0]
            else:
                ticker = selected_popular
    else:
        ticker = st.sidebar.text_input("Stock Ticker", value="").upper()
    
    # Add .HK suffix for Hong Kong stocks if not present
    if market == "Hong Kong Stocks" and ticker and not ticker.endswith('.HK'):
        ticker = f"{ticker}.HK"
    
    period = st.sidebar.selectbox(
        "Time Period",
        ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
        index=5
    )
    
    # Market info
    if market == "Hong Kong Stocks":
        st.sidebar.info("🇭🇰 Hong Kong stocks use .HK suffix (e.g., 0700.HK for Tencent)")
    elif market == "US Stocks":
        st.sidebar.info("🇺🇸 US stocks use standard tickers (e.g., AAPL, MSFT)")
    
    # Fetch data button
    if st.sidebar.button("📊 Fetch Data", type="primary"):
        if not ticker:
            st.error("Please enter a ticker symbol")
        else:
            with st.spinner(f"Fetching data for {ticker}..."):
                try:
                    data = fetch_data(ticker, period)
                    if not data.empty:
                        data = calculate_indicators(data)
                        st.session_state.current_data = data
                        st.session_state.current_ticker = ticker
                        st.session_state.current_period = period
                        st.success(f"✅ Data loaded for {ticker}")
                        
                        # Get and store stock info
                        from data_fetcher import get_stock_info
                        stock_info = get_stock_info(ticker)
                        if stock_info:
                            st.session_state.stock_info = stock_info
                    else:
                        st.error(f"❌ No data found for {ticker}. Please check the ticker symbol.")
                except Exception as e:
                    st.error(f"❌ Error fetching data for {ticker}: {str(e)}")
    
    # Quick stock info
    if ticker:
        st.sidebar.markdown("---")
        st.sidebar.subheader("📋 Stock Info")
        if market == "Hong Kong Stocks":
            st.sidebar.markdown(f"**Ticker:** {ticker}")
            st.sidebar.markdown("**Market:** Hong Kong Stock Exchange")
            st.sidebar.markdown("**Currency:** HKD")
        elif market == "US Stocks":
            st.sidebar.markdown(f"**Ticker:** {ticker}")
            st.sidebar.markdown("**Market:** US Stock Exchange")
            st.sidebar.markdown("**Currency:** USD")
        else:
            st.sidebar.markdown(f"**Ticker:** {ticker}")
            st.sidebar.markdown("**Market:** Other")
    
    # Main content
    if page == "Dashboard":
        st.markdown('<h1 class="main-header">📈 Quant Trader Pro Dashboard</h1>', unsafe_allow_html=True)
        
        if st.session_state.current_data is not None:
            df = st.session_state.current_data
            ticker = st.session_state.current_ticker
            
            # Display stock information if available
            if hasattr(st.session_state, 'stock_info') and st.session_state.stock_info:
                stock_info = st.session_state.stock_info
                st.subheader("📋 Company Information")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Company:** {stock_info['name']}")
                    st.markdown(f"**Sector:** {stock_info['sector']}")
                with col2:
                    st.markdown(f"**Industry:** {stock_info['industry']}")
                    st.markdown(f"**Currency:** {stock_info['currency']}")
                with col3:
                    if stock_info['market_cap']:
                        market_cap_billions = stock_info['market_cap'] / 1e9
                        st.markdown(f"**Market Cap:** ${market_cap_billions:.1f}B")
                    st.markdown(f"**Exchange:** {stock_info['exchange']}")
                st.markdown("---")
            
            # Analysis dashboard
            create_analysis_dashboard(df, ticker)
            
            # Generate analysis
            analysis, recommendation = generate_analysis(df)
            
            # Recommendation card
            st.subheader("🤖 AI Trading Recommendation")
            if recommendation == "Buy":
                st.success(f"🟢 {recommendation} - Strong bullish signals detected")
            elif recommendation == "Sell":
                st.error(f"🔴 {recommendation} - Bearish signals detected")
            else:
                st.warning(f"🟡 {recommendation} - Neutral signals, wait for better entry")
            
            # Analysis details
            with st.expander("📋 Detailed Analysis"):
                st.text(analysis)
            
            # Price chart
            st.subheader("📊 Price Chart")
            fig = create_candlestick_chart(df, ticker)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("👆 Please select a market, enter a ticker symbol, and click 'Fetch Data' to get started")
            
            # Show market examples
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("**🇺🇸 US Stocks**")
                st.markdown("- AAPL (Apple)")
                st.markdown("- MSFT (Microsoft)")
                st.markdown("- GOOGL (Google)")
                st.markdown("- TSLA (Tesla)")
            with col2:
                st.markdown("**🇭🇰 Hong Kong Stocks**")
                st.markdown("- 0700.HK (Tencent)")
                st.markdown("- 0941.HK (China Mobile)")
                st.markdown("- 9988.HK (Alibaba)")
                st.markdown("- 3690.HK (Meituan)")
            with col3:
                st.markdown("**🌍 Other Markets**")
                st.markdown("- TSMC (Taiwan)")
                st.markdown("- ASML (Netherlands)")
                st.markdown("- AMD (US)")
                st.markdown("- INTC (Intel)")
    
    elif page == "Technical Analysis":
        st.markdown('<h1 class="main-header">🔬 Technical Analysis</h1>', unsafe_allow_html=True)
        
        if st.session_state.current_data is not None:
            df = st.session_state.current_data
            ticker = st.session_state.current_ticker
            
            # Indicator selection
            indicators = ['RSI', 'MACD', 'Bollinger Bands', 'Stochastic', 'CCI', 'ADX', 'Williams %R']
            selected_indicators = st.multiselect("Select Indicators", indicators, default=['RSI', 'MACD'])
            
            # Create indicator charts
            if selected_indicators:
                num_indicators = len(selected_indicators)
                fig = make_subplots(
                    rows=num_indicators + 1, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=[f'{ticker} Price'] + selected_indicators,
                    row_heights=[0.4] + [0.6/num_indicators] * num_indicators
                )
                
                # Price chart
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='OHLC'
                ), row=1, col=1)
                
                # Add selected indicators
                for i, indicator in enumerate(selected_indicators, 2):
                    if indicator == 'RSI':
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df['RSI_14'],
                            mode='lines', name='RSI',
                            line=dict(color='blue', width=2)
                        ), row=i, col=1)
                        fig.add_hline(y=70, line_dash="dash", line_color="red", row=i, col=1)
                        fig.add_hline(y=30, line_dash="dash", line_color="green", row=i, col=1)
                    
                    elif indicator == 'MACD':
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df['MACD'],
                            mode='lines', name='MACD',
                            line=dict(color='blue', width=2)
                        ), row=i, col=1)
                    
                    elif indicator == 'Bollinger Bands':
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df['BB_High'],
                            mode='lines', name='BB Upper',
                            line=dict(color='gray', width=1, dash='dash')
                        ), row=i, col=1)
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df['BB_Low'],
                            mode='lines', name='BB Lower',
                            line=dict(color='gray', width=1, dash='dash')
                        ), row=i, col=1)
                        fig.add_trace(go.Scatter(
                            x=df.index, y=df['Close'],
                            mode='lines', name='Price',
                            line=dict(color='black', width=1)
                        ), row=i, col=1)
                
                fig.update_layout(height=300 * (num_indicators + 1), showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
            
            # Indicator values table
            st.subheader("📊 Current Indicator Values")
            latest = df.iloc[-1]
            indicator_data = {
                'Indicator': ['RSI (14)', 'MACD', 'CCI', 'ADX', 'Williams %R', 'Stochastic K', 'Stochastic D'],
                'Value': [
                    f"{latest['RSI_14']:.2f}",
                    f"{latest['MACD']:.3f}",
                    f"{latest['CCI']:.2f}",
                    f"{latest['ADX']:.2f}",
                    f"{latest['WILLR']:.2f}",
                    f"{latest['STOCH_K']:.2f}",
                    f"{latest['STOCH_D']:.2f}"
                ],
                'Status': [
                    "Overbought" if latest['RSI_14'] > 70 else "Oversold" if latest['RSI_14'] < 30 else "Neutral",
                    "Bullish" if latest['MACD'] > 0 else "Bearish",
                    "Overbought" if latest['CCI'] > 100 else "Oversold" if latest['CCI'] < -100 else "Neutral",
                    "Strong Trend" if latest['ADX'] > 25 else "Weak Trend",
                    "Overbought" if latest['WILLR'] > -20 else "Oversold" if latest['WILLR'] < -80 else "Neutral",
                    "Overbought" if latest['STOCH_K'] > 80 else "Oversold" if latest['STOCH_K'] < 20 else "Neutral",
                    "Overbought" if latest['STOCH_D'] > 80 else "Oversold" if latest['STOCH_D'] < 20 else "Neutral"
                ]
            }
            
            indicator_df = pd.DataFrame(indicator_data)
            st.dataframe(indicator_df, use_container_width=True)
            
        else:
            st.info("👆 Please fetch data first from the Dashboard")
    
    elif page == "Backtesting":
        st.markdown('<h1 class="main-header">🧪 Backtesting</h1>', unsafe_allow_html=True)
        
        if st.session_state.current_data is not None:
            df = st.session_state.current_data
            ticker = st.session_state.current_ticker
            
            # Backtest parameters
            col1, col2, col3 = st.columns(3)
            with col1:
                initial_capital = st.number_input("Initial Capital ($)", value=10000, min_value=1000, step=1000)
            with col2:
                position_size = st.slider("Position Size (%)", 10, 100, 100, 10)
            with col3:
                stop_loss = st.number_input("Stop Loss (%)", value=5.0, min_value=1.0, max_value=20.0, step=0.5)
            
            if st.button("🚀 Run Backtest", type="primary"):
                with st.spinner("Running backtest..."):
                    final_capital, trades, portfolio_values = run_enhanced_backtest(df)
                    
                    # Calculate metrics
                    total_return = (final_capital - initial_capital) / initial_capital * 100
                    buy_hold_return = (df.iloc[-1]['Close'] - df.iloc[20]['Close']) / df.iloc[20]['Close'] * 100
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Final Capital", f"${final_capital:,.2f}")
                    with col2:
                        st.metric("Total Return", f"{total_return:.2f}%")
                    with col3:
                        st.metric("Buy & Hold Return", f"{buy_hold_return:.2f}%")
                    with col4:
                        st.metric("Number of Trades", len([t for t in trades if t['action'] == 'SELL']))
                    
                    # Portfolio value chart
                    if portfolio_values:
                        portfolio_df = pd.DataFrame(portfolio_values)
                        fig = px.line(portfolio_df, x='date', y='value', 
                                    title='Portfolio Value Over Time')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Trades table
                    if trades:
                        st.subheader("📋 Trade History")
                        trades_df = pd.DataFrame(trades)
                        st.dataframe(trades_df, use_container_width=True)
                        
                        # Trade analysis
                        profitable_trades = [t for t in trades if t.get('profit', 0) > 0]
                        win_rate = len(profitable_trades) / len([t for t in trades if t['action'] == 'SELL']) * 100 if trades else 0
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Win Rate", f"{win_rate:.1f}%")
                        with col2:
                            avg_profit = np.mean([t.get('profit_pct', 0) for t in trades if t['action'] == 'SELL']) if trades else 0
                            st.metric("Average Return per Trade", f"{avg_profit:.2f}%")
        else:
            st.info("👆 Please fetch data first from the Dashboard")
    
    elif page == "Live Trading":
        st.markdown('<h1 class="main-header">⚡ Live Trading</h1>', unsafe_allow_html=True)
        
        # Check if Alpaca credentials are set
        api_key = os.getenv('APCA_API_KEY_ID')
        api_secret = os.getenv('APCA_API_SECRET_KEY')
        
        if not api_key or not api_secret:
            st.error("⚠️ Alpaca API credentials not found. Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.")
            st.info("For paper trading, also set APCA_API_BASE_URL=https://paper-api.alpaca.markets")
            return
        
        if st.session_state.current_data is not None:
            df = st.session_state.current_data
            ticker = st.session_state.current_ticker
            
            # Current position info
            st.subheader("📊 Current Position")
            
            try:
                api = REST(api_key, api_secret, os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets'))
                account = api.get_account()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Account Value", f"${float(account.cash) + float(account.portfolio_value):,.2f}")
                with col2:
                    st.metric("Buying Power", f"${float(account.buying_power):,.2f}")
                with col3:
                    st.metric("Cash", f"${float(account.cash):,.2f}")
                
                # Check current position
                try:
                    position = api.get_position(ticker)
                    st.success(f"Current position: {position.qty} shares of {ticker} at ${float(position.avg_entry_price):.2f}")
                except:
                    st.info(f"No current position in {ticker}")
                
            except Exception as e:
                st.error(f"Error connecting to Alpaca: {str(e)}")
                return
            
            # Trading interface
            st.subheader("🎯 Trading Interface")
            
            # Get current recommendation
            analysis, recommendation = generate_analysis(df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🤖 AI Recommendation")
                if recommendation == "Buy":
                    st.success(f"🟢 {recommendation}")
                elif recommendation == "Sell":
                    st.error(f"🔴 {recommendation}")
                else:
                    st.warning(f"🟡 {recommendation}")
                
                st.text(analysis)
            
            with col2:
                st.subheader("📝 Manual Trade")
                trade_type = st.selectbox("Trade Type", ["Buy", "Sell"])
                quantity = st.number_input("Quantity", min_value=1, value=1)
                
                if st.button(f"🚀 Execute {trade_type} Order", type="primary"):
                    try:
                        if trade_type == "Buy":
                            api.submit_order(
                                symbol=ticker,
                                qty=quantity,
                                side='buy',
                                type='market',
                                time_in_force='gtc'
                            )
                            st.success(f"✅ Buy order submitted for {quantity} shares of {ticker}")
                        else:
                            api.submit_order(
                                symbol=ticker,
                                qty=quantity,
                                side='sell',
                                type='market',
                                time_in_force='gtc'
                            )
                            st.success(f"✅ Sell order submitted for {quantity} shares of {ticker}")
                    except Exception as e:
                        st.error(f"❌ Order failed: {str(e)}")
            
            # Recent orders
            st.subheader("📋 Recent Orders")
            try:
                orders = api.list_orders(status='all', limit=10)
                if orders:
                    order_data = []
                    for order in orders:
                        order_data.append({
                            'Symbol': order.symbol,
                            'Side': order.side,
                            'Quantity': order.qty,
                            'Status': order.status,
                            'Created': order.created_at.strftime('%Y-%m-%d %H:%M:%S')
                        })
                    orders_df = pd.DataFrame(order_data)
                    st.dataframe(orders_df, use_container_width=True)
                else:
                    st.info("No recent orders found")
            except Exception as e:
                st.error(f"Error fetching orders: {str(e)}")
        
        else:
            st.info("👆 Please fetch data first from the Dashboard")
    
    elif page == "Portfolio":
        st.markdown('<h1 class="main-header">💼 Portfolio</h1>', unsafe_allow_html=True)
        
        # Check if Alpaca credentials are set
        api_key = os.getenv('APCA_API_KEY_ID')
        api_secret = os.getenv('APCA_API_SECRET_KEY')
        
        if not api_key or not api_secret:
            st.error("⚠️ Alpaca API credentials not found. Please set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.")
            return
        
        try:
            api = REST(api_key, api_secret, os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets'))
            account = api.get_account()
            
            # Account overview
            st.subheader("📊 Account Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Value", f"${float(account.portfolio_value):,.2f}")
            with col2:
                st.metric("Cash", f"${float(account.cash):,.2f}")
            with col3:
                st.metric("Buying Power", f"${float(account.buying_power):,.2f}")
            with col4:
                st.metric("Day Trade Count", account.daytrade_count)
            
            # Positions
            st.subheader("📈 Current Positions")
            try:
                positions = api.list_positions()
                if positions:
                    position_data = []
                    for pos in positions:
                        current_value = float(pos.qty) * float(pos.current_price)
                        unrealized_pl = float(pos.unrealized_pl)
                        unrealized_pl_pct = (unrealized_pl / (current_value - unrealized_pl)) * 100 if current_value != unrealized_pl else 0
                        
                        position_data.append({
                            'Symbol': pos.symbol,
                            'Quantity': pos.qty,
                            'Entry Price': f"${float(pos.avg_entry_price):.2f}",
                            'Current Price': f"${float(pos.current_price):.2f}",
                            'Market Value': f"${current_value:,.2f}",
                            'Unrealized P&L': f"${unrealized_pl:,.2f}",
                            'P&L %': f"{unrealized_pl_pct:.2f}%"
                        })
                    
                    positions_df = pd.DataFrame(position_data)
                    st.dataframe(positions_df, use_container_width=True)
                    
                    # Portfolio pie chart
                    if len(positions) > 1:
                        fig = px.pie(
                            values=[float(pos.qty) * float(pos.current_price) for pos in positions],
                            names=[pos.symbol for pos in positions],
                            title="Portfolio Allocation"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No current positions")
                    
            except Exception as e:
                st.error(f"Error fetching positions: {str(e)}")
            
            # Performance history
            st.subheader("📊 Performance History")
            try:
                # Get account history
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                
                history = api.get_portfolio_history(
                    start=start_date.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'),
                    timeframe='1D'
                )
                
                if history:
                    # Convert to DataFrame
                    history_data = []
                    for i, timestamp in enumerate(history.timestamp):
                        history_data.append({
                            'Date': pd.to_datetime(timestamp, unit='s'),
                            'Portfolio Value': history.equity[i],
                            'Profit/Loss': history.profit_loss[i],
                            'Profit/Loss %': history.profit_loss_pct[i] * 100
                        })
                    
                    history_df = pd.DataFrame(history_data)
                    
                    # Performance chart
                    fig = px.line(history_df, x='Date', y='Portfolio Value', 
                                title='Portfolio Value Over Time')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        total_return = (history_df['Portfolio Value'].iloc[-1] - history_df['Portfolio Value'].iloc[0]) / history_df['Portfolio Value'].iloc[0] * 100
                        st.metric("30-Day Return", f"{total_return:.2f}%")
                    with col2:
                        max_drawdown = (history_df['Portfolio Value'].max() - history_df['Portfolio Value'].min()) / history_df['Portfolio Value'].max() * 100
                        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
                    with col3:
                        volatility = history_df['Profit/Loss %'].std()
                        st.metric("Volatility", f"{volatility:.2f}%")
                else:
                    st.info("No performance history available")
                    
            except Exception as e:
                st.error(f"Error fetching performance history: {str(e)}")
        
        except Exception as e:
            st.error(f"Error connecting to Alpaca: {str(e)}")
    
    elif page == "Real-Time Trading":
        st.markdown('<h1 class="main-header">🚀 Real-Time Trading</h1>', unsafe_allow_html=True)
        create_real_time_dashboard()
    
    elif page == "Portfolio Management":
        st.markdown('<h1 class="main-header">💼 Portfolio Management</h1>', unsafe_allow_html=True)
        create_portfolio_dashboard()
    
    elif page == "Settings":
        st.markdown('<h1 class="main-header">⚙️ Settings</h1>', unsafe_allow_html=True)
        
        st.subheader("🔑 API Configuration")
        
        # Display current API status
        api_key = os.getenv('APCA_API_KEY_ID')
        api_secret = os.getenv('APCA_API_SECRET_KEY')
        base_url = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("API Key Status", "✅ Set" if api_key else "❌ Not Set")
        with col2:
            st.metric("API Secret Status", "✅ Set" if api_secret else "❌ Not Set")
        
        st.info(f"Base URL: {base_url}")
        
        # Trading parameters
        st.subheader("📊 Trading Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            default_capital = st.number_input("Default Initial Capital ($)", value=10000, min_value=1000, step=1000)
            default_position_size = st.slider("Default Position Size (%)", 10, 100, 100, 10)
        
        with col2:
            default_stop_loss = st.number_input("Default Stop Loss (%)", value=5.0, min_value=1.0, max_value=20.0, step=0.5)
            default_take_profit = st.number_input("Default Take Profit (%)", value=10.0, min_value=1.0, max_value=50.0, step=0.5)
        
        # Risk management
        st.subheader("⚠️ Risk Management")
        
        max_position_size = st.slider("Maximum Position Size (% of portfolio)", 5, 50, 20, 5)
        max_daily_trades = st.number_input("Maximum Daily Trades", value=10, min_value=1, max_value=100)
        enable_stop_loss = st.checkbox("Enable Automatic Stop Loss", value=True)
        enable_take_profit = st.checkbox("Enable Automatic Take Profit", value=True)
        
        # Save settings
        if st.button("💾 Save Settings", type="primary"):
            st.success("Settings saved successfully!")
        
        # Export/Import
        st.subheader("📁 Data Management")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📤 Export Settings"):
                st.info("Settings export functionality coming soon...")
        
        with col2:
            if st.button("📥 Import Settings"):
                st.info("Settings import functionality coming soon...")

if __name__ == "__main__":
    main() 