import pandas as pd
import numpy as np
import yfinance as yf
import time
from datetime import datetime, timedelta
import threading
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

@dataclass
class TradeSignal:
    timestamp: datetime
    signal_type: SignalType
    ticker: str
    price: float
    confidence: float
    reason: str
    indicators: Dict

class RealTimeTrader:
    """Real-time trading system with live data streaming and signal generation"""
    
    def __init__(self, ticker: str, initial_capital: float = 10000):
        self.ticker = ticker
        self.capital = initial_capital
        self.position = 0
        self.entry_price = 0
        self.is_running = False
        self.signals = []
        self.trades = []
        self.current_data = pd.DataFrame()
        self.lock = threading.Lock()
        
        # Trading parameters
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.stop_loss_pct = 0.05   # 5% stop loss
        self.take_profit_pct = 0.10 # 10% take profit
        self.min_confidence = 0.7   # Minimum confidence for trade
        
    def start_streaming(self):
        """Start real-time data streaming"""
        self.is_running = True
        logger.info(f"Starting real-time streaming for {self.ticker}")
        
        # Start data collection thread
        data_thread = threading.Thread(target=self._collect_data)
        data_thread.daemon = True
        data_thread.start()
        
        # Start signal generation thread
        signal_thread = threading.Thread(target=self._generate_signals)
        signal_thread.daemon = True
        signal_thread.start()
        
        # Start trading thread
        trading_thread = threading.Thread(target=self._execute_trades)
        trading_thread.daemon = True
        trading_thread.start()
        
    def stop_streaming(self):
        """Stop real-time data streaming"""
        self.is_running = False
        logger.info("Stopping real-time streaming")
        
    def _collect_data(self):
        """Collect real-time market data"""
        while self.is_running:
            try:
                # Get current market data
                ticker_data = yf.Ticker(self.ticker)
                current_info = ticker_data.info
                
                # Get recent price data
                hist = ticker_data.history(period="5d", interval="1m")
                
                if not hist.empty:
                    with self.lock:
                        self.current_data = hist
                        
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error collecting data: {e}")
                time.sleep(60)
                
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for real-time data"""
        if df.empty:
            return df
            
        # Basic indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        df['RSI_14'] = self._calculate_rsi(df['Close'], 14)
        df['MACD'], df['MACD_Signal'] = self._calculate_macd(df['Close'])
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
        
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
        
    def _generate_signals(self):
        """Generate real-time trading signals"""
        while self.is_running:
            try:
                with self.lock:
                    if len(self.current_data) < 50:
                        time.sleep(10)
                        continue
                        
                    # Calculate indicators
                    data = self._calculate_indicators(self.current_data.copy())
                    
                    if data.empty:
                        time.sleep(10)
                        continue
                        
                    # Generate signal
                    signal = self._analyze_market(data)
                    
                    if signal:
                        self.signals.append(signal)
                        logger.info(f"Signal generated: {signal.signal_type.value} at ${signal.price:.2f}")
                        
                time.sleep(30)  # Check for signals every 30 seconds
                
            except Exception as e:
                logger.error(f"Error generating signals: {e}")
                time.sleep(30)
                
    def _analyze_market(self, df: pd.DataFrame) -> Optional[TradeSignal]:
        """Analyze market conditions and generate trading signal"""
        if df.empty or len(df) < 50:
            return None
            
        current = df.iloc[-1]
        previous = df.iloc[-2] if len(df) > 1 else current
        
        # Calculate confidence score
        confidence = 0
        confirmations = []
        
        # Price action
        price_up = current['Close'] > previous['Close']
        confirmations.append(price_up)
        
        # RSI analysis
        rsi = current['RSI_14']
        rsi_oversold = rsi < 30
        rsi_overbought = rsi > 70
        rsi_neutral = 30 <= rsi <= 70
        confirmations.append(rsi_neutral)
        
        # MACD analysis
        macd_bullish = current['MACD'] > current['MACD_Signal'] and current['MACD'] > 0
        macd_bearish = current['MACD'] < current['MACD_Signal'] and current['MACD'] < 0
        confirmations.append(macd_bullish)
        
        # Volume analysis
        volume_high = current['Volume_Ratio'] > 1.2
        confirmations.append(volume_high)
        
        # Moving average analysis
        price_above_sma = current['Close'] > current['SMA_20']
        sma_bullish = current['SMA_20'] > current['EMA_20']
        confirmations.extend([price_above_sma, sma_bullish])
        
        # Bollinger Bands analysis
        bb_position = (current['Close'] - current['BB_Low']) / (current['BB_Upper'] - current['BB_Low'])
        bb_middle = 0.3 <= bb_position <= 0.7
        confirmations.append(bb_middle)
        
        # Calculate confidence
        confidence = sum(confirmations) / len(confirmations)
        
        # Generate signal
        signal_type = SignalType.HOLD
        reason = "No clear signal"
        
        if confidence >= self.min_confidence:
            if macd_bullish and price_above_sma and volume_high:
                signal_type = SignalType.BUY
                reason = "Bullish momentum with volume confirmation"
            elif macd_bearish and not price_above_sma and rsi_overbought:
                signal_type = SignalType.SELL
                reason = "Bearish momentum with overbought RSI"
                
        if signal_type != SignalType.HOLD:
            return TradeSignal(
                timestamp=datetime.now(),
                signal_type=signal_type,
                ticker=self.ticker,
                price=current['Close'],
                confidence=confidence,
                reason=reason,
                indicators={
                    'rsi': rsi,
                    'macd': current['MACD'],
                    'volume_ratio': current['Volume_Ratio'],
                    'bb_position': bb_position
                }
            )
            
        return None
        
    def _execute_trades(self):
        """Execute trades based on signals"""
        while self.is_running:
            try:
                with self.lock:
                    if not self.signals:
                        time.sleep(5)
                        continue
                        
                    signal = self.signals.pop(0)
                    
                    # Execute trade logic
                    if signal.signal_type == SignalType.BUY and self.position == 0:
                        self._execute_buy(signal)
                    elif signal.signal_type == SignalType.SELL and self.position > 0:
                        self._execute_sell(signal)
                        
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error executing trades: {e}")
                time.sleep(5)
                
    def _execute_buy(self, signal: TradeSignal):
        """Execute buy order"""
        # Calculate position size based on risk
        risk_amount = self.capital * self.risk_per_trade
        stop_loss_price = signal.price * (1 - self.stop_loss_pct)
        risk_per_share = signal.price - stop_loss_price
        shares = int(risk_amount / risk_per_share)
        
        if shares > 0:
            cost = shares * signal.price
            if cost <= self.capital:
                self.position = shares
                self.entry_price = signal.price
                self.capital -= cost
                
                trade = {
                    'timestamp': signal.timestamp,
                    'action': 'BUY',
                    'shares': shares,
                    'price': signal.price,
                    'cost': cost,
                    'confidence': signal.confidence,
                    'reason': signal.reason
                }
                self.trades.append(trade)
                
                logger.info(f"BUY executed: {shares} shares at ${signal.price:.2f}")
                
    def _execute_sell(self, signal: TradeSignal):
        """Execute sell order"""
        if self.position > 0:
            proceeds = self.position * signal.price
            self.capital += proceeds
            
            # Calculate P&L
            pnl = proceeds - (self.position * self.entry_price)
            pnl_pct = (pnl / (self.position * self.entry_price)) * 100
            
            trade = {
                'timestamp': signal.timestamp,
                'action': 'SELL',
                'shares': self.position,
                'price': signal.price,
                'proceeds': proceeds,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'confidence': signal.confidence,
                'reason': signal.reason
            }
            self.trades.append(trade)
            
            logger.info(f"SELL executed: {self.position} shares at ${signal.price:.2f}, P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
            
            # Reset position
            self.position = 0
            self.entry_price = 0
            
    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        with self.lock:
            current_value = self.capital
            if self.position > 0 and not self.current_data.empty:
                current_price = self.current_data['Close'].iloc[-1]
                current_value += self.position * current_price
                
            return {
                'capital': self.capital,
                'position': self.position,
                'entry_price': self.entry_price,
                'current_value': current_value,
                'total_return': ((current_value - 10000) / 10000) * 100,
                'num_trades': len(self.trades),
                'last_signal': self.signals[-1] if self.signals else None
            }
            
    def get_trade_history(self) -> List[Dict]:
        """Get complete trade history"""
        return self.trades.copy()

# Example usage
if __name__ == "__main__":
    trader = RealTimeTrader("AAPL", initial_capital=10000)
    
    try:
        trader.start_streaming()
        
        # Monitor for 10 minutes
        for i in range(10):
            time.sleep(60)
            status = trader.get_portfolio_status()
            print(f"Portfolio Status: {status}")
            
    except KeyboardInterrupt:
        print("Stopping trader...")
        trader.stop_streaming()
        
        # Print final results
        status = trader.get_portfolio_status()
        print(f"Final Portfolio Status: {status}")
        
        trades = trader.get_trade_history()
        print(f"Total Trades: {len(trades)}")
        for trade in trades:
            print(f"Trade: {trade}") 