import pandas as pd
import numpy as np

def find_support_resistance(df, window=20, threshold=0.02):
    """Find support and resistance levels"""
    highs = df['High'].rolling(window=window, center=True).max()
    lows = df['Low'].rolling(window=window, center=True).min()
    
    # Find local maxima and minima
    resistance_levels = []
    support_levels = []
    
    for i in range(window, len(df) - window):
        if df['High'].iloc[i] == highs.iloc[i]:
            resistance_levels.append(df['High'].iloc[i])
        if df['Low'].iloc[i] == lows.iloc[i]:
            support_levels.append(df['Low'].iloc[i])
    
    return support_levels, resistance_levels

def calculate_volume_profile(df, window=20):
    """Calculate volume-weighted average price and volume zones"""
    df['VWAP'] = (df['Close'] * df['Volume']).rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
    df['Volume_SMA'] = df['Volume'].rolling(window=window).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    return df

def detect_divergence(df, window=14):
    """Detect RSI divergence with price"""
    price_highs = df['High'].rolling(window=window, center=True).max()
    price_lows = df['Low'].rolling(window=window, center=True).min()
    rsi_highs = df['RSI_14'].rolling(window=window, center=True).max()
    rsi_lows = df['RSI_14'].rolling(window=window, center=True).min()
    
    # Bullish divergence: price makes lower lows, RSI makes higher lows
    bullish_divergence = (price_lows.iloc[-1] < price_lows.iloc[-2]) and (rsi_lows.iloc[-1] > rsi_lows.iloc[-2])
    
    # Bearish divergence: price makes higher highs, RSI makes lower highs
    bearish_divergence = (price_highs.iloc[-1] > price_highs.iloc[-2]) and (rsi_highs.iloc[-1] < rsi_highs.iloc[-2])
    
    return bullish_divergence, bearish_divergence

def generate_analysis(df: pd.DataFrame):
    """
    Generate a comprehensive analysis and trading recommendation based on multiple indicators.
    Returns (report: str, recommendation: str)
    """
    latest = df.iloc[-1]
    report = []
    
    # Calculate additional indicators
    df = calculate_volume_profile(df)
    support_levels, resistance_levels = find_support_resistance(df)
    bullish_div, bearish_div = detect_divergence(df)
    
    # Ensure Volume_Ratio exists and is not NaN
    if 'Volume_Ratio' not in df.columns:
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    # Fill missing or NaN values with 1.0 (neutral)
    df['Volume_Ratio'] = df['Volume_Ratio'].fillna(1.0)
    
    # 1. Trend Analysis
    trend_strength = 0
    if latest['Close'] > latest['SMA_20']:
        trend = 'uptrend'
        trend_strength += 1
    else:
        trend = 'downtrend'
        trend_strength -= 1
    
    if latest['Close'] > latest['EMA_20']:
        trend_strength += 1
    else:
        trend_strength -= 1
    
    # ADX for trend strength
    if latest['ADX'] > 25:
        trend_strength += 1
        report.append(f"Strong trend detected (ADX: {latest['ADX']:.2f})")
    else:
        report.append(f"Weak trend detected (ADX: {latest['ADX']:.2f})")
    
    report.append(f"Trend: {trend} (Close: {latest['Close']:.2f}, SMA20: {latest['SMA_20']:.2f})")
    
    # 2. Momentum Analysis
    momentum_score = 0
    
    # RSI
    if latest['RSI_14'] > 70:
        momentum = 'overbought'
        momentum_score -= 2
    elif latest['RSI_14'] < 30:
        momentum = 'oversold'
        momentum_score += 2
    else:
        momentum = 'neutral'
        if latest['RSI_14'] > 50:
            momentum_score += 1
        else:
            momentum_score -= 1
    
    report.append(f"Momentum: {momentum} (RSI14: {latest['RSI_14']:.2f})")
    
    # Stochastic
    if latest['STOCH_K'] < 20:
        momentum_score += 1
    elif latest['STOCH_K'] > 80:
        momentum_score -= 1
    
    # 3. MACD Analysis
    macd_score = 0
    if latest['MACD'] > latest['MACD_Signal']:
        macd_trend = 'bullish'
        macd_score += 1
    else:
        macd_trend = 'bearish'
        macd_score -= 1
    
    if latest['MACD'] > 0:
        macd_score += 1
    
    report.append(f"MACD: {macd_trend} (MACD: {latest['MACD']:.2f}, Signal: {latest['MACD_Signal']:.2f})")
    
    # 4. Volume Analysis
    volume_score = 0
    volume_ratio = latest['Volume_Ratio'] if 'Volume_Ratio' in latest and not pd.isna(latest['Volume_Ratio']) else 1.0
    if volume_ratio > 1.5:
        volume_score += 2
        report.append(f"High volume detected (Ratio: {volume_ratio:.2f})")
    elif volume_ratio > 1.2:
        volume_score += 1
        report.append(f"Above average volume (Ratio: {volume_ratio:.2f})")
    else:
        report.append(f"Low volume (Ratio: {volume_ratio:.2f})")
    
    # 5. Support/Resistance Analysis
    current_price = latest['Close']
    nearest_support = max([s for s in support_levels if s < current_price], default=0)
    nearest_resistance = min([r for r in resistance_levels if r > current_price], default=float('inf'))
    
    if nearest_support > 0:
        support_distance = (current_price - nearest_support) / current_price * 100
        report.append(f"Nearest support: ${nearest_support:.2f} ({support_distance:.1f}% away)")
    
    if nearest_resistance != float('inf'):
        resistance_distance = (nearest_resistance - current_price) / current_price * 100
        report.append(f"Nearest resistance: ${nearest_resistance:.2f} ({resistance_distance:.1f}% away)")
    
    # 6. Divergence Analysis
    if bullish_div:
        report.append("BULLISH DIVERGENCE DETECTED - Potential reversal signal")
        momentum_score += 2
    elif bearish_div:
        report.append("BEARISH DIVERGENCE DETECTED - Potential reversal signal")
        momentum_score -= 2
    
    # 7. Bollinger Bands Analysis
    bb_position = (current_price - latest['BB_Low']) / (latest['BB_High'] - latest['BB_Low'])
    if bb_position < 0.2:
        report.append("Price near lower Bollinger Band - potential oversold")
        momentum_score += 1
    elif bb_position > 0.8:
        report.append("Price near upper Bollinger Band - potential overbought")
        momentum_score -= 1
    
    # 8. CCI Analysis
    if latest['CCI'] > 100:
        report.append("CCI indicates overbought conditions")
        momentum_score -= 1
    elif latest['CCI'] < -100:
        report.append("CCI indicates oversold conditions")
        momentum_score += 1
    
    # 9. Williams %R Analysis
    if latest['WILLR'] < -80:
        report.append("Williams %R indicates oversold")
        momentum_score += 1
    elif latest['WILLR'] > -20:
        report.append("Williams %R indicates overbought")
        momentum_score -= 1
    
    # Calculate overall score
    total_score = trend_strength + momentum_score + macd_score + volume_score
    
    # Enhanced recommendation logic
    if total_score >= 4 and volume_score > 0:
        recommendation = 'Buy'
        confidence = 'High'
    elif total_score >= 2:
        recommendation = 'Buy'
        confidence = 'Medium'
    elif total_score <= -4:
        recommendation = 'Sell'
        confidence = 'High'
    elif total_score <= -2:
        recommendation = 'Sell'
        confidence = 'Medium'
    else:
        recommendation = 'Hold'
        confidence = 'Low'
    
    report.append(f"Overall Score: {total_score}")
    report.append(f"Recommendation: {recommendation} (Confidence: {confidence})")
    
    return '\n'.join(report), recommendation 