#!/usr/bin/env python3
"""
Test script for improved backtest with better returns
"""

import yfinance as yf
import pandas as pd
from improved_backtest import run_improved_backtest, run_strategy_comparison, print_improved_results

def test_improved_backtest():
    """Test the improved backtest with real stock data"""
    
    print("🚀 Testing Improved Backtest with Better Returns")
    print("="*60)
    
    # Download stock data
    ticker = "AAPL"
    print(f"📈 Downloading {ticker} data...")
    
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y")
    
    if df.empty:
        print("❌ No data downloaded!")
        return
    
    print(f"✅ Downloaded {len(df)} days of data")
    print(f"📊 Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    
    # Test different strategies
    print("\n🔍 Testing different strategies...")
    
    strategies = ['ml', 'momentum', 'mean_reversion', 'combined']
    results = {}
    
    for strategy in strategies:
        print(f"\n📊 Testing {strategy.upper()} strategy...")
        try:
            result = run_improved_backtest(df, strategy=strategy, initial_capital=10000)
            results[strategy] = result
            print_improved_results(result)
        except Exception as e:
            print(f"❌ Error testing {strategy}: {str(e)}")
    
    # Find best strategy
    if results:
        best_strategy = max(results.keys(), key=lambda x: results[x].total_return)
        best_result = results[best_strategy]
        
        print("\n" + "="*60)
        print("🏆 BEST PERFORMING STRATEGY")
        print("="*60)
        print(f"Strategy: {best_strategy.upper()}")
        print(f"Total Return: {best_result.total_return:.2f}%")
        print(f"Annualized Return: {best_result.annualized_return:.2f}%")
        print(f"Sharpe Ratio: {best_result.sharpe_ratio:.2f}")
        print(f"Max Drawdown: {best_result.max_drawdown:.2f}%")
        print(f"Win Rate: {best_result.win_rate:.1f}%")
        print(f"Number of Trades: {best_result.num_trades}")
        
        # Compare with buy and hold
        buy_hold_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
        print(f"\n📈 Buy & Hold Return: {buy_hold_return:.2f}%")
        print(f"🚀 Strategy vs Buy & Hold: {best_result.total_return - buy_hold_return:.2f}%")
        
        if best_result.total_return > buy_hold_return:
            print("✅ Strategy outperforms buy & hold!")
        else:
            print("⚠️ Strategy underperforms buy & hold")
    
    return results

def test_multiple_stocks():
    """Test the strategy on multiple stocks"""
    
    print("\n" + "="*60)
    print("📊 TESTING MULTIPLE STOCKS")
    print("="*60)
    
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX']
    results_summary = []
    
    for ticker in stocks:
        print(f"\n📈 Testing {ticker}...")
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="1y")
            
            if not df.empty:
                # Test combined strategy (usually best)
                result = run_improved_backtest(df, strategy='combined', initial_capital=10000)
                
                results_summary.append({
                    'Ticker': ticker,
                    'Total Return': result.total_return,
                    'Annualized Return': result.annualized_return,
                    'Sharpe Ratio': result.sharpe_ratio,
                    'Max Drawdown': result.max_drawdown,
                    'Win Rate': result.win_rate,
                    'Num Trades': result.num_trades
                })
                
                print(f"  Return: {result.total_return:.2f}% | Sharpe: {result.sharpe_ratio:.2f} | Trades: {result.num_trades}")
            else:
                print(f"  ❌ No data for {ticker}")
                
        except Exception as e:
            print(f"  ❌ Error testing {ticker}: {str(e)}")
    
    # Summary
    if results_summary:
        df_summary = pd.DataFrame(results_summary)
        print("\n" + "="*60)
        print("📊 SUMMARY RESULTS")
        print("="*60)
        print(df_summary.to_string(index=False))
        
        avg_return = df_summary['Total Return'].mean()
        avg_sharpe = df_summary['Sharpe Ratio'].mean()
        print(f"\n📈 Average Return: {avg_return:.2f}%")
        print(f"📊 Average Sharpe Ratio: {avg_sharpe:.2f}")
        
        # Best performing stock
        best_stock = df_summary.loc[df_summary['Total Return'].idxmax()]
        print(f"🏆 Best Stock: {best_stock['Ticker']} ({best_stock['Total Return']:.2f}%)")

if __name__ == "__main__":
    # Test single stock with detailed results
    test_improved_backtest()
    
    # Test multiple stocks
    test_multiple_stocks() 