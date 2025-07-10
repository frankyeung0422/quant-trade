#!/usr/bin/env python3
"""
Quant Trader Pro - Demo Script
A demonstration of the quant trading functionality without the web interface.
"""

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from data_fetcher import fetch_data
from indicators import calculate_indicators
from analysis import generate_analysis
from backtest import run_backtest

console = Console()

def demo_analysis(ticker: str = "AAPL", period: str = "6mo"):
    """Demonstrate stock analysis functionality."""
    console.print(f"\n[bold blue]📊 Analyzing {ticker} ({period})[/bold blue]")
    console.print("=" * 60)
    
    # Fetch data
    console.print("[yellow]Fetching data...[/yellow]")
    data = fetch_data(ticker, period)
    if data.empty:
        console.print(f"[red]No data found for {ticker}![/red]")
        return
    
    # Calculate indicators
    console.print("[yellow]Calculating technical indicators...[/yellow]")
    data = calculate_indicators(data)
    
    # Generate analysis
    console.print("[yellow]Generating analysis and recommendations...[/yellow]")
    analysis, recommendation = generate_analysis(data)
    
    # Display results
    latest = data.iloc[-1]
    
    # Create metrics table
    table = Table(title=f"{ticker} Analysis Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Status", style="green")
    
    # Price metrics
    current_price = latest['Close']
    price_change = current_price - data.iloc[-2]['Close']
    price_change_pct = (price_change / data.iloc[-2]['Close']) * 100
    
    table.add_row("Current Price", f"${current_price:.2f}", 
                  f"{'📈' if price_change > 0 else '📉'} {price_change_pct:+.2f}%")
    
    # Technical indicators
    rsi_status = "Overbought 🔴" if latest['RSI_14'] > 70 else "Oversold 🟢" if latest['RSI_14'] < 30 else "Neutral 🟡"
    table.add_row("RSI (14)", f"{latest['RSI_14']:.1f}", rsi_status)
    
    macd_status = "Bullish 🟢" if latest['MACD'] > 0 else "Bearish 🔴"
    table.add_row("MACD", f"{latest['MACD']:.3f}", macd_status)
    
    trend_status = "Uptrend 📈" if current_price > latest['SMA_20'] else "Downtrend 📉"
    table.add_row("Trend", f"SMA20: ${latest['SMA_20']:.2f}", trend_status)
    
    console.print(table)
    
    # Recommendation
    console.print(f"\n[bold]🤖 AI Recommendation:[/bold]")
    if recommendation == "Buy":
        console.print(f"[bold green]🟢 {recommendation} - Strong bullish signals detected[/bold green]")
    elif recommendation == "Sell":
        console.print(f"[bold red]🔴 {recommendation} - Bearish signals detected[/bold red]")
    else:
        console.print(f"[bold yellow]🟡 {recommendation} - Neutral signals, wait for better entry[/bold yellow]")
    
    # Detailed analysis
    console.print(f"\n[bold]📋 Detailed Analysis:[/bold]")
    console.print(analysis)
    
    return data, recommendation

def demo_backtest(data, ticker: str):
    """Demonstrate backtesting functionality."""
    console.print(f"\n[bold blue]🧪 Backtesting {ticker}[/bold blue]")
    console.print("=" * 60)
    
    # Run enhanced backtest
    capital = 10000
    position = 0
    entry_price = 0
    trades = []
    
    for i in range(20, len(data)):
        sub_df = data.iloc[:i+1]
        _, rec = generate_analysis(sub_df)
        price = data.iloc[i]['Close']
        date = data.index[i]
        
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
    
    # Final liquidation
    if position > 0:
        final_price = data.iloc[-1]['Close']
        capital = position * final_price
        profit = final_price - entry_price
        profit_pct = (profit / entry_price) * 100
        trades.append({
            'date': data.index[-1],
            'action': 'SELL',
            'price': final_price,
            'shares': position,
            'profit': profit,
            'profit_pct': profit_pct
        })
    
    # Calculate metrics
    total_return = (capital - 10000) / 10000 * 100
    buy_hold_return = (data.iloc[-1]['Close'] - data.iloc[20]['Close']) / data.iloc[20]['Close'] * 100
    
    # Display results
    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Initial Capital", "$10,000.00")
    table.add_row("Final Capital", f"${capital:,.2f}")
    table.add_row("Total Return", f"{total_return:.2f}%")
    table.add_row("Buy & Hold Return", f"{buy_hold_return:.2f}%")
    table.add_row("Number of Trades", str(len([t for t in trades if t['action'] == 'SELL'])))
    
    # Win rate
    profitable_trades = [t for t in trades if t.get('profit', 0) > 0]
    win_rate = len(profitable_trades) / len([t for t in trades if t['action'] == 'SELL']) * 100 if trades else 0
    table.add_row("Win Rate", f"{win_rate:.1f}%")
    
    console.print(table)
    
    # Trade history
    if trades:
        console.print(f"\n[bold]📋 Trade History:[/bold]")
        trade_table = Table()
        trade_table.add_column("Date", style="cyan")
        trade_table.add_column("Action", style="green")
        trade_table.add_column("Price", style="yellow")
        trade_table.add_column("Shares", style="blue")
        trade_table.add_column("Profit/Loss", style="magenta")
        
        for trade in trades[-5:]:  # Show last 5 trades
            profit_str = f"${trade.get('profit', 0):.2f}" if trade.get('profit') else "-"
            trade_table.add_row(
                trade['date'].strftime('%Y-%m-%d'),
                trade['action'],
                f"${trade['price']:.2f}",
                f"{trade['shares']:.2f}",
                profit_str
            )
        
        console.print(trade_table)

def demo_multiple_stocks():
    """Demonstrate analysis of multiple stocks."""
    console.print(f"\n[bold blue]📈 Multi-Stock Analysis[/bold blue]")
    console.print("=" * 60)
    
    stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    results = []
    
    for stock in stocks:
        try:
            data = fetch_data(stock, '1mo')
            if not data.empty:
                data = calculate_indicators(data)
                _, recommendation = generate_analysis(data)
                current_price = data.iloc[-1]['Close']
                results.append({
                    'Ticker': stock,
                    'Price': f"${current_price:.2f}",
                    'Recommendation': recommendation
                })
        except Exception as e:
            console.print(f"[red]Error analyzing {stock}: {e}[/red]")
    
    if results:
        table = Table(title="Multi-Stock Analysis Results")
        table.add_column("Ticker", style="cyan")
        table.add_column("Current Price", style="yellow")
        table.add_column("Recommendation", style="green")
        
        for result in results:
            rec_emoji = "🟢" if result['Recommendation'] == 'Buy' else "🔴" if result['Recommendation'] == 'Sell' else "🟡"
            table.add_row(result['Ticker'], result['Price'], f"{rec_emoji} {result['Recommendation']}")
        
        console.print(table)

def main():
    """Main demo function."""
    console.print("[bold green]🚀 Quant Trader Pro - Demo[/bold green]")
    console.print("=" * 60)
    console.print("This demo showcases the key features of the quant trading system.")
    console.print("=" * 60)
    
    # Demo 1: Single stock analysis
    data, recommendation = demo_analysis("AAPL", "6mo")
    
    # Demo 2: Backtesting
    demo_backtest(data, "AAPL")
    
    # Demo 3: Multiple stocks
    demo_multiple_stocks()
    
    console.print(f"\n[bold green]✅ Demo completed![/bold green]")
    console.print("To use the full web interface, run: [cyan]streamlit run web_app.py[/cyan]")
    console.print("Or use the launcher: [cyan]python run_app.py[/cyan]")

if __name__ == "__main__":
    main() 