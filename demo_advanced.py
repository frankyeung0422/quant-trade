#!/usr/bin/env python3
"""
Advanced Quant Trader Demo
Showcases all the new features including real-time trading, portfolio management, and deep learning.
"""

import pandas as pd
import numpy as np
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import track
import time
from datetime import datetime

# Import our modules
from data_fetcher import fetch_data
from indicators import calculate_indicators
from analysis import generate_analysis
from backtest import run_backtest
from improved_backtest import run_improved_backtest
from strategy_comparison import run_strategy_comparison
from ai_optimizer import auto_optimize_all_strategies
from advanced_features import run_advanced_analysis
from enhanced_ml import run_ml_comparison
from improved_ml import run_improved_ml_comparison
from adaptive_strategy import run_adaptive_backtest
from ensemble_strategy import run_ensemble_backtest
from walk_forward import run_walk_forward_validation
from multi_asset import run_multi_asset_analysis
from real_time_trader import RealTimeTrader
from portfolio_manager import PortfolioManager
from deep_learning_strategy import run_deep_learning_analysis

console = Console()

def print_header():
    """Print demo header"""
    console.print("""
[bold blue]╔══════════════════════════════════════════════════════════════════════════════╗
║                    🚀 ADVANCED QUANT TRADER DEMO 🚀                    ║
║                                                                        ║
║  Real-time Trading • Portfolio Management • Deep Learning • AI        ║
╚══════════════════════════════════════════════════════════════════════════════╝
[/bold blue]""")

def demo_basic_analysis():
    """Demo basic analysis features"""
    console.print("\n[bold green]📊 1. BASIC ANALYSIS & BACKTESTING[/bold green]")
    
    # Fetch data
    console.print("Fetching data for AAPL...")
    data = fetch_data("AAPL", "1y")
    data = calculate_indicators(data)
    
    # Generate analysis
    analysis, recommendation = generate_analysis(data)
    console.print(f"[bold yellow]AI Recommendation: {recommendation}[/bold yellow]")
    
    # Run backtest
    console.print("Running backtest...")
    result = run_backtest(data)
    
    # Display results
    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Return", f"{result['total_return']:.2f}%")
    table.add_row("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
    table.add_row("Max Drawdown", f"{result['max_drawdown']:.2f}%")
    table.add_row("Win Rate", f"{result['win_rate']:.2f}%")
    table.add_row("Number of Trades", str(result['num_trades']))
    
    console.print(table)

def demo_advanced_strategies():
    """Demo advanced strategy features"""
    console.print("\n[bold green]🤖 2. ADVANCED STRATEGIES & AI OPTIMIZATION[/bold green]")
    
    data = fetch_data("AAPL", "1y")
    data = calculate_indicators(data)
    
    # Strategy comparison
    console.print("Running strategy comparison...")
    run_strategy_comparison(data)
    
    # AI optimization
    console.print("Running AI parameter optimization...")
    optimization_results = auto_optimize_all_strategies(data, metric='sharpe_ratio', n_calls=20)
    
    table = Table(title="AI Optimization Results")
    table.add_column("Strategy", style="cyan")
    table.add_column("Optimized Score", style="magenta")
    table.add_column("Parameters", style="green")
    
    for strategy_name, (params, score) in optimization_results.items():
        table.add_row(strategy_name, f"{score:.4f}", str(params)[:50] + "...")
    
    console.print(table)

def demo_machine_learning():
    """Demo machine learning features"""
    console.print("\n[bold green]🧠 3. MACHINE LEARNING & DEEP LEARNING[/bold green]")
    
    data = fetch_data("AAPL", "1y")
    data = calculate_indicators(data)
    
    # ML comparison
    console.print("Comparing ML algorithms...")
    ml_results = run_ml_comparison(data)
    
    # Deep learning analysis
    console.print("Running deep learning analysis...")
    dl_results = run_deep_learning_analysis("AAPL", "1y", "lstm")
    
    if 'error' not in dl_results:
        table = Table(title="Deep Learning Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Model Type", dl_results['model_type'])
        table.add_row("Total Return", f"{dl_results['performance']['total_return']:.2f}%")
        table.add_row("Direction Accuracy", f"{dl_results['training_results']['direction_accuracy']:.2f}")
        table.add_row("MSE", f"{dl_results['training_results']['mse']:.6f}")
        table.add_row("Number of Trades", str(dl_results['performance']['num_trades']))
        
        console.print(table)

def demo_portfolio_management():
    """Demo portfolio management features"""
    console.print("\n[bold green]💼 4. PORTFOLIO MANAGEMENT & RISK ANALYTICS[/bold green]")
    
    # Initialize portfolio manager
    pm = PortfolioManager(initial_capital=100000)
    
    # Add sample positions
    console.print("Adding sample positions...")
    pm.add_position('AAPL', 100, 150.0)
    pm.add_position('MSFT', 50, 300.0)
    pm.add_position('GOOGL', 30, 2500.0)
    pm.add_position('TSLA', 20, 200.0)
    pm.add_position('NVDA', 40, 400.0)
    
    # Update prices
    pm.update_prices({
        'AAPL': 155.0,
        'MSFT': 310.0,
        'GOOGL': 2600.0,
        'TSLA': 220.0,
        'NVDA': 450.0
    })
    
    # Get portfolio summary
    summary = pm.get_portfolio_summary()
    
    table = Table(title="Portfolio Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")
    
    table.add_row("Total Value", f"${summary['total_value']:,.2f}")
    table.add_row("Total Return", f"{summary['total_return']:.2%}")
    table.add_row("Cash", f"${summary['cash']:,.2f}")
    table.add_row("Positions", str(summary['num_positions']))
    
    console.print(table)
    
    # Portfolio optimization
    console.print("Running portfolio optimization...")
    optimization = pm.optimize_portfolio(target_volatility=0.15)
    
    if optimization:
        opt_table = Table(title="Portfolio Optimization")
        opt_table.add_column("Metric", style="cyan")
        opt_table.add_column("Value", style="magenta")
        
        opt_table.add_row("Expected Return", f"{optimization['expected_return']:.2%}")
        opt_table.add_row("Expected Volatility", f"{optimization['expected_volatility']:.2%}")
        opt_table.add_row("Sharpe Ratio", f"{optimization['sharpe_ratio']:.2f}")
        
        console.print(opt_table)
        
        # Show optimal weights
        console.print("\n[bold yellow]Optimal Portfolio Weights:[/bold yellow]")
        for ticker, weight in optimization['weights'].items():
            console.print(f"  {ticker}: {weight:.1%}")

def demo_real_time_trading():
    """Demo real-time trading features"""
    console.print("\n[bold green]🚀 5. REAL-TIME TRADING SYSTEM[/bold green]")
    
    console.print("Starting real-time trading simulation...")
    console.print("[yellow]Note: This is a simulation. No real trades will be executed.[/yellow]")
    
    # Create real-time trader
    trader = RealTimeTrader("AAPL", initial_capital=10000)
    
    # Simulate real-time trading for a short period
    console.print("Simulating 2 minutes of real-time trading...")
    
    for i in track(range(2), description="Trading simulation"):
        # Simulate some market activity
        time.sleep(1)
        
        # Get current status
        status = trader.get_portfolio_status()
        
        if i == 1:  # Show final status
            table = Table(title="Real-Time Trading Status")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            
            table.add_row("Total Value", f"${status['current_value']:,.2f}")
            table.add_row("Total Return", f"{status['total_return']:.2f}%")
            table.add_row("Cash", f"${status['capital']:,.2f}")
            table.add_row("Trades", str(status['num_trades']))
            
            console.print(table)

def demo_multi_asset_analysis():
    """Demo multi-asset portfolio analysis"""
    console.print("\n[bold green]📈 6. MULTI-ASSET PORTFOLIO ANALYSIS[/bold green]")
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    console.print(f"Analyzing portfolio: {', '.join(tickers)}")
    
    # Run multi-asset analysis
    portfolio = run_multi_asset_analysis(tickers, period='1y', strategy_type='ensemble')
    
    if portfolio:
        table = Table(title="Multi-Asset Portfolio Results")
        table.add_column("Ticker", style="cyan")
        table.add_column("Return", style="magenta")
        table.add_column("Sharpe", style="green")
        table.add_column("Max DD", style="red")
        
        for ticker, results in portfolio.items():
            if isinstance(results, dict) and 'total_return' in results:
                table.add_row(
                    ticker,
                    f"{results['total_return']:.2f}%",
                    f"{results.get('sharpe_ratio', 0):.2f}",
                    f"{results.get('max_drawdown', 0):.2f}%"
                )
        
        console.print(table)

def demo_walk_forward_validation():
    """Demo walk-forward validation"""
    console.print("\n[bold green]🔬 7. WALK-FORWARD VALIDATION[/bold green]")
    
    data = fetch_data("AAPL", "2y")  # Use 2 years for walk-forward
    data = calculate_indicators(data)
    
    console.print("Running walk-forward validation...")
    results = run_walk_forward_validation(data, strategy_type='ensemble', optimize_params=True)
    
    if results:
        table = Table(title="Walk-Forward Validation Results")
        table.add_column("Period", style="cyan")
        table.add_column("Return", style="magenta")
        table.add_column("Sharpe", style="green")
        table.add_column("Trades", style="yellow")
        
        for period, result in results.items():
            table.add_row(
                period,
                f"{result.get('total_return', 0):.2f}%",
                f"{result.get('sharpe_ratio', 0):.2f}",
                str(result.get('num_trades', 0))
            )
        
        console.print(table)

def demo_advanced_features():
    """Demo advanced features"""
    console.print("\n[bold green]⚡ 8. ADVANCED FEATURES[/bold green]")
    
    data = fetch_data("AAPL", "1y")
    data = calculate_indicators(data)
    
    # Adaptive strategy
    console.print("Running adaptive strategy...")
    adaptive_result = run_adaptive_backtest(data, strategy_type='adaptive')
    
    # Ensemble strategy
    console.print("Running ensemble strategy...")
    ensemble_result = run_ensemble_backtest(data)
    
    # Advanced analysis
    console.print("Running advanced analysis...")
    from strategies import TrendFollowingStrategy
    from ai_optimizer import trend_param_space
    advanced_results = run_advanced_analysis(data, TrendFollowingStrategy, trend_param_space)
    
    table = Table(title="Advanced Strategy Results")
    table.add_column("Strategy", style="cyan")
    table.add_column("Return", style="magenta")
    table.add_column("Sharpe", style="green")
    
    if adaptive_result:
        table.add_row("Adaptive", f"{adaptive_result.get('total_return', 0):.2f}%", 
                     f"{adaptive_result.get('sharpe_ratio', 0):.2f}")
    
    if ensemble_result:
        table.add_row("Ensemble", f"{ensemble_result.get('total_return', 0):.2f}%", 
                     f"{ensemble_result.get('sharpe_ratio', 0):.2f}")
    
    console.print(table)

def main():
    """Run the complete demo"""
    print_header()
    
    try:
        # Run all demos
        demo_basic_analysis()
        demo_advanced_strategies()
        demo_machine_learning()
        demo_portfolio_management()
        demo_real_time_trading()
        demo_multi_asset_analysis()
        demo_walk_forward_validation()
        demo_advanced_features()
        
        console.print("\n[bold green]✅ Demo completed successfully![/bold green]")
        console.print("\n[bold blue]Key Features Demonstrated:[/bold blue]")
        console.print("• 📊 Basic analysis and backtesting")
        console.print("• 🤖 AI-powered strategy optimization")
        console.print("• 🧠 Machine learning and deep learning models")
        console.print("• 💼 Advanced portfolio management")
        console.print("• 🚀 Real-time trading simulation")
        console.print("• 📈 Multi-asset portfolio analysis")
        console.print("• 🔬 Walk-forward validation")
        console.print("• ⚡ Advanced adaptive strategies")
        
        console.print("\n[bold yellow]To run specific features:[/bold yellow]")
        console.print("• python main.py --ticker AAPL --backtest")
        console.print("• python main.py --ticker AAPL --optimize")
        console.print("• python main.py --ticker AAPL --deep-learning --model-type lstm")
        console.print("• python main.py --ticker AAPL --real-time")
        console.print("• python main.py --ticker AAPL --portfolio")
        console.print("• streamlit run web_app.py")
        
    except KeyboardInterrupt:
        console.print("\n[bold red]Demo interrupted by user.[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]Error during demo: {e}[/bold red]")

if __name__ == "__main__":
    main() 