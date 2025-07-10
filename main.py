import argparse
from rich import print
from data_fetcher import fetch_data
from indicators import calculate_indicators
from analysis import generate_analysis
from backtest import run_backtest, print_backtest_results
from improved_backtest import run_improved_backtest, print_improved_backtest_results
from strategy_comparison import run_strategy_comparison
from ai_optimizer import auto_optimize_all_strategies
from advanced_features import run_advanced_analysis
from enhanced_ml import run_ml_comparison, EnhancedMLStrategy, EnsembleMLStrategy
from improved_ml import run_improved_ml_comparison, ImprovedMLStrategy
from adaptive_strategy import run_adaptive_backtest
from ensemble_strategy import run_ensemble_backtest
from walk_forward import run_walk_forward_validation
from improved_optimizer import optimize_improved_backtest
from multi_asset import run_multi_asset_analysis
from visualization import save_all_plots
from quant_trader import execute_trades
from real_time_trader import RealTimeTrader
from portfolio_manager import PortfolioManager
from deep_learning_strategy import run_deep_learning_analysis

def main():
    parser = argparse.ArgumentParser(description='Stock Quant Trader')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--period', type=str, default='1y', help='Data period (e.g., 1y, 6mo, 1mo)')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--compare', action='store_true', help='Compare multiple strategies')
    parser.add_argument('--ml', action='store_true', help='Use ML-based strategy')
    parser.add_argument('--ml-compare', action='store_true', help='Compare different ML algorithms')
    parser.add_argument('--improved-ml', action='store_true', help='Use improved ML strategy with better features')
    parser.add_argument('--ensemble-ml', action='store_true', help='Use ensemble ML strategy')
    parser.add_argument('--improved-backtest', action='store_true', help='Run improved backtest with better risk management')
    parser.add_argument('--adaptive', action='store_true', help='Run adaptive strategy based on market regime')
    parser.add_argument('--multi-timeframe', action='store_true', help='Run multi-timeframe strategy')
    parser.add_argument('--ensemble', action='store_true', help='Run ensemble strategy combining multiple approaches')
    parser.add_argument('--optimize', action='store_true', help='Optimize strategy parameters using Bayesian optimization')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward validation for robustness')
    parser.add_argument('--multi-asset', nargs='+', help='Run multi-asset portfolio analysis (provide tickers)')
    parser.add_argument('--plot', action='store_true', help='Generate and save performance plots')
    parser.add_argument('--advanced', action='store_true', help='Run advanced analysis with walk-forward validation')
    parser.add_argument('--trade', action='store_true', help='Execute trades (requires broker API setup)')
    parser.add_argument('--real-time', action='store_true', help='Start real-time trading system')
    parser.add_argument('--deep-learning', action='store_true', help='Run deep learning analysis with LSTM/Transformer models')
    parser.add_argument('--portfolio', action='store_true', help='Run portfolio management analysis')
    parser.add_argument('--model-type', type=str, default='lstm', choices=['lstm', 'transformer', 'ensemble'], help='Deep learning model type')
    args = parser.parse_args()

    print(f"[bold green]Fetching data for {args.ticker} ({args.period})...[/bold green]")
    data = fetch_data(args.ticker, args.period)
    if data.empty:
        print(f"[bold red]No data found for {args.ticker}![/bold red]")
        return

    print("[bold blue]Calculating indicators...[/bold blue]")
    data = calculate_indicators(data)

    print("[bold yellow]Generating analysis and recommendations...[/bold yellow]")
    analysis, recommendation = generate_analysis(data)
    print(analysis)
    print(f"[bold magenta]Recommendation: {recommendation}[/bold magenta]")

    if args.backtest:
        print("[bold cyan]Running backtest...[/bold cyan]")
        result = run_backtest(data)
        print_backtest_results(result)

    if args.compare:
        print("[bold magenta]Running strategy comparison...[/bold magenta]")
        run_strategy_comparison(data)

    if args.optimize:
        print("[bold green]Running AI parameter optimization...[/bold green]")
        optimization_results = auto_optimize_all_strategies(data, metric='sharpe_ratio', n_calls=50)
        print("\n[bold yellow]Optimization Results:[/bold yellow]")
        for strategy_name, (params, score) in optimization_results.items():
            print(f"{strategy_name}: {score:.4f} (params: {params})")

    if args.ml:
        print("[bold cyan]Running ML strategy backtest...[/bold cyan]")
        from strategies import MLStrategy
        ml_strategy = MLStrategy()
        # Create a copy of data with ML signals
        ml_data = data.copy()
        ml_data['Strategy_Signal'] = [ml_strategy.generate_signal(ml_data, i) for i in range(len(ml_data))]
        # Run backtest with ML signals
        result = run_backtest(ml_data)
        print_backtest_results(result)

    if args.ml_compare:
        print("[bold magenta]Comparing ML algorithms...[/bold magenta]")
        ml_results = run_ml_comparison(data)
        print("\n[bold green]ML Comparison Complete![/bold green]")

    if args.improved_ml:
        print("[bold cyan]Running improved ML strategy...[/bold cyan]")
        improved_ml = ImprovedMLStrategy(algorithm='random_forest')
        improved_ml.train(data, target_type='smart_direction')
        # Create a copy of data with improved ML signals
        improved_data = data.copy()
        improved_data['Strategy_Signal'] = [
            improved_ml.generate_signal(improved_data, i) for i in range(len(improved_data))
        ]
        # Run improved backtest with ML signals
        result = run_improved_backtest(improved_data)
        print_improved_backtest_results(result)

    if args.improved_backtest:
        print("[bold blue]Running improved backtest...[/bold blue]")
        result = run_improved_backtest(data)
        print_improved_backtest_results(result)

    if args.adaptive:
        print("[bold green]Running adaptive strategy...[/bold green]")
        result = run_adaptive_backtest(data, strategy_type='adaptive')
        print_improved_backtest_results(result)

    if args.multi_timeframe:
        print("[bold purple]Running multi-timeframe strategy...[/bold purple]")
        result = run_adaptive_backtest(data, strategy_type='multi_timeframe')
        print_improved_backtest_results(result)

    if args.ensemble:
        print("[bold cyan]Running ensemble strategy...[/bold cyan]")
        result = run_ensemble_backtest(data)
        print_improved_backtest_results(result)

    if args.optimize:
        print("[bold yellow]Optimizing strategy parameters...[/bold yellow]")
        best_params, best_score, _ = optimize_improved_backtest(data, n_calls=30, maximize_metric='total_return')
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score:.2f}%")
        
        # Run backtest with optimized parameters
        result = run_improved_backtest(data, **best_params)
        print_improved_backtest_results(result)

    if args.walk_forward:
        print("[bold magenta]Running walk-forward validation...[/bold magenta]")
        results = run_walk_forward_validation(data, strategy_type='ensemble', optimize_params=True)

    if args.multi_asset:
        print(f"[bold green]Running multi-asset analysis on {args.multi_asset}...[/bold green]")
        portfolio = run_multi_asset_analysis(args.multi_asset, period=args.period, strategy_type='ensemble')

    if args.plot:
        print("[bold blue]Generating performance plots...[/bold blue]")
        if 'result' in locals():
            save_all_plots(result, data, f"results_{args.ticker}")

    if args.ensemble_ml:
        print("[bold yellow]Running ensemble ML strategy...[/bold yellow]")
        ensemble_ml = EnsembleMLStrategy()
        # Create a copy of data with ensemble ML signals
        ensemble_data = data.copy()
        ensemble_data['Strategy_Signal'] = [
            ensemble_ml.generate_signal(ensemble_data, i) for i in range(len(ensemble_data))
        ]
        # Run backtest with ensemble ML signals
        result = run_backtest(ensemble_data)
        print_backtest_results(result)

    if args.advanced:
        print("[bold purple]Running advanced analysis...[/bold purple]")
        from strategies import TrendFollowingStrategy
        from ai_optimizer import trend_param_space
        advanced_results = run_advanced_analysis(data, TrendFollowingStrategy, trend_param_space)
        print("\n[bold green]Advanced Analysis Complete![/bold green]")

    if args.trade:
        print("[bold red]Executing trades...[/bold red]")
        execute_trades(recommendation, args.ticker)

    if args.real_time:
        print("[bold green]Starting real-time trading system...[/bold green]")
        trader = RealTimeTrader(args.ticker, initial_capital=10000)
        try:
            trader.start_streaming()
            print("[bold yellow]Real-time trading started. Press Ctrl+C to stop.[/bold yellow]")
            
            # Monitor for 10 minutes
            import time
            for i in range(10):
                time.sleep(60)
                status = trader.get_portfolio_status()
                print(f"Portfolio Status: {status}")
                
        except KeyboardInterrupt:
            print("[bold red]Stopping real-time trading...[/bold red]")
            trader.stop_streaming()
            
            # Print final results
            status = trader.get_portfolio_status()
            print(f"Final Portfolio Status: {status}")
            
            trades = trader.get_trade_history()
            print(f"Total Trades: {len(trades)}")
            for trade in trades:
                print(f"Trade: {trade}")

    if args.deep_learning:
        print(f"[bold purple]Running deep learning analysis with {args.model_type} model...[/bold purple]")
        results = run_deep_learning_analysis(args.ticker, args.period, args.model_type)
        
        if 'error' not in results:
            print(f"\n[bold green]Deep Learning Results:[/bold green]")
            print(f"Model Type: {results['model_type']}")
            print(f"Total Return: {results['performance']['total_return']:.2f}%")
            print(f"Number of Trades: {results['performance']['num_trades']}")
            print(f"Win Rate: {results['performance']['win_rate']:.2f}%")
            print(f"Direction Accuracy: {results['training_results']['direction_accuracy']:.2f}")
            print(f"MSE: {results['training_results']['mse']:.6f}")
            print(f"MAE: {results['training_results']['mae']:.6f}")
        else:
            print(f"[bold red]Error: {results['error']}[/bold red]")

    if args.portfolio:
        print("[bold cyan]Running portfolio management analysis...[/bold cyan]")
        pm = PortfolioManager(initial_capital=100000)
        
        # Add some sample positions
        try:
            pm.add_position('AAPL', 100, 150.0)
            pm.add_position('MSFT', 50, 300.0)
            pm.add_position('GOOGL', 30, 2500.0)
            
            # Update prices
            pm.update_prices({
                'AAPL': 155.0,
                'MSFT': 310.0,
                'GOOGL': 2600.0
            })
            
            # Get portfolio summary
            summary = pm.get_portfolio_summary()
            print(f"\n[bold green]Portfolio Summary:[/bold green]")
            print(f"Total Value: ${summary['total_value']:,.2f}")
            print(f"Total Return: {summary['total_return']:.2%}")
            print(f"Cash: ${summary['cash']:,.2f}")
            print(f"Positions: {summary['num_positions']}")
            
            # Calculate metrics
            metrics = pm.get_portfolio_metrics()
            print(f"\n[bold yellow]Portfolio Metrics:[/bold yellow]")
            print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
            print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
            print(f"Beta: {metrics.beta:.2f}")
            print(f"Alpha: {metrics.alpha:.2%}")
            
            # Optimize portfolio
            optimization = pm.optimize_portfolio(target_volatility=0.15)
            if optimization:
                print(f"\n[bold magenta]Portfolio Optimization:[/bold magenta]")
                print(f"Expected Return: {optimization['expected_return']:.2%}")
                print(f"Expected Volatility: {optimization['expected_volatility']:.2%}")
                print(f"Sharpe Ratio: {optimization['sharpe_ratio']:.2f}")
                
        except Exception as e:
            print(f"[bold red]Error in portfolio analysis: {e}[/bold red]")

if __name__ == "__main__":
    main() 