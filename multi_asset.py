import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from data_fetcher import fetch_data as fetch_stock_data
import matplotlib.pyplot as plt
from improved_backtest import run_improved_backtest
from ensemble_strategy import run_ensemble_backtest
from walk_forward import run_walk_forward_validation
from visualization import plot_strategy_comparison, save_all_plots

class MultiAssetPortfolio:
    """Multi-asset portfolio for diversification"""
    
    def __init__(self, tickers: List[str], weights: List[float] = None):
        """
        Initialize multi-asset portfolio
        
        Args:
            tickers: List of stock tickers
            weights: Portfolio weights (if None, equal weight)
        """
        self.tickers = tickers
        self.weights = weights or [1.0 / len(tickers)] * len(tickers)
        self.data = {}
        self.results = {}
        
        if len(self.weights) != len(self.tickers):
            raise ValueError("Number of weights must match number of tickers")
    
    def fetch_data(self, period: str = "1y"):
        """Fetch data for all tickers"""
        print(f"Fetching data for {len(self.tickers)} tickers...")
        
        for ticker in self.tickers:
            try:
                print(f"Fetching {ticker}...")
                data = fetch_stock_data(ticker, period)
                if data is not None and not data.empty:
                    self.data[ticker] = data
                else:
                    print(f"Warning: No data for {ticker}")
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
        
        print(f"Successfully fetched data for {len(self.data)} tickers")
    
    def run_individual_backtests(self, strategy_type='ensemble'):
        """Run backtest on each individual asset"""
        print(f"Running {strategy_type} backtests on individual assets...")
        
        for ticker in self.data:
            try:
                print(f"Running backtest for {ticker}...")
                
                if strategy_type == 'ensemble':
                    result = run_ensemble_backtest(self.data[ticker])
                else:
                    result = run_improved_backtest(self.data[ticker], allow_short=True)
                
                self.results[ticker] = result
                print(f"{ticker}: {result.total_return:.2f}% return, {result.num_trades} trades")
                
            except Exception as e:
                print(f"Error running backtest for {ticker}: {e}")
    
    def calculate_portfolio_metrics(self) -> Dict:
        """Calculate portfolio-level metrics"""
        if not self.results:
            return {}
        
        # Calculate weighted portfolio metrics
        weighted_return = 0.0
        weighted_sharpe = 0.0
        total_trades = 0
        portfolio_equity = []
        
        for ticker, result in self.results.items():
            weight = self.weights[self.tickers.index(ticker)]
            weighted_return += result.total_return * weight
            weighted_sharpe += result.sharpe_ratio * weight
            total_trades += result.num_trades
        
        # Calculate portfolio drawdown (simplified)
        max_drawdown = max([r.max_drawdown for r in self.results.values()])
        
        portfolio_metrics = {
            'total_return': weighted_return,
            'sharpe_ratio': weighted_sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'num_assets': len(self.results),
            'avg_return_per_asset': np.mean([r.total_return for r in self.results.values()]),
            'std_return_per_asset': np.std([r.total_return for r in self.results.values()]),
            'best_asset': max(self.results.items(), key=lambda x: x[1].total_return)[0],
            'worst_asset': min(self.results.items(), key=lambda x: x[1].total_return)[0]
        }
        
        return portfolio_metrics
    
    def run_walk_forward_portfolio(self, strategy_type='ensemble'):
        """Run walk-forward validation on portfolio"""
        print("Running walk-forward validation on portfolio...")
        
        portfolio_results = []
        
        for ticker in self.data:
            try:
                print(f"Running walk-forward for {ticker}...")
                result = run_walk_forward_validation(
                    self.data[ticker], 
                    strategy_type=strategy_type,
                    optimize_params=True
                )
                portfolio_results.append({
                    'ticker': ticker,
                    'results': result
                })
            except Exception as e:
                print(f"Error in walk-forward for {ticker}: {e}")
        
        return portfolio_results
    
    def print_portfolio_summary(self):
        """Print portfolio summary"""
        if not self.results:
            print("No results to display")
            return
        
        portfolio_metrics = self.calculate_portfolio_metrics()
        
        print("\n" + "="*80)
        print("MULTI-ASSET PORTFOLIO SUMMARY")
        print("="*80)
        print(f"Number of Assets: {portfolio_metrics['num_assets']}")
        print(f"Portfolio Weights: {dict(zip(self.tickers, self.weights))}")
        print()
        
        print("PORTFOLIO METRICS:")
        print(f"Weighted Total Return: {portfolio_metrics['total_return']:.2f}%")
        print(f"Weighted Sharpe Ratio: {portfolio_metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {portfolio_metrics['max_drawdown']:.2f}%")
        print(f"Total Trades: {portfolio_metrics['total_trades']}")
        print()
        
        print("ASSET-LEVEL STATISTICS:")
        print(f"Average Return per Asset: {portfolio_metrics['avg_return_per_asset']:.2f}%")
        print(f"Standard Deviation: {portfolio_metrics['std_return_per_asset']:.2f}%")
        print(f"Best Asset: {portfolio_metrics['best_asset']}")
        print(f"Worst Asset: {portfolio_metrics['worst_asset']}")
        print()
        
        print("INDIVIDUAL ASSET RESULTS:")
        print("-" * 80)
        for ticker, result in self.results.items():
            weight = self.weights[self.tickers.index(ticker)]
            print(f"{ticker} (Weight: {weight:.2f}): {result.total_return:.2f}% return, "
                  f"{result.num_trades} trades, {result.win_rate:.1f}% win rate")
        print("="*80)
    
    def save_portfolio_plots(self, base_filename="portfolio_results"):
        """Save portfolio analysis plots"""
        if not self.results:
            print("No results to plot")
            return
        
        # Create strategy comparison plot
        fig = plot_strategy_comparison(self.results, "Multi-Asset Portfolio Comparison")
        fig.savefig(f"{base_filename}_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save individual asset plots
        for ticker, result in self.results.items():
            save_all_plots(result, self.data[ticker], f"{base_filename}_{ticker}")
        
        print(f"Portfolio plots saved as {base_filename}_*.png")

def run_multi_asset_analysis(tickers: List[str], period: str = "1y", strategy_type='ensemble'):
    """Convenience function to run multi-asset analysis"""
    portfolio = MultiAssetPortfolio(tickers)
    portfolio.fetch_data(period)
    portfolio.run_individual_backtests(strategy_type)
    portfolio.print_portfolio_summary()
    portfolio.save_portfolio_plots()
    return portfolio

# Example usage
if __name__ == "__main__":
    # Example portfolio
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    portfolio = run_multi_asset_analysis(tickers, period="1y", strategy_type='ensemble') 