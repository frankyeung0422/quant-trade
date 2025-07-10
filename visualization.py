import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import List, Dict
from improved_backtest import ImprovedBacktestResult

def plot_equity_curve(result: ImprovedBacktestResult, df: pd.DataFrame, title="Equity Curve"):
    """Plot equity curve with drawdown and trade markers"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Convert equity curve to dates - ensure same length
    start_idx = 50  # Starting index for backtest
    end_idx = start_idx + len(result.equity_curve)
    if end_idx > len(df):
        end_idx = len(df)
    dates = df.index[start_idx:end_idx]
    
    # Ensure equity curve and dates have same length
    if len(dates) != len(result.equity_curve):
        min_len = min(len(dates), len(result.equity_curve))
        dates = dates[:min_len]
        equity_curve = result.equity_curve[:min_len]
    else:
        equity_curve = result.equity_curve
    
    # Plot equity curve
    ax1.plot(dates, equity_curve, 'b-', linewidth=2, label='Portfolio Value')
    ax1.set_title(title, fontsize=16, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add trade markers
    if result.trades:
        for trade in result.trades:
            # Find trade dates (approximate)
            entry_idx = len(result.equity_curve) - len(df) + 50 + trade.get('hold_days', 0)
            if entry_idx < len(dates):
                entry_date = dates[entry_idx]
                ax1.scatter(entry_date, result.equity_curve[entry_idx], 
                           c='green' if trade['profit_pct'] > 0 else 'red', 
                           s=50, alpha=0.7, marker='o')
    
    # Plot drawdown
    equity_series = pd.Series(equity_curve, index=dates)
    rolling_max = equity_series.expanding().max()
    drawdown = (equity_series - rolling_max) / rolling_max * 100
    
    ax2.fill_between(dates, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
    ax2.plot(dates, drawdown, 'r-', linewidth=1)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Format x-axis
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    return fig

def plot_trade_analysis(result: ImprovedBacktestResult, title="Trade Analysis"):
    """Plot trade analysis including P&L distribution and trade duration"""
    if not result.trades:
        print("No trades to analyze")
        return None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Extract trade data
    profits = [t['profit_pct'] for t in result.trades]
    durations = [t['hold_days'] for t in result.trades]
    reasons = [t['exit_reason'] for t in result.trades]
    
    # P&L distribution
    ax1.hist(profits, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(np.mean(profits), color='red', linestyle='--', label=f'Mean: {np.mean(profits):.2f}%')
    ax1.set_title('P&L Distribution', fontweight='bold')
    ax1.set_xlabel('Profit/Loss (%)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Trade duration
    ax2.hist(durations, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
    ax2.axvline(np.mean(durations), color='red', linestyle='--', label=f'Mean: {np.mean(durations):.1f} days')
    ax2.set_title('Trade Duration', fontweight='bold')
    ax2.set_xlabel('Hold Days')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Exit reasons
    reason_counts = pd.Series(reasons).value_counts()
    ax3.pie(reason_counts.values, labels=reason_counts.index, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Exit Reasons', fontweight='bold')
    
    # Cumulative P&L
    cumulative_pnl = np.cumsum(profits)
    ax4.plot(range(1, len(cumulative_pnl) + 1), cumulative_pnl, 'b-', linewidth=2)
    ax4.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax4.set_title('Cumulative P&L', fontweight='bold')
    ax4.set_xlabel('Trade Number')
    ax4.set_ylabel('Cumulative P&L (%)')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_performance_metrics(result: ImprovedBacktestResult, title="Performance Metrics"):
    """Plot performance metrics dashboard"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Key metrics
    metrics = ['Total Return', 'Win Rate', 'Sharpe Ratio', 'Max Drawdown']
    values = [result.total_return, result.win_rate, result.sharpe_ratio, result.max_drawdown]
    colors = ['green' if v > 0 else 'red' for v in values]
    
    bars = ax1.bar(metrics, values, color=colors, alpha=0.7)
    ax1.set_title('Key Performance Metrics', fontweight='bold')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}', ha='center', va='bottom')
    
    # Risk metrics
    risk_metrics = ['Profit Factor', 'Avg Win', 'Avg Loss', 'Num Trades']
    risk_values = [result.profit_factor, result.avg_win, result.avg_loss, result.num_trades]
    
    ax2.bar(risk_metrics, risk_values, color='orange', alpha=0.7)
    ax2.set_title('Risk Metrics', fontweight='bold')
    ax2.set_ylabel('Value')
    ax2.grid(True, alpha=0.3)
    
    # Monthly returns (if available)
    if result.daily_returns:
        daily_returns = pd.Series(result.daily_returns)
        # Create a simple monthly aggregation without resample
        if len(daily_returns) > 0:
            # Group by month (assuming 21 trading days per month)
            monthly_groups = []
            for i in range(0, len(daily_returns), 21):
                monthly_groups.append(daily_returns[i:i+21].sum() * 100)
            
            if monthly_groups:
                ax3.plot(range(len(monthly_groups)), monthly_groups, 'b-', linewidth=2)
                ax3.axhline(0, color='red', linestyle='--', alpha=0.5)
                ax3.set_title('Monthly Returns', fontweight='bold')
                ax3.set_ylabel('Return (%)')
                ax3.set_xlabel('Month')
                ax3.grid(True, alpha=0.3)
    
    # Win/Loss ratio
    if result.trades:
        wins = sum(1 for t in result.trades if t['profit_pct'] > 0)
        losses = len(result.trades) - wins
        
        ax4.pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%', 
                colors=['green', 'red'], startangle=90)
        ax4.set_title('Win/Loss Ratio', fontweight='bold')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_strategy_comparison(results: Dict[str, ImprovedBacktestResult], title="Strategy Comparison"):
    """Plot comparison of multiple strategies"""
    strategies = list(results.keys())
    
    # Prepare data
    metrics = ['Total Return', 'Sharpe Ratio', 'Win Rate', 'Max Drawdown']
    data = []
    
    for strategy in strategies:
        result = results[strategy]
        data.append([result.total_return, result.sharpe_ratio, result.win_rate, result.max_drawdown])
    
    data = np.array(data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        values = data[:, i]
        bars = axes[i].bar(strategies, values, alpha=0.7)
        axes[i].set_title(f'{metric}', fontweight='bold')
        axes[i].set_ylabel(metric)
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height,
                        f'{value:.2f}', ha='center', va='bottom')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def save_all_plots(result: ImprovedBacktestResult, df: pd.DataFrame, base_filename="backtest_results"):
    """Save all plots to files"""
    # Equity curve
    fig1 = plot_equity_curve(result, df, "Equity Curve Analysis")
    fig1.savefig(f"{base_filename}_equity_curve.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Trade analysis
    fig2 = plot_trade_analysis(result, "Trade Analysis")
    if fig2:
        fig2.savefig(f"{base_filename}_trade_analysis.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    # Performance metrics
    fig3 = plot_performance_metrics(result, "Performance Dashboard")
    fig3.savefig(f"{base_filename}_performance_metrics.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"Plots saved as {base_filename}_*.png") 