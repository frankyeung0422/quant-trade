import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import yfinance as yf
from scipy.optimize import minimize
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class Position:
    ticker: str
    shares: float
    entry_price: float
    current_price: float
    entry_date: datetime
    stop_loss: float
    take_profit: float
    risk_amount: float

@dataclass
class PortfolioMetrics:
    total_value: float
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    volatility: float
    correlation_matrix: pd.DataFrame
    sector_allocation: Dict
    risk_metrics: Dict

class PortfolioManager:
    """Advanced portfolio management with risk analytics and optimization"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.portfolio_history: List[Dict] = []
        
        # Risk parameters
        self.max_position_size = 0.20  # Max 20% in single position
        self.max_sector_allocation = 0.40  # Max 40% in single sector
        self.target_volatility = 0.15  # 15% annual volatility target
        self.risk_free_rate = 0.02  # 2% risk-free rate
        
        # Position sizing parameters
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.kelly_fraction = 0.5  # Conservative Kelly criterion
        
    def add_position(self, ticker: str, shares: float, price: float, 
                    stop_loss: float = None, take_profit: float = None):
        """Add a new position to the portfolio"""
        if ticker in self.positions:
            raise ValueError(f"Position in {ticker} already exists")
            
        cost = shares * price
        if cost > self.cash:
            raise ValueError(f"Insufficient cash. Need ${cost:.2f}, have ${self.cash:.2f}")
            
        # Calculate stop loss and take profit if not provided
        if stop_loss is None:
            stop_loss = price * 0.95  # 5% stop loss
        if take_profit is None:
            take_profit = price * 1.10  # 10% take profit
            
        position = Position(
            ticker=ticker,
            shares=shares,
            entry_price=price,
            current_price=price,
            entry_date=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_amount=cost * self.risk_per_trade
        )
        
        self.positions[ticker] = position
        self.cash -= cost
        
        # Record trade
        self.trade_history.append({
            'timestamp': datetime.now(),
            'action': 'BUY',
            'ticker': ticker,
            'shares': shares,
            'price': price,
            'cost': cost
        })
        
    def remove_position(self, ticker: str, price: float):
        """Remove a position from the portfolio"""
        if ticker not in self.positions:
            raise ValueError(f"No position in {ticker}")
            
        position = self.positions[ticker]
        proceeds = position.shares * price
        pnl = proceeds - (position.shares * position.entry_price)
        
        self.cash += proceeds
        del self.positions[ticker]
        
        # Record trade
        self.trade_history.append({
            'timestamp': datetime.now(),
            'action': 'SELL',
            'ticker': ticker,
            'shares': position.shares,
            'price': price,
            'proceeds': proceeds,
            'pnl': pnl
        })
        
    def update_prices(self, prices: Dict[str, float]):
        """Update current prices for all positions"""
        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker].current_price = price
                
    def calculate_position_size(self, ticker: str, entry_price: float, 
                              stop_loss: float, confidence: float = 0.5) -> float:
        """Calculate optimal position size using Kelly criterion and risk management"""
        
        # Get historical data for volatility calculation
        try:
            ticker_data = yf.Ticker(ticker)
            hist = ticker_data.history(period="1y")
            
            if hist.empty:
                return 0
                
            # Calculate historical volatility
            returns = hist['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate win rate and average win/loss from historical data
            # This is a simplified approach - in practice, you'd use your strategy's backtest results
            win_rate = 0.55  # Assume 55% win rate
            avg_win = 0.08   # Assume 8% average win
            avg_loss = 0.05  # Assume 5% average loss
            
        except Exception:
            # Use conservative defaults if data unavailable
            volatility = 0.25
            win_rate = 0.5
            avg_win = 0.05
            avg_loss = 0.05
            
        # Kelly criterion calculation
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly_fraction = max(0, min(kelly_fraction, 1))  # Clamp between 0 and 1
        
        # Apply conservative Kelly fraction
        kelly_fraction *= self.kelly_fraction
        
        # Risk-based position sizing
        risk_per_share = entry_price - stop_loss
        max_risk_amount = self.get_total_value() * self.risk_per_trade
        max_shares_by_risk = max_risk_amount / risk_per_share
        
        # Volatility-based position sizing
        target_volatility_contribution = self.target_volatility / len(self.positions + [ticker])
        max_shares_by_volatility = (self.get_total_value() * target_volatility_contribution) / (entry_price * volatility)
        
        # Maximum position size constraint
        max_shares_by_size = (self.get_total_value() * self.max_position_size) / entry_price
        
        # Take the minimum of all constraints
        max_shares = min(max_shares_by_risk, max_shares_by_volatility, max_shares_by_size)
        
        # Apply Kelly fraction and confidence
        optimal_shares = max_shares * kelly_fraction * confidence
        
        return max(0, optimal_shares)
        
    def get_total_value(self) -> float:
        """Calculate total portfolio value"""
        total = self.cash
        for position in self.positions.values():
            total += position.shares * position.current_price
        return total
        
    def get_portfolio_metrics(self, benchmark_ticker: str = 'SPY') -> PortfolioMetrics:
        """Calculate comprehensive portfolio metrics"""
        
        if not self.portfolio_history:
            return self._create_empty_metrics()
            
        # Create portfolio returns series
        portfolio_values = [entry['value'] for entry in self.portfolio_history]
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
        
        # Get benchmark data
        try:
            benchmark_data = yf.Ticker(benchmark_ticker).history(period="1y")
            benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        except:
            benchmark_returns = pd.Series([0] * len(portfolio_returns))
            
        # Calculate metrics
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        excess_returns = portfolio_returns - self.risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        
        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        sortino_ratio = excess_returns.mean() / downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk and Conditional VaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Beta and Alpha
        if len(benchmark_returns) > 0:
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
            
            benchmark_return = benchmark_returns.mean() * 252
            alpha = (portfolio_returns.mean() * 252) - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        else:
            beta = 1
            alpha = 0
            
        # Correlation matrix
        correlation_matrix = self._calculate_correlation_matrix()
        
        # Sector allocation
        sector_allocation = self._calculate_sector_allocation()
        
        # Risk metrics
        risk_metrics = {
            'var_99': np.percentile(portfolio_returns, 1),
            'cvar_99': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 1)].mean(),
            'skewness': stats.skew(portfolio_returns),
            'kurtosis': stats.kurtosis(portfolio_returns),
            'calmar_ratio': total_return / abs(max_drawdown) if max_drawdown != 0 else 0,
            'information_ratio': (portfolio_returns.mean() - benchmark_returns.mean()) / (portfolio_returns.std() - benchmark_returns.std()) if len(benchmark_returns) > 0 else 0
        }
        
        return PortfolioMetrics(
            total_value=portfolio_values[-1],
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            var_95=var_95,
            cvar_95=cvar_95,
            beta=beta,
            alpha=alpha,
            volatility=volatility,
            correlation_matrix=correlation_matrix,
            sector_allocation=sector_allocation,
            risk_metrics=risk_metrics
        )
        
    def _calculate_correlation_matrix(self) -> pd.DataFrame:
        """Calculate correlation matrix for all positions"""
        if len(self.positions) < 2:
            return pd.DataFrame()
            
        # Get historical data for all positions
        tickers = list(self.positions.keys())
        price_data = {}
        
        for ticker in tickers:
            try:
                data = yf.Ticker(ticker).history(period="1y")
                if not data.empty:
                    price_data[ticker] = data['Close']
            except:
                continue
                
        if len(price_data) < 2:
            return pd.DataFrame()
            
        # Create DataFrame and calculate correlations
        df = pd.DataFrame(price_data)
        returns = df.pct_change().dropna()
        
        return returns.corr()
        
    def _calculate_sector_allocation(self) -> Dict:
        """Calculate sector allocation of portfolio"""
        sector_allocation = {}
        
        for ticker in self.positions:
            try:
                ticker_info = yf.Ticker(ticker).info
                sector = ticker_info.get('sector', 'Unknown')
                
                position_value = self.positions[ticker].shares * self.positions[ticker].current_price
                sector_allocation[sector] = sector_allocation.get(sector, 0) + position_value
                
            except:
                sector_allocation['Unknown'] = sector_allocation.get('Unknown', 0) + position_value
                
        # Convert to percentages
        total_value = self.get_total_value()
        if total_value > 0:
            sector_allocation = {k: v / total_value for k, v in sector_allocation.items()}
            
        return sector_allocation
        
    def _create_empty_metrics(self) -> PortfolioMetrics:
        """Create empty metrics when no portfolio history exists"""
        return PortfolioMetrics(
            total_value=self.initial_capital,
            total_return=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            var_95=0,
            cvar_95=0,
            beta=1,
            alpha=0,
            volatility=0,
            correlation_matrix=pd.DataFrame(),
            sector_allocation={},
            risk_metrics={}
        )
        
    def optimize_portfolio(self, target_return: float = None, 
                         target_volatility: float = None) -> Dict:
        """Optimize portfolio weights using Modern Portfolio Theory"""
        
        if len(self.positions) < 2:
            return {}
            
        # Get historical data
        tickers = list(self.positions.keys())
        price_data = {}
        
        for ticker in tickers:
            try:
                data = yf.Ticker(ticker).history(period="1y")
                if not data.empty:
                    price_data[ticker] = data['Close']
            except:
                continue
                
        if len(price_data) < 2:
            return {}
            
        # Calculate returns and covariance matrix
        df = pd.DataFrame(price_data)
        returns = df.pct_change().dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        
        # Portfolio optimization function
        def portfolio_stats(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return portfolio_return, portfolio_volatility
            
        def objective(weights):
            portfolio_return, portfolio_volatility = portfolio_stats(weights)
            
            if target_return is not None and target_volatility is not None:
                return abs(portfolio_return - target_return) + abs(portfolio_volatility - target_volatility)
            elif target_return is not None:
                return abs(portfolio_return - target_return)
            elif target_volatility is not None:
                return abs(portfolio_volatility - target_volatility)
            else:
                # Maximize Sharpe ratio
                return -portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0
                
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
        ]
        
        if target_return is not None:
            constraints.append({'type': 'eq', 'fun': lambda x: portfolio_stats(x)[0] - target_return})
            
        if target_volatility is not None:
            constraints.append({'type': 'eq', 'fun': lambda x: portfolio_stats(x)[1] - target_volatility})
            
        # Bounds (no short selling)
        bounds = tuple((0, 1) for _ in range(len(tickers)))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/len(tickers)] * len(tickers))
        
        # Optimize
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            optimal_return, optimal_volatility = portfolio_stats(optimal_weights)
            
            return {
                'weights': dict(zip(tickers, optimal_weights)),
                'expected_return': optimal_return,
                'expected_volatility': optimal_volatility,
                'sharpe_ratio': optimal_return / optimal_volatility if optimal_volatility > 0 else 0
            }
        else:
            return {}
            
    def rebalance_portfolio(self, target_weights: Dict[str, float]):
        """Rebalance portfolio to target weights"""
        current_value = self.get_total_value()
        
        for ticker, target_weight in target_weights.items():
            target_value = current_value * target_weight
            
            if ticker in self.positions:
                current_value_position = self.positions[ticker].shares * self.positions[ticker].current_price
                value_diff = target_value - current_value_position
                
                if abs(value_diff) > current_value * 0.01:  # Only rebalance if difference > 1%
                    if value_diff > 0:
                        # Need to buy more
                        shares_to_buy = value_diff / self.positions[ticker].current_price
                        if value_diff <= self.cash:
                            self.positions[ticker].shares += shares_to_buy
                            self.cash -= value_diff
                    else:
                        # Need to sell some
                        shares_to_sell = abs(value_diff) / self.positions[ticker].current_price
                        if shares_to_sell <= self.positions[ticker].shares:
                            self.positions[ticker].shares -= shares_to_sell
                            self.cash += abs(value_diff)
            else:
                # New position
                if target_value <= self.cash:
                    shares = target_value / self._get_current_price(ticker)
                    self.add_position(ticker, shares, self._get_current_price(ticker))
                    
    def _get_current_price(self, ticker: str) -> float:
        """Get current price for a ticker"""
        try:
            return yf.Ticker(ticker).info['regularMarketPrice']
        except:
            return 0
            
    def update_portfolio_history(self):
        """Update portfolio history for tracking"""
        current_value = self.get_total_value()
        self.portfolio_history.append({
            'timestamp': datetime.now(),
            'value': current_value,
            'cash': self.cash,
            'positions': len(self.positions)
        })
        
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary"""
        total_value = self.get_total_value()
        total_return = (total_value - self.initial_capital) / self.initial_capital
        
        position_summary = {}
        for ticker, position in self.positions.items():
            current_value = position.shares * position.current_price
            position_return = (position.current_price - position.entry_price) / position.entry_price
            position_summary[ticker] = {
                'shares': position.shares,
                'entry_price': position.entry_price,
                'current_price': position.current_price,
                'current_value': current_value,
                'return': position_return,
                'weight': current_value / total_value if total_value > 0 else 0
            }
            
        return {
            'total_value': total_value,
            'cash': self.cash,
            'total_return': total_return,
            'num_positions': len(self.positions),
            'positions': position_summary,
            'trade_count': len(self.trade_history)
        }

# Example usage
if __name__ == "__main__":
    # Create portfolio manager
    pm = PortfolioManager(initial_capital=100000)
    
    # Add some positions
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
    print("Portfolio Summary:")
    print(summary)
    
    # Calculate metrics
    metrics = pm.get_portfolio_metrics()
    print(f"\nPortfolio Metrics:")
    print(f"Total Return: {metrics.total_return:.2%}")
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    
    # Optimize portfolio
    optimization = pm.optimize_portfolio(target_volatility=0.15)
    if optimization:
        print(f"\nOptimized Weights: {optimization['weights']}")
        print(f"Expected Return: {optimization['expected_return']:.2%}")
        print(f"Expected Volatility: {optimization['expected_volatility']:.2%}") 