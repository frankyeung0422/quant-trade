# Advanced Stock Quant Trader

A comprehensive quantitative trading system with AI-powered optimization, machine learning strategies, and advanced risk management.

## 🚀 Features

### Core Trading System
- **Technical Analysis**: 15+ technical indicators (RSI, MACD, Bollinger Bands, ADX, CCI, etc.)
- **Multiple Strategies**: Trend following, mean reversion, momentum, and volume-based strategies
- **Risk Management**: Position sizing, stop-loss, take-profit, and transaction costs
- **Backtesting**: Comprehensive performance metrics and trade analysis
- **Real-Time Trading**: Live market data streaming and automated trading execution
- **Portfolio Management**: Advanced portfolio optimization and risk analytics

### 🤖 AI & Machine Learning
- **AI Parameter Optimization**: Bayesian optimization for strategy parameters
- **ML Strategies**: Random Forest, Gradient Boosting, AdaBoost, SVM, Neural Networks
- **Deep Learning**: LSTM, Transformer models, and ensemble architectures
- **Feature Engineering**: 30+ advanced technical and market features
- **Ensemble Methods**: Voting and weighted ensemble strategies
- **Hyperparameter Tuning**: Automated model optimization
- **Real-Time ML**: Live model training and prediction updates

### 📊 Advanced Analysis
- **Walk-Forward Validation**: Robust out-of-sample testing
- **Market Regime Detection**: Bull/bear/sideways/volatile market identification
- **Performance Analytics**: Sharpe ratio, Sortino ratio, Calmar ratio, VaR, CVaR
- **Strategy Comparison**: Multi-strategy backtesting and comparison

### 🔧 Risk Management
- **Dynamic Position Sizing**: Risk-based position sizing with Kelly criterion
- **ATR-based Stop Losses**: Adaptive stop-loss levels
- **Portfolio Optimization**: Multi-asset portfolio management with Modern Portfolio Theory
- **Drawdown Protection**: Maximum drawdown monitoring and protection
- **Real-Time Risk Monitoring**: Live risk metrics and alerts
- **VaR & CVaR Analysis**: Advanced risk measurement techniques

## 📦 Installation

```bash
git clone <repository-url>
cd stock_quant_trader
pip install -r requirements.txt
```

## 🎯 Usage Examples

### Quick Demo
```bash
# Run comprehensive demo showcasing all features
python demo_advanced.py
```

### Basic Backtesting
```bash
# Run basic backtest
python main.py --ticker AAPL --period 1y --backtest

# Compare multiple strategies
python main.py --ticker AAPL --period 1y --compare
```

### AI Optimization
```bash
# Optimize strategy parameters using AI
python main.py --ticker AAPL --period 1y --optimize

# Run advanced analysis with walk-forward validation
python main.py --ticker AAPL --period 1y --advanced
```

### Machine Learning Strategies
```bash
# Use basic ML strategy
python main.py --ticker AAPL --period 1y --ml

# Compare different ML algorithms
python main.py --ticker AAPL --period 1y --ml-compare

# Use ensemble ML strategy
python main.py --ticker AAPL --period 1y --ensemble-ml

# Deep learning analysis
python main.py --ticker AAPL --period 1y --deep-learning --model-type lstm

# Real-time trading
python main.py --ticker AAPL --real-time

# Portfolio management
python main.py --ticker AAPL --portfolio
```

## 📈 Strategy Performance

### Available Strategies
1. **Trend Following**: Moving average crossover
2. **RSI Mean Reversion**: Oversold/overbought signals
3. **MACD Crossover**: MACD signal line crossovers
4. **Bollinger Bands**: Price band mean reversion
5. **Volume Price Action**: Volume-confirmed price movements
6. **Multi-Signal**: Combined strategy confirmation
7. **ML Random Forest**: Machine learning-based signals
8. **Ensemble ML**: Multiple ML model voting
9. **LSTM Deep Learning**: Long Short-Term Memory neural networks
10. **Transformer Models**: Attention-based deep learning models
11. **Adaptive Strategies**: Market regime-aware strategies
12. **Real-Time Trading**: Live market data strategies

### Performance Metrics
- **Total Return**: Overall strategy performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Calmar Ratio**: Return to maximum drawdown ratio
- **Value at Risk (VaR)**: 95% confidence interval loss
- **Conditional VaR**: Expected loss beyond VaR

## 🔬 Advanced Features

### AI Parameter Optimization
- **Bayesian Optimization**: Efficient parameter search
- **Multi-Objective**: Optimize for return, risk, or both
- **Walk-Forward**: Out-of-sample validation
- **Regime Adaptation**: Market condition-specific parameters

### Machine Learning Capabilities
- **Feature Selection**: Automatic selection of best features
- **Hyperparameter Tuning**: Grid search and cross-validation
- **Ensemble Methods**: Combine multiple models
- **Probability Thresholds**: Confidence-based signal filtering
- **Deep Learning Models**: LSTM, Transformer, and ensemble architectures
- **Real-Time Training**: Continuous model updates with new data
- **Transfer Learning**: Pre-trained models for faster adaptation

### Market Regime Detection
- **Volatility Analysis**: Market volatility regimes
- **Trend Strength**: ADX-based trend identification
- **Regime-Specific Parameters**: Adaptive strategy parameters
- **Regime History**: Historical regime tracking

## 📊 Example Output

```
============================================================
BACKTEST RESULTS
============================================================
Total Return: 15.67%
Number of Trades: 23
Win Rate: 65.22%
Average Win: 3.45%
Average Loss: -2.12%
Profit Factor: 2.18
Max Drawdown: 8.34%
Sharpe Ratio: 1.87
============================================================

TRADE DETAILS:
------------------------------------------------------------
Trade 1: Entry: $150.25, Exit: $155.30, P&L: 3.36%, Reason: take_profit
Trade 2: Entry: $148.90, Exit: $145.20, P&L: -2.48%, Reason: stop_loss
...
```

## 🛠️ Configuration

### Strategy Parameters
- **Risk per Trade**: Default 2% of capital
- **Stop Loss**: 5% or 2x ATR
- **Take Profit**: 2:1 risk-reward ratio
- **Transaction Costs**: 0.1% per trade

### ML Model Settings
- **Feature Selection**: Top 20 features
- **Confidence Threshold**: 60% for signal generation
- **Cross-Validation**: 5-fold time series split
- **Hyperparameter Tuning**: Grid search with F1 scoring

## 📚 API Reference

### Main Functions
```python
# Run backtest
from backtest import run_backtest, print_backtest_results
result = run_backtest(data)
print_backtest_results(result)

# Compare strategies
from strategy_comparison import run_strategy_comparison
comparison = run_strategy_comparison(data)

# AI optimization
from ai_optimizer import auto_optimize_all_strategies
results = auto_optimize_all_strategies(data, metric='sharpe_ratio')

# Advanced analysis
from advanced_features import run_advanced_analysis
advanced_results = run_advanced_analysis(data, strategy_class, param_space)

# ML comparison
from enhanced_ml import run_ml_comparison
ml_results = run_ml_comparison(data)
```

## 🔍 Troubleshooting

### Common Issues
1. **Missing Data**: Ensure sufficient historical data (minimum 100 days)
2. **ML Training**: Some algorithms may fail with insufficient data
3. **Optimization Time**: AI optimization can take several minutes
4. **Memory Usage**: Large datasets may require more RAM

### Performance Tips
- Use longer time periods for more robust results
- Run walk-forward validation for realistic performance
- Consider transaction costs in your analysis
- Monitor maximum drawdown for risk management

## 🆕 Recent Enhancements

- **✅ Real-time Trading**: Live market data streaming and automated execution
- **✅ Portfolio Management**: Advanced portfolio optimization with Modern Portfolio Theory
- **✅ Deep Learning**: LSTM, Transformer, and ensemble models
- **✅ Web Interface**: Enhanced Streamlit dashboard with real-time monitoring
- **✅ Risk Analytics**: VaR, CVaR, and advanced risk metrics
- **✅ Multi-Asset Support**: Comprehensive portfolio analysis

## 📈 Future Enhancements

- **Alternative Data**: News sentiment, options flow, and social media analysis
- **Cloud Deployment**: AWS/Azure integration for scalable trading
- **Mobile App**: iOS/Android trading application
- **Advanced AI**: GPT integration for market analysis
- **Crypto Trading**: Cryptocurrency market support
- **Options Trading**: Options strategies and analysis

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Past performance does not guarantee future results. Always do your own research and consider consulting with a financial advisor before making investment decisions. 