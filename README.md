# FAT BUNNY Strategy Optimizer

This tool helps optimize the parameters of the FAT BUNNY trading strategy using machine learning. It uses Optuna for hyperparameter optimization and provides comprehensive backtesting results.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Export data from TradingView:
   - Go to your chart in TradingView
   - Click on "Export Data" (under Chart menu)
   - Select your desired date range
   - Make sure to include OHLCV (Open, High, Low, Close, Volume) data
   - Save the CSV file

3. Run the optimizer:
```bash
python backtest_fat_bunny.py
```

## Input Data Format

The CSV file should have the following columns:
- timestamp: Date and time of the candle
- open: Opening price
- high: Highest price
- low: Lowest price
- close: Closing price
- volume: Trading volume

## Parameters Optimized

The tool optimizes the following parameters:
- htf_period: Higher timeframe period for channel calculation (4-48)
- risk_reward_ratio: Risk to reward ratio (0.5-3.0)
- leverage: Trading leverage (1-50)
- trade_size: Percentage of balance to risk per trade (1.0-50.0)

## Optimization Metrics

The optimization process considers multiple metrics with the following weights:
- Total Return: 40%
- Win Rate: 20%
- Sharpe Ratio: 20%
- Maximum Drawdown: 20%

## Output

The tool will provide:
- Optimal parameter values
- Strategy performance metrics including:
  - Total Return
  - Number of Trades
  - Win Rate
  - Profit Factor
  - Maximum Drawdown
  - Sharpe Ratio

## Example Usage

```bash
python backtest_fat_bunny.py
Enter the path to your CSV file with OHLCV data: data/BTCUSDT_1h.csv
Enter number of optimization trials (default 100): 200
``` 
