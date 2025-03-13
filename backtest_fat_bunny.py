import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
import optuna
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class FatBunnyBacktest:
    def __init__(self, data, initial_capital=1000):
        """
        Initialize backtester with OHLCV data
        data: pandas DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        initial_capital: Starting capital for the backtest (default: 1000)
        """
        self.data = data
        self.initial_capital = initial_capital
        self.results = []
        self.trade_history = []
        self.equity_curve = []
        
    def calculate_channel(self, df, htf_period):
        """Calculate channel high and low values"""
        df['channel_high'] = df['high'].rolling(window=htf_period).max()
        df['channel_low'] = df['low'].rolling(window=htf_period).min()
        return df
    
    def plot_results(self, trades_df):
        """Plot trading results with price, entry/exit points, and equity curve"""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                          vertical_spacing=0.03, 
                          subplot_titles=('Price Action & Trades', 'Equity Curve'),
                          row_heights=[0.7, 0.3])

        # Price candlesticks
        fig.add_trace(
            go.Candlestick(x=self.data.index,
                          open=self.data['open'],
                          high=self.data['high'],
                          low=self.data['low'],
                          close=self.data['close'],
                          name='Price'),
            row=1, col=1
        )

        # Plot long entries and exits
        longs = trades_df[trades_df['type'] == 'long']
        fig.add_trace(
            go.Scatter(x=longs.index, y=longs['entry'],
                      mode='markers',
                      marker=dict(symbol='triangle-up', size=10, color='green'),
                      name='Long Entry'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=longs.index, y=longs['exit'],
                      mode='markers',
                      marker=dict(symbol='triangle-down', size=10, color='red'),
                      name='Long Exit'),
            row=1, col=1
        )

        # Plot short entries and exits
        shorts = trades_df[trades_df['type'] == 'short']
        fig.add_trace(
            go.Scatter(x=shorts.index, y=shorts['entry'],
                      mode='markers',
                      marker=dict(symbol='triangle-down', size=10, color='red'),
                      name='Short Entry'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=shorts.index, y=shorts['exit'],
                      mode='markers',
                      marker=dict(symbol='triangle-up', size=10, color='green'),
                      name='Short Exit'),
            row=1, col=1
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(x=trades_df.index, y=trades_df['balance'],
                      mode='lines',
                      name='Equity Curve'),
            row=2, col=1
        )

        fig.update_layout(
            title='FAT BUNNY Strategy Backtest Results',
            xaxis_rangeslider_visible=False,
            height=800
        )

        return fig

    def backtest_strategy(self, params):
        """
        Backtest the FAT BUNNY strategy with given parameters
        
        params: dict with keys:
        - htf_period: Higher timeframe period for channel calculation
        - risk_reward_ratio: Risk to reward ratio
        - leverage: Trading leverage
        - trade_size: Percentage of balance to risk per trade
        """
        df = self.data.copy()
        
        # Initialize parameters
        htf_period = params['htf_period']
        risk_reward_ratio = params['risk_reward_ratio']
        leverage = params['leverage']
        trade_size = params['trade_size'] / 100
        
        # Calculate channels
        df = self.calculate_channel(df, htf_period)
        
        # Initialize trading variables
        balance = self.initial_capital  # Use configurable starting balance
        position = None
        entry_price = None
        stop_loss = None
        take_profit = None
        trades = []
        self.trade_history = []  # Reset trade history
        entry_time = None
        
        # Iterate through data
        for i in range(htf_period, len(df)):
            current_time = df.index[i]
            current_price = df['close'].iloc[i]
            
            # Check for position exit
            if position:
                if position == 'long':
                    if df['high'].iloc[i] >= take_profit:
                        pnl = (take_profit - entry_price) / entry_price * leverage
                        balance *= (1 + pnl * trade_size)
                        self.trade_history.append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'type': 'long',
                            'entry': entry_price,
                            'exit': take_profit,
                            'pnl': pnl * 100,
                            'balance': balance
                        })
                        position = None
                    elif df['low'].iloc[i] <= stop_loss:
                        pnl = (stop_loss - entry_price) / entry_price * leverage
                        balance *= (1 + pnl * trade_size)
                        self.trade_history.append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'type': 'long',
                            'entry': entry_price,
                            'exit': stop_loss,
                            'pnl': pnl * 100,
                            'balance': balance
                        })
                        position = None
                        
                elif position == 'short':
                    if df['low'].iloc[i] <= take_profit:
                        pnl = (entry_price - take_profit) / entry_price * leverage
                        balance *= (1 + pnl * trade_size)
                        self.trade_history.append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'type': 'short',
                            'entry': entry_price,
                            'exit': take_profit,
                            'pnl': pnl * 100,
                            'balance': balance
                        })
                        position = None
                    elif df['high'].iloc[i] >= stop_loss:
                        pnl = (entry_price - stop_loss) / entry_price * leverage
                        balance *= (1 + pnl * trade_size)
                        self.trade_history.append({
                            'entry_time': entry_time,
                            'exit_time': current_time,
                            'type': 'short',
                            'entry': entry_price,
                            'exit': stop_loss,
                            'pnl': pnl * 100,
                            'balance': balance
                        })
                        position = None
            
            # Check for new position entry
            if not position:
                # Long entry
                if current_price > df['channel_high'].iloc[i-1]:
                    position = 'long'
                    entry_price = current_price
                    entry_time = current_time
                    stop_loss = df['low'].iloc[i]
                    take_profit = entry_price + (entry_price - stop_loss) * risk_reward_ratio
                
                # Short entry
                elif current_price < df['channel_low'].iloc[i-1]:
                    position = 'short'
                    entry_price = current_price
                    entry_time = current_time
                    stop_loss = df['high'].iloc[i]
                    take_profit = entry_price - (stop_loss - entry_price) * risk_reward_ratio
        
        # Calculate metrics
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            trades_df.set_index('entry_time', inplace=True)
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = winning_trades / total_trades
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if len(trades_df[trades_df['pnl'] < 0]) > 0 else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            max_drawdown = self.calculate_max_drawdown(trades_df['balance'])
            sharpe = self.calculate_sharpe_ratio(trades_df['pnl'] / 100)
            
            return {
                'final_balance': balance,
                'total_return': ((balance - self.initial_capital) / self.initial_capital) * 100,
                'total_trades': total_trades,
                'win_rate': win_rate * 100,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe,
                'params': params,
                'trades_df': trades_df
            }
        
        return {
            'final_balance': balance,
            'total_return': 0,
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'params': params,
            'trades_df': pd.DataFrame()
        }
    
    @staticmethod
    def calculate_max_drawdown(equity_curve):
        """Calculate maximum drawdown from equity curve"""
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        return abs(drawdown.min()) * 100
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio from returns"""
        excess_returns = returns - risk_free_rate/252  # Assuming daily data
        if len(excess_returns) < 2:
            return 0
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
    
    def optimize_parameters(self, n_trials=100):
        """
        Optimize strategy parameters using Optuna
        """
        def objective(trial):
            params = {
                'htf_period': trial.suggest_categorical('htf_period', [2, 3, 5, 10]),  # Specific timeframe choices
                'risk_reward_ratio': trial.suggest_float('risk_reward_ratio', 0.5, 3.0),
                'leverage': trial.suggest_int('leverage', 1, 50),
                'trade_size': trial.suggest_float('trade_size', 1.0, 50.0)
            }
            
            result = self.backtest_strategy(params)
            
            # Create a composite score considering multiple metrics
            score = (
                result['total_return'] * 0.4 +  # 40% weight on returns
                result['win_rate'] * 0.2 +      # 20% weight on win rate
                result['sharpe_ratio'] * 20 * 0.2 +  # 20% weight on Sharpe ratio
                (100 - result['max_drawdown']) * 0.2  # 20% weight on minimizing drawdown
            )
            
            return score
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        best_result = self.backtest_strategy(best_params)
        
        return best_result, best_params

def load_and_prepare_data(file_path):
    """
    Load and prepare data from CSV file
    Expected TradingView columns: time, open, high, low, close
    """
    try:
        df = pd.read_csv(file_path)
        
        # Verify required columns exist
        required_columns = ['time', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert timestamp and set as index
        df['timestamp'] = pd.to_datetime(df['time'])
        df = df.drop('time', axis=1)
        df.set_index('timestamp', inplace=True)
        
        # Add volume column with 0s if it doesn't exist
        if 'volume' not in df.columns:
            df['volume'] = 0
        
        # Fix any numeric formatting issues
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop any rows with NaN values
        df = df.dropna()
        
        return df
        
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}. Please ensure your CSV file has the columns: time, open, high, low, close")

if __name__ == "__main__":
    # Example usage
    print("FAT BUNNY Strategy Optimizer")
    print("----------------------------")
    
    # Get input from user
    file_path = input("Enter the path to your CSV file with OHLCV data: ")
    initial_capital = float(input("Enter starting capital (default 1000): ") or 1000)
    n_trials = int(input("Enter number of optimization trials (default 100): ") or 100)
    
    try:
        # Load data
        data = load_and_prepare_data(file_path)
        print(f"\nLoaded data from {file_path}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        
        # Create backtester instance with specified initial capital
        backtester = FatBunnyBacktest(data, initial_capital=initial_capital)
        
        # Run optimization
        print(f"\nRunning optimization with {n_trials} trials...")
        best_result, best_params = backtester.optimize_parameters(n_trials)
        
        # Print results
        print("\nOptimization Results:")
        print("--------------------")
        print(f"Best Parameters:")
        for param, value in best_params.items():
            print(f"- {param}: {value}")
        
        print("\nStrategy Performance:")
        print(f"- Total Return: {best_result['total_return']:.2f}%")
        print(f"- Total Trades: {best_result['total_trades']}")
        print(f"- Win Rate: {best_result['win_rate']:.2f}%")
        print(f"- Profit Factor: {best_result['profit_factor']:.2f}")
        print(f"- Max Drawdown: {best_result['max_drawdown']:.2f}%")
        print(f"- Sharpe Ratio: {best_result['sharpe_ratio']:.2f}")
        
        # Display trade history
        if 'trades_df' in best_result and not best_result['trades_df'].empty:
            print("\nTrade History:")
            print(best_result['trades_df'].to_string())
            
            # Create and show the plot
            fig = backtester.plot_results(best_result['trades_df'])
            fig.show()
        
    except Exception as e:
        print(f"\nError: {str(e)}") 
