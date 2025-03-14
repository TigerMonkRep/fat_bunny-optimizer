import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
import optuna
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from abc import ABC, abstractmethod
warnings.filterwarnings('ignore')

class Strategy(ABC):
    """Base Strategy class that all indicators must inherit from"""
    
    @abstractmethod
    def calculate_indicators(self, df, params):
        """Calculate the indicator values"""
        pass
    
    @abstractmethod
    def check_entry_signals(self, df, i, params):
        """Check for entry signals"""
        pass
    
    @abstractmethod
    def get_optimization_params(self, trial):
        """Define the parameters to optimize"""
        pass

class FatBunnyStrategy(Strategy):
    def __init__(self):
        self.safety_delay = 0
        self.reverse_cooldown = 0
        self.last_trade_type = None
        self.channel_formed = False
        self.trade_taken_in_channel = False
        
    def calculate_indicators(self, df, params):
        """Calculate channel high and low values using higher timeframe logic"""
        htf_period = params['htf_period']
        # Resample to higher timeframe
        htf_data = df.resample(f'{htf_period}T').agg({
            'high': 'max',
            'low': 'min'
        })
        
        # Forward fill the values to match original timeframe
        df['channel_high'] = htf_data['high'].reindex(df.index).ffill()
        df['channel_low'] = htf_data['low'].reindex(df.index).ffill()
        
        # Shift by 1 period to match TradingView's behavior
        df['channel_high'] = df['channel_high'].shift(1)
        df['channel_low'] = df['channel_low'].shift(1)
        
        return df
    
    def check_entry_signals(self, df, i, params):
        """Check for entry signals based on channel breakouts"""
        if df['close'].iloc[i] > df['channel_high'].iloc[i] and not self.trade_taken_in_channel:
            self.trade_taken_in_channel = True
            return 'long', df['low'].iloc[i]
        elif df['close'].iloc[i] < df['channel_low'].iloc[i] and not self.trade_taken_in_channel:
            self.trade_taken_in_channel = True
            return 'short', df['high'].iloc[i]
        return None, None
    
    def get_optimization_params(self, trial):
        """Define Fat Bunny optimization parameters"""
        return {
            'htf_period': trial.suggest_categorical('htf_period', [2, 3, 5, 10]),
            'risk_reward_ratio': trial.suggest_float('risk_reward_ratio', 0.5, 3.0),
            'leverage': 1,
            'trade_size': trial.suggest_float('trade_size', 1.0, 20.0)
        }

class LittlePonyStrategy(Strategy):
    def __init__(self):
        self.last_pivot_type = None
        self.bull_gap_active = False
        self.bear_gap_active = False
        self.bull_gap_lower = None
        self.bull_gap_upper = None
        self.bear_gap_lower = None
        self.bear_gap_upper = None
        self.valid_bull_retracement = False
        self.valid_bear_retracement = False
        self.last_pivot_high = None
        self.last_pivot_low = None
        self.last_pivot_high_idx = None
        self.last_pivot_low_idx = None
        
    def calculate_zigzag(self, df, lb, rb):
        """Calculate ZigZag pivots exactly as in Pine Script"""
        df['pivot_high'] = None
        df['pivot_low'] = None
        df['hh'] = False
        df['hl'] = False
        df['lh'] = False
        df['ll'] = False
        
        window = lb + rb + 1
        
        for i in range(window, len(df) - window):
            # Check left side
            left_higher = all(df['high'].iloc[i] > df['high'].iloc[i-lb:i])
            left_lower = all(df['low'].iloc[i] < df['low'].iloc[i-lb:i])
            
            # Check right side
            right_higher = all(df['high'].iloc[i] > df['high'].iloc[i+1:i+rb+1])
            right_lower = all(df['low'].iloc[i] < df['low'].iloc[i+1:i+rb+1])
            
            # Identify pivot points
            if left_higher and right_higher:
                df.at[df.index[i], 'pivot_high'] = df['high'].iloc[i]
                
                # Classify as HH or LH
                if self.last_pivot_high is not None:
                    if df['high'].iloc[i] > self.last_pivot_high:
                        df.at[df.index[i], 'hh'] = True
                    else:
                        df.at[df.index[i], 'lh'] = True
                self.last_pivot_high = df['high'].iloc[i]
                self.last_pivot_high_idx = i
                
            if left_lower and right_lower:
                df.at[df.index[i], 'pivot_low'] = df['low'].iloc[i]
                
                # Classify as LL or HL
                if self.last_pivot_low is not None:
                    if df['low'].iloc[i] < self.last_pivot_low:
                        df.at[df.index[i], 'll'] = True
                    else:
                        df.at[df.index[i], 'hl'] = True
                self.last_pivot_low = df['low'].iloc[i]
                self.last_pivot_low_idx = i
        
        return df
    
    def detect_fvg(self, df, threshold_percent):
        """Detect Fair Value Gaps exactly as in Pine Script"""
        df['bull_fvg'] = False
        df['bear_fvg'] = False
        df['bull_fvg_lower'] = None
        df['bull_fvg_upper'] = None
        df['bear_fvg_lower'] = None
        df['bear_fvg_upper'] = None
        
        for i in range(2, len(df)):
            # Bullish FVG
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                gap_size = (df['low'].iloc[i] - df['high'].iloc[i-2]) / df['high'].iloc[i-2] * 100
                if gap_size >= threshold_percent:
                    df.at[df.index[i], 'bull_fvg'] = True
                    df.at[df.index[i], 'bull_fvg_lower'] = df['high'].iloc[i-2]
                    df.at[df.index[i], 'bull_fvg_upper'] = df['low'].iloc[i]
                    self.bull_gap_active = True
                    self.bull_gap_lower = df['high'].iloc[i-2]
                    self.bull_gap_upper = df['low'].iloc[i]
            
            # Bearish FVG
            if df['high'].iloc[i] < df['low'].iloc[i-2]:
                gap_size = (df['low'].iloc[i-2] - df['high'].iloc[i]) / df['high'].iloc[i] * 100
                if gap_size >= threshold_percent:
                    df.at[df.index[i], 'bear_fvg'] = True
                    df.at[df.index[i], 'bear_fvg_lower'] = df['high'].iloc[i]
                    df.at[df.index[i], 'bear_fvg_upper'] = df['low'].iloc[i-2]
                    self.bear_gap_active = True
                    self.bear_gap_lower = df['high'].iloc[i]
                    self.bear_gap_upper = df['low'].iloc[i-2]
            
            # Reset FVG states if gaps are filled
            if self.bull_gap_active:
                if df['low'].iloc[i] <= self.bull_gap_lower:
                    self.bull_gap_active = False
                    self.valid_bull_retracement = False
            
            if self.bear_gap_active:
                if df['high'].iloc[i] >= self.bear_gap_upper:
                    self.bear_gap_active = False
                    self.valid_bear_retracement = False
        
        return df
    
    def calculate_indicators(self, df, params):
        """Calculate all necessary indicators"""
        # Calculate ZigZag patterns
        df = self.calculate_zigzag(df, params['left_bars'], params['right_bars'])
        
        # Detect FVGs
        df = self.detect_fvg(df, params['fvg_threshold'])
        
        return df
    
    def check_entry_signals(self, df, i, params):
        """Check for entry signals exactly as in Pine Script"""
        if i < 2:  # Need at least 3 bars of data
            return None, None, None
            
        # Check for FVG retracements
        if self.bull_gap_active:
            # Check if price retraces into the gap
            if df['low'].iloc[i] <= self.bull_gap_upper and df['high'].iloc[i] >= self.bull_gap_lower:
                # Validate retracement
                if df['close'].iloc[i] > df['open'].iloc[i]:  # Bullish candle
                    entry_price = df['close'].iloc[i]
                    tp_price, sl_price = self.calculate_tp_sl(df, i, 'long', entry_price, params)
                    if tp_price is not None and sl_price is not None:
                        return 'long', sl_price, tp_price
            
        if self.bear_gap_active:
            # Check if price retraces into the gap
            if df['high'].iloc[i] >= self.bear_gap_lower and df['low'].iloc[i] <= self.bear_gap_upper:
                # Validate retracement
                if df['close'].iloc[i] < df['open'].iloc[i]:  # Bearish candle
                    entry_price = df['close'].iloc[i]
                    tp_price, sl_price = self.calculate_tp_sl(df, i, 'short', entry_price, params)
                    if tp_price is not None and sl_price is not None:
                        return 'short', sl_price, tp_price
        
        return None, None, None
    
    def get_optimization_params(self, trial):
        """Define Little Pony optimization parameters"""
        # Core strategy parameters
        params = {
            'left_bars': trial.suggest_int('left_bars', 3, 10),
            'right_bars': trial.suggest_int('right_bars', 3, 10),
            'fvg_threshold': trial.suggest_float('fvg_threshold', 0.0, 1.0),
            'leverage': 1,
            'trade_size': trial.suggest_float('trade_size', 1.0, 20.0),
            
            # Toggle different TP/SL methods
            'use_rr_tp': trial.suggest_categorical('use_rr_tp', [True, False]),
            'use_tp_adjustment': trial.suggest_categorical('use_tp_adjustment', [True, False]),
            'use_pivot_sl': trial.suggest_categorical('use_pivot_sl', [True, False]),
            'use_candle_pivot_sl': trial.suggest_categorical('use_candle_pivot_sl', [True, False]),
            
            # TP parameters
            'min_tp_distance': trial.suggest_float('min_tp_distance', 0.2, 2.0),
            'tp_adjust_percent': trial.suggest_float('tp_adjust_percent', -1.0, 1.0),
            
            # SL parameters
            'min_sl_distance': trial.suggest_float('min_sl_distance', 0.2, 2.0),
            'sl_adjust_percent': trial.suggest_float('sl_adjust_percent', -1.0, 1.0),
            
            # Risk-Reward parameters
            'risk_reward_ratio': trial.suggest_float('risk_reward_ratio', 0.5, 3.0),
            
            # Pivot point parameters
            'pivot_lookback': trial.suggest_int('pivot_lookback', 5, 20),
            'candle_pivot_lookback': trial.suggest_int('candle_pivot_lookback', 5, 20)
        }
        
        # Ensure at least one SL method is active
        if not params['use_pivot_sl'] and not params['use_candle_pivot_sl']:
            params['use_pivot_sl'] = True  # Default to pivot SL if neither is selected
            
        return params

    def find_pivot_sl(self, df, i, position_type, params):
        """Find stop loss level using pivot points"""
        if position_type == 'long':
            # Look for recent LL/HL patterns for long SL
            for j in range(max(0, i-params['pivot_lookback']), i):
                if df['ll'].iloc[j] or df['hl'].iloc[j]:
                    return df['low'].iloc[j] * (1 - params['sl_adjust_percent']/100)
        else:  # short
            # Look for recent HH/LH patterns for short SL
            for j in range(max(0, i-params['pivot_lookback']), i):
                if df['hh'].iloc[j] or df['lh'].iloc[j]:
                    return df['high'].iloc[j] * (1 + params['sl_adjust_percent']/100)
        return None

    def find_candle_pivot_sl(self, df, i, position_type, params):
        """Find stop loss level using candle pivot points"""
        lookback = params['candle_pivot_lookback']
        if position_type == 'long':
            lowest_low = df['low'].iloc[max(0, i-lookback):i].min()
            return lowest_low * (1 - params['sl_adjust_percent']/100)
        else:  # short
            highest_high = df['high'].iloc[max(0, i-lookback):i].max()
            return highest_high * (1 + params['sl_adjust_percent']/100)
    
    def calculate_tp_sl(self, df, i, position_type, entry_price, params):
        """Calculate take profit and stop loss levels exactly as in Pine Script"""
        tp_price = None
        sl_price = None
        
        # Calculate base TP using risk-reward if enabled
        if params['use_rr_tp']:
            if position_type == 'long':
                sl_distance = entry_price - params['min_sl_distance']
                tp_distance = entry_price * params['risk_reward_ratio'] * (entry_price - sl_distance) / entry_price
                tp_price = entry_price + tp_distance
            else:  # short
                sl_distance = entry_price + params['min_sl_distance']
                tp_distance = entry_price * params['risk_reward_ratio'] * (sl_distance - entry_price) / entry_price
                tp_price = entry_price - tp_distance
        
        # Adjust TP by percentage if enabled
        if params['use_tp_adjustment'] and tp_price is not None:
            tp_price *= (1 + params['tp_adjust_percent']/100)
        
        # Calculate SL based on selected method
        if params['use_pivot_sl']:
            pivot_sl = self.find_pivot_sl(df, i, position_type, params)
            if pivot_sl is not None:
                sl_price = pivot_sl
        
        if params['use_candle_pivot_sl'] and sl_price is None:
            candle_pivot_sl = self.find_candle_pivot_sl(df, i, position_type, params)
            if candle_pivot_sl is not None:
                sl_price = candle_pivot_sl
        
        # Validate minimum distances
        if position_type == 'long':
            min_tp_price = entry_price * (1 + params['min_tp_distance']/100)
            max_sl_price = entry_price * (1 - params['min_sl_distance']/100)
            
            if tp_price is None or tp_price < min_tp_price:
                tp_price = min_tp_price
            if sl_price is None or sl_price > max_sl_price:
                sl_price = max_sl_price
                
        else:  # short
            min_tp_price = entry_price * (1 - params['min_tp_distance']/100)
            max_sl_price = entry_price * (1 + params['min_sl_distance']/100)
            
            if tp_price is None or tp_price > min_tp_price:
                tp_price = min_tp_price
            if sl_price is None or sl_price < max_sl_price:
                sl_price = max_sl_price
        
        # Apply final adjustments
        if params['use_tp_adjustment']:
            tp_price *= (1 + params['tp_adjust_percent']/100)
        if sl_price is not None:
            sl_price *= (1 + params['sl_adjust_percent']/100)
        
        return tp_price, sl_price

class Backtester:
    def __init__(self, data, strategy, initial_capital=1000):
        """
        Initialize backtester with OHLCV data and strategy
        data: pandas DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        strategy: Instance of a Strategy class
        initial_capital: Starting capital for the backtest (default: 1000)
        """
        self.data = data
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.results = []
        self.trade_history = []
        self.equity_curve = []
        
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
            title='Strategy Backtest Results',
            xaxis_rangeslider_visible=False,
            height=800
        )

        return fig

    def backtest_strategy(self, params):
        """
        Backtest the strategy with given parameters
        """
        df = self.data.copy()
        
        # Calculate indicators based on strategy
        df = self.strategy.calculate_indicators(df, params)
        
        # Initialize trading variables
        balance = self.initial_capital
        position = None
        entry_price = None
        stop_loss = None
        take_profit = None
        self.trade_history = []
        entry_time = None
        
        risk_reward_ratio = params['risk_reward_ratio']
        leverage = params['leverage']
        trade_size = params['trade_size'] / 100
        
        # Iterate through data
        for i in range(1, len(df)):
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
                position_type, sl_price, tp_price = self.strategy.check_entry_signals(df, i, params)
                if position_type:
                    position = position_type
                    entry_price = df['close'].iloc[i]
                    entry_time = current_time
                    stop_loss = sl_price
                    take_profit = tp_price
        
        # Calculate metrics
        if self.trade_history:
            trades_df = pd.DataFrame(self.trade_history)
            trades_df.set_index('entry_time', inplace=True)
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
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
            params = self.strategy.get_optimization_params(trial)
            
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

# Dictionary of available strategies
AVAILABLE_STRATEGIES = {
    'fat_bunny': FatBunnyStrategy,
    'little_pony': LittlePonyStrategy
}

if __name__ == "__main__":
    # Example usage
    print("Strategy Optimizer")
    print("----------------")
    
    # Print available strategies
    print("\nAvailable Strategies:")
    for key in AVAILABLE_STRATEGIES.keys():
        print(f"- {key}")
    
    # Get input from user
    strategy_name = input("\nEnter strategy name: ").lower()
    if strategy_name not in AVAILABLE_STRATEGIES:
        print(f"Error: Strategy '{strategy_name}' not found")
        exit(1)
        
    file_path = input("Enter the path to your CSV file with OHLCV data: ")
    initial_capital = float(input("Enter starting capital (default 1000): ") or 1000)
    n_trials = int(input("Enter number of optimization trials (default 100): ") or 100)
    
    try:
        # Load data
        data = load_and_prepare_data(file_path)
        print(f"\nLoaded data from {file_path}")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        
        # Create strategy instance
        strategy = AVAILABLE_STRATEGIES[strategy_name]()
        
        # Create backtester instance
        backtester = Backtester(data, strategy, initial_capital=initial_capital)
        
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
