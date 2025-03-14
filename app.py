import streamlit as st
import pandas as pd
from backtest_fat_bunny import Backtester, FatBunnyStrategy, load_and_prepare_data

st.set_page_config(layout="wide")
st.title("FAT BUNNY Strategy Optimizer")

# File uploader
uploaded_file = st.file_uploader("Upload your TradingView CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    try:
        data = load_and_prepare_data(uploaded_file)
        st.write("Data Overview")
        st.write(f"Date range: {data.index.min()} to {data.index.max()}")
        st.write(f"Number of candles: {len(data)}")

        # Optimization settings
        st.header("Optimization Settings")
        col1, col2 = st.columns(2)
        with col1:
            initial_capital = st.number_input("Initial Capital", min_value=100, value=1000, step=100)
        with col2:
            n_trials = st.number_input("Number of optimization trials", min_value=10, value=100, step=10)

        if st.button("Start Optimization"):
            # Create strategy and backtester instances
            strategy = FatBunnyStrategy()
            backtester = Backtester(data, strategy, initial_capital=initial_capital)
            
            with st.spinner('Running optimization...'):
                # Run optimization
                best_result, best_params = backtester.optimize_parameters(n_trials)
                
                # Display results
                st.header("Optimization Results")
                
                # Parameters
                st.subheader("Best Parameters:")
                for param, value in best_params.items():
                    st.write(f"- {param}: {value}")
                
                # Performance metrics
                st.subheader("Strategy Performance:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Return", f"{best_result['total_return']:.2f}%")
                    st.metric("Total Trades", best_result['total_trades'])
                with col2:
                    st.metric("Win Rate", f"{best_result['win_rate']:.2f}%")
                    st.metric("Profit Factor", f"{best_result['profit_factor']:.2f}")
                with col3:
                    st.metric("Max Drawdown", f"{best_result['max_drawdown']:.2f}%")
                    st.metric("Sharpe Ratio", f"{best_result['sharpe_ratio']:.2f}")
                
                # Display chart
                if 'trades_df' in best_result and not best_result['trades_df'].empty:
                    st.subheader("Trading Chart")
                    fig = backtester.plot_results(best_result['trades_df'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display trade history
                    st.subheader("Trade History")
                    trades_df = best_result['trades_df'].copy()
                    trades_df = trades_df.reset_index()
                    trades_df['entry_time'] = trades_df['entry_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    trades_df['exit_time'] = trades_df['exit_time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    trades_df['pnl'] = trades_df['pnl'].round(2)
                    trades_df['balance'] = trades_df['balance'].round(2)
                    st.dataframe(trades_df, use_container_width=True)
                    
                    # Download trades as CSV
                    st.download_button(
                        label="Download Trade History as CSV",
                        data=trades_df.to_csv(index=False),
                        file_name="trade_history.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"Error: {str(e)}") 
