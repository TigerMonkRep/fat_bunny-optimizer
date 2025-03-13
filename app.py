import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from backtest_fat_bunny import FatBunnyBacktest, load_and_prepare_data
import io
import base64

st.set_page_config(page_title="FAT BUNNY Strategy Optimizer", layout="wide")

st.title("FAT BUNNY Strategy Optimizer")
st.markdown("""
This tool helps optimize the parameters of the FAT BUNNY trading strategy using machine learning.
Upload your TradingView data and find the best parameters for your trading strategy.
""")

# File uploader
uploaded_file = st.file_uploader("Upload your TradingView CSV file", type="csv")

if uploaded_file is not None:
    try:
        # Load data
        data = load_and_prepare_data(uploaded_file)
        
        # Show data info
        st.subheader("Data Overview")
        st.write(f"Date range: {data.index.min()} to {data.index.max()}")
        st.write(f"Number of candles: {len(data)}")
        
        # Optimization settings
        st.subheader("Optimization Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            n_trials = st.number_input("Number of optimization trials", 
                                     min_value=10, 
                                     max_value=1000, 
                                     value=100,
                                     help="More trials = better results but longer runtime")
        
        # Run optimization
        if st.button("Start Optimization"):
            with st.spinner("Running optimization..."):
                backtester = FatBunnyBacktest(data)
                best_result, best_params = backtester.optimize_parameters(n_trials)
                
                # Display results
                st.subheader("Optimization Results")
                
                # Parameters
                st.write("Best Parameters:")
                for param, value in best_params.items():
                    st.write(f"- {param}: {value}")
                
                # Performance metrics
                st.write("\nStrategy Performance:")
                metrics = {
                    "Total Return": f"{best_result['total_return']:.2f}%",
                    "Total Trades": best_result['total_trades'],
                    "Win Rate": f"{best_result['win_rate']:.2f}%",
                    "Profit Factor": f"{best_result['profit_factor']:.2f}",
                    "Max Drawdown": f"{best_result['max_drawdown']:.2f}%",
                    "Sharpe Ratio": f"{best_result['sharpe_ratio']:.2f}"
                }
                
                # Create metrics display
                col1, col2, col3 = st.columns(3)
                cols = [col1, col2, col3]
                for i, (metric, value) in enumerate(metrics.items()):
                    with cols[i % 3]:
                        st.metric(metric, value)
                
                # Run backtest with best parameters
                final_result = backtester.backtest_strategy(best_params)
                
                if 'trades' in final_result:
                    trades_df = pd.DataFrame(final_result['trades'])
                    
                    # Create equity curve
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=range(len(trades_df)),
                        y=trades_df['balance'],
                        mode='lines',
                        name='Equity Curve'
                    ))
                    fig.update_layout(
                        title="Equity Curve",
                        xaxis_title="Trade Number",
                        yaxis_title="Balance",
                        showlegend=True
                    )
                    st.plotly_chart(fig)
                    
                    # Download results
                    csv = trades_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="backtest_results.csv">Download Detailed Results CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.write("Please make sure your CSV file has the correct format (timestamp, open, high, low, close, volume)") 
