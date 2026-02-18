import streamlit as st
import pandas as pd
import time
import sys
import os

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(__file__))

from data.shiller_pe_loader import load_shiller_pe
from risk.shiller_pe_risk_web import compute_risk, plot_charts, plot_correlation_charts
from strategies.strat_test_web import run_strategy, plot_strategy_results
from strategies.analyze_performance import analyze_by_decades
from strategies.analyze_sharpe_ratio import analyze_risk_adjusted_returns

# Set page config with custom logo
logo_path = os.path.join(os.path.dirname(__file__), "logo_transparent.png")
page_icon = logo_path if os.path.exists(logo_path) else "📈"

st.set_page_config(
    page_title="Shiller P/E Dynamic Valuation",
    page_icon=page_icon,
    layout="wide"
)

# Custom CSS for styling the sidebar navigation
st.markdown("""
<style>
    /* Style the container for the radio options */
    [data-testid="stRadio"] > div {
        gap: 10px;
    }

    /* Style each option label */
    [data-testid="stRadio"] div[role="radiogroup"] label {
        background-color: #0E1117; /* Match sidebar background or slightly lighter */
        padding: 10px 15px;
        border-radius: 8px;
        border: 1px solid rgba(250, 250, 250, 0.1);
        cursor: pointer;
        transition: all 0.3s ease;
        width: 100%;
        display: flex;
        justify-content: center; /* Center text */
        align-items: center;
    }

    /* Hover effect */
    [data-testid="stRadio"] div[role="radiogroup"] label:hover {
        background-color: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.3);
    }

    /* Selected state */
    [data-testid="stRadio"] label:has(input:checked) {
        background-color: rgba(255, 75, 75, 0.1); /* Subtle red tint */
        border-color: #FF4B4B;
        box-shadow: 0 0 10px rgba(255, 75, 75, 0.2);
    }
    
    /* Selected text color */
    [data-testid="stRadio"] label:has(input:checked) p {
        color: #FF4B4B;
        font-weight: 600;
    }

    /* Hide the default radio circle - Aggressive selector */
    [data-testid="stRadio"] label > div:first-child {
        display: none !important;
    }
    
    /* Ensure text is centered and looks good */
    [data-testid="stRadio"] p {
        font-size: 16px;
        margin: 0;
        text-align: center;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar controls
logo_path = os.path.join(os.path.dirname(__file__), "logo_transparent.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)

st.sidebar.subheader("Tabs")
app_mode = st.sidebar.radio("Navigation", ["Home", "Market Analysis"], label_visibility="collapsed")

# Load data
@st.cache_data(ttl=3600)
def get_data():
    with st.spinner("Loading data..."):
        data_pe, data_sp500 = load_shiller_pe()
    return data_pe, data_sp500

try:
    data_pe, data_sp500 = get_data()
    st.sidebar.caption(f"Data Last Updated: {data_pe['Date'].iloc[-1]}")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Compute Risk
@st.cache_data
def get_risk_data(data_pe, data_sp500):
    with st.spinner("Computing risk metrics..."):
        risk_data = compute_risk(data_pe, data_sp500)
    return risk_data

risk_data = get_risk_data(data_pe, data_sp500)

if app_mode == "Home":
    st.title("Dynamic Valuation Metric for S&P 500 Based on Shiller P/E")
    st.markdown("""
    This application analyzes the S&P 500 valuation using the Shiller P/E ratio and simulates a dynamic investment strategy.
    
    Use the navigation sidebar to explore the **Market Analysis** or run the **Strategy Simulator**.
    """)

    st.divider()

    # Get latest risk
    latest_risk = risk_data['Risk'].iloc[-1]
    latest_date = risk_data['Date'].iloc[-1].strftime('%Y-%m-%d')

    # Determine label and color
    if latest_risk < 0.1:
        risk_label = "Extreme Undervaluation"
        risk_color = "green"
    elif latest_risk < 0.4:
        risk_label = "Undervaluation"
        risk_color = "lightgreen"
    elif latest_risk < 0.6:
        risk_label = "Fair Value"
        risk_color = "gray"
    elif latest_risk < 0.9:
        risk_label = "Overvaluation"
        risk_color = "orange"
    else:
        risk_label = "Extreme Overvaluation"
        risk_color = "red"

    st.subheader("Current Market Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Risk Level", f"{latest_risk:.0%}", delta=risk_label, delta_color="inverse")
    
    with col2:
        latest_pe = risk_data['PE_Ratio'].iloc[-1]
        st.metric("Shiller P/E Ratio", f"{latest_pe:.2f}")

    with col3:
        latest_price = risk_data['S&P_500'].iloc[-1]
        st.metric("S&P 500 Price", f"{latest_price:,.2f}")

    st.caption(f"As of {latest_date}")

elif app_mode == "Market Analysis":
    risk_figs = plot_charts(risk_data)
    
    # Tabs for full-screen graphs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Risk Analysis", "Risk vs Price", "Logarithmic Corridor", "Rolling Statistics", "Historical Statistics", "Correlation Analysis"])
    
    with tab1:
        st.plotly_chart(risk_figs[2], use_container_width=True, key="risk_analysis")
        st.caption("S&P 500 price color-coded by the calculated Risk metric. Red indicates high risk (overvaluation), blue indicates low risk (undervaluation).")
        
    with tab2:
        st.plotly_chart(risk_figs[4], use_container_width=True, key="risk_price")
        st.caption("Comparison of the Risk Metric (0-1) and the S&P 500 price (log scale).")

    with tab3:
        st.plotly_chart(risk_figs[3], use_container_width=True, key="log_corridor")
        st.caption("Logarithmic view of the Shiller P/E ratio and its rolling bounds.")
        
    with tab4:
        st.plotly_chart(risk_figs[0], use_container_width=True, key="rolling_stats")
        st.caption("Shiller P/E ratio with rolling mean and ±3 standard deviation bands.")
        
    with tab5:
        st.plotly_chart(risk_figs[1], use_container_width=True, key="historical_stats")
        st.caption("Shiller P/E ratio compared to its historical average and ±2 standard deviation bands.")

    with tab6:
        st.header("Forward Return Analysis")
        st.markdown("""
        This section analyzes the relationship between current valuation metrics (Risk and Shiller P/E) and the subsequent **annualized return** of the S&P 500.
        
        Use the controls below to customize the analysis period and return horizon.
        """)
        
        # Controls for Analysis
        col_ctrl1, col_ctrl2 = st.columns(2)
        
        with col_ctrl1:
            start_year_corr = st.slider(
                "Start Analysis From Year", 
                min_value=1900, 
                max_value=2015, 
                value=1950, 
                step=5,
                key="corr_start_year"
            )
            
        with col_ctrl2:
            return_horizon = st.selectbox(
                "Return Horizon (Years)",
                options=[1, 3, 5, 10, 20],
                index=3, # Default to 10
                format_func=lambda x: f"{x} Years",
                key="corr_horizon"
            )
        
        st.markdown(f"### {return_horizon}-Year Annualized Return Correlations (since {start_year_corr})")

        with st.spinner("Computing correlation analysis..."):
            corr_figs = plot_correlation_charts(risk_data, start_year=start_year_corr, return_years=return_horizon)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Monthly Data Points")
            st.plotly_chart(corr_figs[0], use_container_width=True, key="corr_risk_monthly")
        with col2:
            st.markdown("#### Monthly Data Points")
            st.plotly_chart(corr_figs[1], use_container_width=True, key="corr_pe_monthly")
            
        st.divider()
        
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Yearly Averaged Data")
            st.caption("Data is averaged annually (mean Risk/PE/Price for the year) to reduce noise.")
            st.plotly_chart(corr_figs[2], use_container_width=True, key="corr_risk_yearly")
        with col4:
            st.markdown("#### Yearly Averaged Data")
            st.caption("Data is averaged annually (mean Risk/PE/Price for the year) to reduce noise.")
            st.plotly_chart(corr_figs[3], use_container_width=True, key="corr_pe_yearly")

elif app_mode == "Strategy Simulator":
    st.sidebar.markdown("---")
    st.sidebar.header("Strategy Parameters")
    start_year = st.sidebar.number_input("Start Year", min_value=1900, max_value=2024, value=1950, step=1)
    initial_investment = st.sidebar.number_input("Initial Investment ($)", min_value=0, value=0, step=1000)
    monthly_investment = st.sidebar.number_input("Monthly Investment ($)", min_value=0, value=200, step=50)

    st.header("Strategy Performance")

    with st.expander("Strategy Logic & Rationale"):
        st.markdown("""
        **Rationale**
        The Dynamic Dollar Cost Averaging (dDCA) strategy aims to outperform standard DCA by adjusting investment contributions based on market valuation risk.
        
        **Mechanism**
        *   **Risk Metric**: Derived from the Shiller P/E ratio's position relative to its historical and rolling statistical bounds.
        *   **High Risk (>0.9)**: Market is overvalued. Sell a portion (e.g., 10%) of the portfolio to lock in profits and build cash reserves.
        *   **Low Risk**: Market is undervalued. Aggressively invest using monthly contributions and accumulated cash reserves.
        *   **Neutral Risk**: Continue standard or slightly adjusted investment.
        """)

    if st.button("Run Strategy Simulation"):
        with st.spinner("Running strategy..."):
            # Run the main strategy
            results = run_strategy(risk_data, start_year=start_year, initial_investment=initial_investment, monthly_investment=monthly_investment)
            
            # Plot strategy results
            strategy_figs = plot_strategy_results(results)
            
            # Tabs for strategy charts
            strat_tab0, strat_tab1, strat_tab2, strat_tab3, strat_tab4, strat_tab5, strat_tab6, strat_tab7 = st.tabs([
                "Executive Summary", "Portfolio Value", "Portfolio Composition", "Monthly Cashflows", "Cumulative Cashflow", "Percentage Profit", "Drawdown", "Performance Metrics"
            ])
            
            with strat_tab0:
                st.subheader("Executive Summary")
                
                # Calculate metrics
                _, risk_metrics = analyze_risk_adjusted_returns(results)
                
                # Benchmark Metrics
                bench_invested = abs(results['Benchmark_MaxNegativeCashflow'].iloc[-1])
                bench_final = results['Benchmark_Portfolio'].iloc[-1]
                bench_gain_pct = ((bench_final / bench_invested) - 1) * 100 if bench_invested > 0 else 0
                bench_irr = risk_metrics['Benchmark IRR (Annual)'] * 100
                
                # Strategy Metrics
                strat_invested = abs(results['Strategy_MaxNegativeCashflow'].iloc[-1])
                strat_final = results['Strategy_Portfolio'].iloc[-1]
                strat_gain_pct = ((strat_final / strat_invested) - 1) * 100 if strat_invested > 0 else 0
                strat_irr = risk_metrics['Strategy IRR (Annual)'] * 100
                
                # Display Metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📉 DCA (Benchmark)")
                    st.metric("Total Invested", f"${bench_invested:,.0f}")
                    st.metric("Final Portfolio Value", f"${bench_final:,.0f}")
                    st.metric("Total Gain", f"{bench_gain_pct:,.1f}%")
                    st.metric("Annualized IRR", f"{bench_irr:.2f}%")
                    
                with col2:
                    st.markdown("### 🚀 Dynamic DCA (Strategy)")
                    st.metric("Total Invested", f"${strat_invested:,.0f}", delta=f"${strat_invested - bench_invested:,.0f}", delta_color="inverse")
                    st.metric("Final Portfolio Value", f"${strat_final:,.0f}", delta=f"${strat_final - bench_final:,.0f}")
                    st.metric("Total Gain", f"{strat_gain_pct:,.1f}%", delta=f"{strat_gain_pct - bench_gain_pct:.1f}%")
                    st.metric("Annualized IRR", f"{strat_irr:.2f}%", delta=f"{strat_irr - bench_irr:.2f}%")
            
            with strat_tab1:
                st.plotly_chart(strategy_figs[0], use_container_width=True, key="strat_portfolio")

            with strat_tab2:
                st.plotly_chart(strategy_figs[5], use_container_width=True, key="strat_composition")
                
            with strat_tab3:
                st.plotly_chart(strategy_figs[1], use_container_width=True, key="strat_monthly")
                
            with strat_tab4:
                st.plotly_chart(strategy_figs[2], use_container_width=True, key="strat_cumulative")
                
            with strat_tab5:
                st.plotly_chart(strategy_figs[3], use_container_width=True, key="strat_profit")
                
            with strat_tab6:
                st.plotly_chart(strategy_figs[4], use_container_width=True, key="strat_drawdown")

            with strat_tab7:
                # Analyze by decades (Tables)
                st.subheader("Performance by Decades")
                with st.spinner("Analyzing performance across decades..."):
                    returns_df, risk_metrics_df = analyze_by_decades(risk_data, start_year=start_year, 
                                                                    initial_investment=initial_investment, 
                                                                    monthly_investment=monthly_investment)
                
                st.markdown("### Total Returns Comparison")
                st.dataframe(returns_df)
                
                st.markdown("### Risk-Adjusted Metrics")
                st.dataframe(risk_metrics_df)
                
                st.markdown("### Maximum Drawdown")
                st.dataframe(returns_df[['Period', 'Years', 'Benchmark Max Drawdown %', 'Strategy Max Drawdown %']])

