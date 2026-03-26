import streamlit as st
import pandas as pd
import numpy as np
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
app_mode = st.sidebar.radio("Navigation", ["Home", "Market Analysis", "Price Targets"], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.subheader("Settings")
use_nominal = st.sidebar.toggle("Show Nominal Prices", value=False, help="Toggle between Real (inflation-adjusted) and Nominal S&P 500 prices in charts")

# Load data
@st.cache_data(ttl=60)
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

# Ensure S&P_500_Nominal column exists (may be missing if CPI load failed or cache is stale)
if 'S&P_500_Nominal' not in risk_data.columns:
    risk_data = risk_data.copy()
    risk_data['S&P_500_Nominal'] = risk_data['S&P_500']

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
    if latest_risk < 0.2:
        risk_label = "Extreme Undervaluation"
        risk_color = "green"
    elif latest_risk < 0.4:
        risk_label = "Undervaluation"
        risk_color = "lightgreen"
    elif latest_risk < 0.7:
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
    risk_figs = plot_charts(risk_data, use_nominal=use_nominal)
    
    # Tabs for full-screen graphs
    tab1, tab2, tab2b, tab3, tab4, tab5, tab6 = st.tabs(["Risk Analysis", "Risk vs Price", "Risk Distribution", "Logarithmic Corridor", "Rolling Statistics", "Historical Statistics", "Correlation Analysis"])
    
    with tab1:
        st.plotly_chart(risk_figs[2], use_container_width=True, key="risk_analysis")
        st.caption("S&P 500 price color-coded by the calculated Risk metric. Red indicates high risk (overvaluation), blue indicates low risk (undervaluation).")
        
    with tab2:
        st.plotly_chart(risk_figs[4], use_container_width=True, key="risk_price")
        st.caption("Comparison of the Risk Metric (0-1) and the S&P 500 price (log scale).")

    with tab2b:
        import plotly.graph_objects as go_dist
        
        start_year_dist = st.slider(
            "Start Analysis From Year", 
            min_value=1900, 
            max_value=2020, 
            value=1950, 
            step=5,
            key="dist_start_year"
        )
        
        # Filter data by start year
        filtered_risk = risk_data[risk_data['Date'].dt.year >= start_year_dist]
        current_risk = risk_data['Risk'].iloc[-1]
        
        # Compute bins
        bins = np.arange(0, 1.1, 0.1)
        bin_labels = [f'{bins[i]:.1f}–{bins[i+1]:.1f}' for i in range(len(bins)-1)]
        risk_bins = pd.cut(filtered_risk['Risk'], bins=bins, labels=bin_labels, include_lowest=True)
        bin_counts = risk_bins.value_counts().reindex(bin_labels).fillna(0)
        total_months = bin_counts.sum()
        bin_pcts = (bin_counts / total_months * 100)
        
        # Percentile within filtered data
        pct_filtered = (filtered_risk['Risk'] <= current_risk).mean() * 100
        
        # Executive summary
        if pct_filtered >= 80:
            emoji, tone = "🔴", "significantly overvalued"
        elif pct_filtered >= 60:
            emoji, tone = "🟠", "above average valuation"
        elif pct_filtered >= 40:
            emoji, tone = "🟡", "near fair value"
        elif pct_filtered >= 20:
            emoji, tone = "🟢", "below average valuation"
        else:
            emoji, tone = "🔵", "significantly undervalued"
        
        st.markdown(f"""
        ### {emoji} Current Valuation Context (since {start_year_dist})
        
        With a risk level of **{current_risk:.2f}**, the market is currently **more expensive than {pct_filtered:.0f}% of all months** since {start_year_dist}.
        
        This places the current market at a level of **{tone}** — only **{100 - pct_filtered:.0f}%** of months since {start_year_dist} have been more expensive than today.
        """)
        
        # Build chart
        jet_colors = [
            '#00007F', '#0000FF', '#007FFF', '#00FFFF', '#7FFF7F',
            '#FFFF00', '#FF7F00', '#FF0000', '#7F0000', '#7F0000'
        ]
        current_bin_idx = min(int(current_risk * 10), 9)
        
        border_widths = [1] * 10
        border_colors = ['rgba(255,255,255,0.2)'] * 10
        border_widths[current_bin_idx] = 4
        border_colors[current_bin_idx] = 'white'
        
        bar_texts = []
        for i, (c, p) in enumerate(zip(bin_counts.values, bin_pcts.values)):
            label = f'{int(c)} mo. ({p:.1f}%)'
            if i == current_bin_idx:
                label = f'▼ CURRENT ▼<br>{int(c)} mo. ({p:.1f}%)'
            bar_texts.append(label)
        
        fig_dist = go_dist.Figure()
        fig_dist.add_trace(go_dist.Bar(
            x=bin_labels,
            y=bin_counts.values,
            marker_color=jet_colors,
            marker_line_width=border_widths,
            marker_line_color=border_colors,
            text=bar_texts,
            textposition='outside',
            textfont=dict(size=13),
            hovertemplate='Risk: %{x}<br>Months: %{y}<br><extra></extra>'
        ))
        
        fig_dist.update_layout(
            title=f'Risk Level Distribution — Months Spent at Each Level (since {start_year_dist})',
            xaxis_title='Risk Level',
            yaxis_title='Number of Months',
            height=700,
            font=dict(size=16),
            xaxis=dict(tickfont=dict(size=16), title_font=dict(size=20)),
            yaxis=dict(tickfont=dict(size=16), title_font=dict(size=20)),
            title_font=dict(size=28),
            bargap=0.15
        )
        
        st.plotly_chart(fig_dist, use_container_width=True, key="risk_distribution")
        st.caption("Distribution of months spent at each risk level. Adjust the slider to analyze different historical periods.")

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

elif app_mode == "Price Targets":
    st.title("S&P 500 Price Targets & Drawdown Analysis")
    st.markdown("""
    This analysis calculates the S&P 500 price levels required to reach specific risk thresholds. 
    These targets are based on the current 10-year average earnings (E10) and the statistical 
    boundaries of the Shiller P/E ratio.
    """)

    # Calculate components
    latest_data = risk_data.iloc[-1]
    latest_price = latest_data['S&P_500']
    latest_pe = latest_data['PE_Ratio']
    latest_upper = latest_data['Upper_Bound']
    latest_lower = latest_data['Lower_Bound']
    latest_earnings = latest_price / latest_pe
    
    # Re-calculate Risk_raw bounds from the entire dataset to match compute_risk logic
    risk_raw = (risk_data['PE_Ratio'] - risk_data['Lower_Bound']) / (risk_data['Upper_Bound'] - risk_data['Lower_Bound'])
    risk_raw_min = risk_raw.min()
    risk_raw_max = risk_raw.max()
    
    st.divider()
    
    # Current Status Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current S&P 500", f"{latest_price:,.2f}")
    with col2:
        st.metric("Current Risk", f"{latest_data['Risk']:.1%}")
    with col3:
        st.metric("Current Shiller P/E", f"{latest_pe:.2f}")
    with col4:
        st.metric("Implied E10 Earnings", f"${latest_earnings:,.2f}")

    st.markdown("### Price Targets by Risk Level")
    
    targets = []
    # Use a range of risk levels including extreme values
    risk_levels = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    
    for r in risk_levels:
        # Invert the normalization: Risk = (R_raw - min) / (max - min)
        r_raw = r * (risk_raw_max - risk_raw_min) + risk_raw_min
        # Invert the raw risk: R_raw = (PE - Lower) / (Upper - Lower)
        target_pe = r_raw * (latest_upper - latest_lower) + latest_lower
        # Calculate price based on target PE and current earnings
        target_price = target_pe * latest_earnings
        # Calculate drawdown/gain from current price
        change_pct = (target_price / latest_price) - 1
        
        # Determine status label
        if r >= 0.9: status = "🔴 Extreme Overvaluation"
        elif r >= 0.7: status = "🟠 Overvaluation"
        elif r >= 0.4: status = "🟡 Fair Value"
        elif r >= 0.2: status = "🟢 Undervaluation"
        else: status = "🔵 Extreme Undervaluation"
        
        targets.append({
            "Risk Threshold": f"{r:.0%}",
            "Status": status,
            "Target Price": f"${target_price:,.2f}",
            "Target Shiller P/E": f"{target_pe:.2f}",
            "Required Change": f"{change_pct:+.2%}"
        })
    
    df_targets = pd.DataFrame(targets)
    
    # Style the dataframe
    def style_change(val):
        color = 'red' if '-' in val else 'green'
        if val == '+0.00%': color = 'white'
        return f'color: {color}'

    # Display as a nicely formatted table
    st.table(df_targets)
    
    st.info("""
    **Calculation Methodology:**
    1. **E10 Earnings**: Derived as `Current S&P 500 / Current Shiller P/E`.
    2. **Target P/E**: Derived by inverting the Risk metric formula using the latest rolling statistical boundaries (±3 Standard Deviations).
    3. **Target Price**: `Target P/E * E10 Earnings`.
    4. **Required Change**: Percentage difference between the Target Price and the Current Price.
    """)

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

