import pandas as pd
import numpy as np
import plotly.graph_objects as go
import numpy_financial as npf

def run_strategy(data, start_year=1960, initial_investment=0, monthly_investment=200):
    # Filter data to start from the specified year
    data = data[data['Date'].dt.year >= start_year].copy()
    data.reset_index(drop=True, inplace=True)

    # Add columns to track portfolio value, cash flow, and profit/loss for each strategy
    columns = [
        'Benchmark_Portfolio', 'Strategy_Portfolio',
        'Benchmark_Cashflow', 'Strategy_Cashflow',
        'Benchmark_ProfitLoss', 'Strategy_ProfitLoss',
        'Benchmark_Invested', 'Strategy_Invested',
        'Benchmark_MaxNegativeCashflow', 'Strategy_MaxNegativeCashflow',
        'Benchmark_Profit_New', 'Strategy_Profit_New'
    ]
    for col in columns:
        data[col] = 0.0

    # Initialize variables
    benchmark_shares = 0
    strategy_shares = 0
    peak_shares = 0  # Track maximum shares to use for selling
    benchmark_cashflow = 0
    strategy_cashflow = 0
    benchmark_invested = 0
    strategy_invested = 0
    strategy_new_money_invested = 0  # Track only new money invested, excluding reinvestments
    cash_indicator = 0
    strategy_cashflow_history = []
    max_historical_cash = 0

    # Backtesting loop
    for i in range(len(data)):
        price = data.loc[i, 'S&P_500']
        current_date = data.loc[i, 'Date']
        is_month_start = current_date.is_month_start

        # Benchmark Strategy: Invest only at the beginning of each month
        if is_month_start:
            benchmark_shares += monthly_investment / price
            benchmark_cashflow -= monthly_investment  # Investment is negative cashflow
            benchmark_invested += monthly_investment
        
        # Update portfolio value daily
        data.loc[i, 'Benchmark_Portfolio'] = benchmark_shares * price
        data.loc[i, 'Benchmark_Cashflow'] = benchmark_cashflow
        data.loc[i, 'Benchmark_Invested'] = benchmark_invested
        data.loc[i, 'Benchmark_ProfitLoss'] = benchmark_shares * price + benchmark_cashflow

        # Strategy
        risk = data.loc[i, 'Risk']

        if risk > 0.9:  # Sell 10% of peak shares - can happen any day
            sell_shares = min(peak_shares * 0.1, strategy_shares)  # Only sell what is available
            sell_amount = sell_shares * price
            strategy_shares -= sell_shares
            strategy_cashflow += sell_amount  # Sale is positive cashflow
            cash_indicator += 1
            max_historical_cash = max(max_historical_cash, strategy_cashflow)

        elif 0.8 <= risk <= 0.9:
            pass        
        else:  # Buy - only at the beginning of the month
            if is_month_start:
                # For reinvestment, we want the full amount from sales, not just profits
                sales_proceeds = max(0, strategy_cashflow + strategy_new_money_invested)  # Add back invested amount to get total sales cash
                
                if strategy_cashflow <= -strategy_new_money_invested:  # Keep investing if below required capital
                    # Continue investing fixed amounts if below required capital
                    buy_amount = np.select(
                        [
                            (0.7 <= risk < 0.8), (0.6 <= risk < 0.7), (0.5 <= risk < 0.6),
                            (0.4 <= risk < 0.5), (0.3 <= risk < 0.4), (0.2 <= risk < 0.3),
                            (0.1 <= risk < 0.2), (risk < 0.1)
                        ],
                        [50, 100, 150, 200, 250, 400, 500, 600],
                        default=0
                    )
                    strategy_new_money_invested += buy_amount  # Track new money invested
                else:
                    # Calculate percentage-based reinvestment amount
                    cash_available = sales_proceeds
                    buy_percentage = np.select(
                        [
                            (0.7 <= risk < 0.8), (0.6 <= risk < 0.7), (0.5 <= risk < 0.6),
                            (0.4 <= risk < 0.5), (0.3 <= risk < 0.4), (0.2 <= risk < 0.3),
                            (0.1 <= risk < 0.2), (risk < 0.1)
                        ],
                        [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20],
                        default=0
                    )
                    potential_buy = cash_available * buy_percentage  # Use direct percentage of available cash
                    
                    # If the calculated buy amount is too small and we have cash, invest it all
                    if potential_buy < 500 and cash_available > 0 and risk < 0.5:
                        # First use all available cash
                        buy_amount = cash_available
                        
                        # Then calculate fixed amount for additional investment
                        additional_buy = np.select(
                            [
                                (0.7 <= risk < 0.8), (0.6 <= risk < 0.7), (0.5 <= risk < 0.6),
                                (0.4 <= risk < 0.5), (0.3 <= risk < 0.4), (0.2 <= risk < 0.3),
                                (0.1 <= risk < 0.2), (risk < 0.1)
                            ],
                            [50, 100, 150, 200, 250, 400, 500, 600],
                            default=0
                        )
                        if additional_buy > 0:
                            strategy_new_money_invested += additional_buy  # Track only the fixed amount as new money
                            # Will be purchased in the main purchase block below
                            buy_amount += additional_buy
                    else:
                        # If risk is too high or potential buy is sufficient, use regular percentage or fixed amount
                        if potential_buy < 500:
                            buy_amount = np.select(
                                [
                                    (0.7 <= risk < 0.8), (0.6 <= risk < 0.7), (0.5 <= risk < 0.6),
                                    (0.4 <= risk < 0.5), (0.3 <= risk < 0.4), (0.2 <= risk < 0.3),
                                    (0.1 <= risk < 0.2), (risk < 0.1)
                                ],
                                [50, 100, 150, 200, 250, 400, 500, 600],
                                default=0
                            )
                            strategy_new_money_invested += buy_amount  # Track new money since we're using fixed amounts
                        else:
                            buy_amount = potential_buy
                            # Don't increment strategy_new_money_invested here since this is reinvestment

                # Actually make the purchase - now all purchases happen here
                if buy_amount > 0:  # Only buy if we have an amount to invest
                    strategy_shares += buy_amount / price
                    peak_shares = max(peak_shares, strategy_shares)  # Update peak shares after buying
                    strategy_cashflow -= buy_amount  # Purchase is negative cashflow
                    strategy_invested += buy_amount
                    cash_indicator = 0

        strategy_cashflow_history.append(strategy_cashflow)

        # Update portfolio metrics daily
        strategy_portfolio_value = strategy_shares * price
        data.loc[i, 'Strategy_Portfolio'] = strategy_portfolio_value
        data.loc[i, 'Strategy_Cashflow'] = strategy_cashflow
        data.loc[i, 'Strategy_Invested'] = strategy_invested
        data.loc[i, 'Strategy_ProfitLoss'] = strategy_portfolio_value + strategy_cashflow
        data.loc[i, 'Benchmark_MaxNegativeCashflow'] = -benchmark_invested  # Track total capital invested for benchmark
        data.loc[i, 'Strategy_MaxNegativeCashflow'] = -strategy_new_money_invested  # Only track new money invested
        data.loc[i, 'Benchmark_Profit_New'] = benchmark_shares * price - benchmark_invested
        data.loc[i, 'Strategy_Profit_New'] = strategy_portfolio_value - strategy_new_money_invested  # Use new money invested for profit calculation
        
    # Calculate percentage profits - showing total return (100% means original investment)
    data['Benchmark_Percentage_Profit'] = np.where(
        data['Benchmark_MaxNegativeCashflow'] != 0,
        ((data['Benchmark_ProfitLoss'] / abs(data['Benchmark_MaxNegativeCashflow'])) * 100),
        0
    )
    
    data['Strategy_Percentage_Profit'] = np.where(
        data['Strategy_MaxNegativeCashflow'] != 0,
        ((data['Strategy_ProfitLoss'] / abs(data['Strategy_MaxNegativeCashflow'])) * 100),
        0
    )
    
    # Replace infinite values with NaN
    data['Benchmark_Percentage_Profit'] = data['Benchmark_Percentage_Profit'].replace([np.inf, -np.inf], np.nan)
    data['Strategy_Percentage_Profit'] = data['Strategy_Percentage_Profit'].replace([np.inf, -np.inf], np.nan)
    
    return data

def plot_strategy_results(data):
    # Common chart settings
    chart_settings = dict(
        template="plotly_white",
        height=800,
        font=dict(size=24, color='black'),  # Increased base font size
        xaxis=dict(
            tickformat='%Y',  # Show only years
            tickangle=0,      # Horizontal labels
            title_font=dict(size=32, color='black'),  # Increased axis title
            tickfont=dict(size=24),  # Increased tick labels
        ),
        yaxis=dict(
            title_font=dict(size=32, color='black'),  # Increased axis title
            tickfont=dict(size=24),  # Increased tick labels
        ),        
        title_font=dict(size=36, color='black'),  # Increased chart title
        legend=dict(
            font=dict(size=24),  # Increased legend text
            bgcolor='rgba(255,255,255,0.9)',  # More opaque background
            bordercolor='black',  # Add border
            borderwidth=1,
            yanchor="middle",
            y=0.5,  # Center vertically
            xanchor="left",
            x=1.02  # Position just outside the right side of the plot
        )
    )    # Portfolio Values Plot
    fig = go.Figure()
    # Portfolio value (shares × SP500 price)
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Benchmark_Portfolio'], mode='lines', name='DCA Portfolio Value', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Strategy_Portfolio'], mode='lines', name='dDCA Portfolio Value', line=dict(color='orange')))
    
    # Total value (portfolio + cash if any from sales)
    fig.add_trace(go.Scatter(x=data['Date'], 
                            y=data['Benchmark_Portfolio'] + data['Benchmark_Cashflow'].clip(lower=0),  # Only add positive cash
                            mode='lines', name='DCA Total Value', 
                            line=dict(color='cyan', dash='dot')))
    fig.add_trace(go.Scatter(x=data['Date'], 
                            y=data['Strategy_Portfolio'] + data['Strategy_Cashflow'].clip(lower=0),  # Only add positive cash
                            mode='lines', name='dDCA Total Value', 
                            line=dict(color='red', dash='dot')))
    fig.update_layout(
        title="Portfolio Performance: DCA vs. dDCA",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        legend_title="Portfolio",
        **chart_settings
    )
    fig.show()

    # Monthly Cashflows Plot (investments and withdrawals)
    fig_monthly = go.Figure()
    # Group data by month and take first value of each month for accumulated cashflows
    monthly_data = data.groupby(pd.Grouper(key='Date', freq='MS')).agg({
        'Benchmark_Cashflow': 'first',
        'Strategy_Cashflow': 'first',
        'Benchmark_Portfolio': 'first',
        'Strategy_Portfolio': 'first'
    }).reset_index()
    
    # Calculate actual monthly cashflows (difference between consecutive accumulated values)
    monthly_benchmark = -monthly_data['Benchmark_Cashflow'].diff().fillna(-monthly_data['Benchmark_Cashflow'])
    monthly_strategy = -monthly_data['Strategy_Cashflow'].diff().fillna(-monthly_data['Strategy_Cashflow'])
    
    # For the last month, replace cashflow with portfolio value
    if len(monthly_benchmark) > 0:
        monthly_benchmark.iloc[-1] = monthly_data['Benchmark_Portfolio'].iloc[-1]
        monthly_strategy.iloc[-1] = monthly_data['Strategy_Portfolio'].iloc[-1]

    fig_monthly.add_trace(go.Bar(x=monthly_data['Date'], y=monthly_benchmark, name='DCA Monthly Cashflow', marker_color='blue'))
    fig_monthly.add_trace(go.Bar(x=monthly_data['Date'], y=monthly_strategy, name='dDCA Monthly Cashflow', marker_color='orange'))
    fig_monthly.update_layout(
        title="Monthly Investments/Withdrawals: DCA vs. dDCA",
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        barmode='group',
        **chart_settings
    )
    fig_monthly.show()

    # Cashflow Plot
    fig_cashflow = go.Figure()
    fig_cashflow.add_trace(go.Scatter(x=data['Date'], y=data['Benchmark_Cashflow'], mode='lines', name='DCA Cashflow', line=dict(color='blue')))
    fig_cashflow.add_trace(go.Scatter(x=data['Date'], y=data['Strategy_Cashflow'], mode='lines', name='dDCA Cashflow', line=dict(color='orange')))
    fig_cashflow.add_trace(go.Scatter(x=data['Date'], y=data['Benchmark_MaxNegativeCashflow'], mode='lines', name='DCA Capital Required', line=dict(color='red')))
    fig_cashflow.add_trace(go.Scatter(x=data['Date'], y=data['Strategy_MaxNegativeCashflow'], mode='lines', name='dDCA Capital Required', line=dict(color='green')))
    fig_cashflow.update_layout(
        title="Cumulative Cashflow: DCA vs. dDCA",
        xaxis_title="Date",
        yaxis_title="Cashflow ($)",
        legend_title="Cashflow",
        **chart_settings
    )
    fig_cashflow.show()

    # Percentage Profit Plot
    fig_percentage_profit = go.Figure()
    fig_percentage_profit.add_trace(go.Scatter(x=data['Date'], y=data['Benchmark_Percentage_Profit'], mode='lines', name='DCA % Profit', line=dict(color='green')))
    fig_percentage_profit.add_trace(go.Scatter(x=data['Date'], y=data['Strategy_Percentage_Profit'], mode='lines', name='dDCA % Profit', line=dict(color='purple')))
    fig_percentage_profit.update_layout(
        title="Percentage Profit Evolution: DCA vs. dDCA",
        xaxis_title="Date",
        yaxis_title="Percentage Profit (%)",
        legend_title="Percentage Profit",
        **chart_settings
    )
    fig_percentage_profit.show()

    # Drawdown Plot (from peak, using Total Value)
    # Calculate Total Value for both strategies
    benchmark_total = data['Benchmark_Portfolio'] + data['Benchmark_Cashflow'].clip(lower=0)
    strategy_total = data['Strategy_Portfolio'] + data['Strategy_Cashflow'].clip(lower=0)

    # Calculate running peak
    benchmark_peak = benchmark_total.cummax()
    strategy_peak = strategy_total.cummax()

    # Calculate drawdown as percentage from peak
    benchmark_drawdown = 100 * (benchmark_total - benchmark_peak) / benchmark_peak
    strategy_drawdown = 100 * (strategy_total - strategy_peak) / strategy_peak

    # Plot drawdown
    fig_drawdown = go.Figure()
    fig_drawdown.add_trace(go.Scatter(x=data['Date'], y=benchmark_drawdown, mode='lines', name='DCA Drawdown', line=dict(color='cyan')))
    fig_drawdown.add_trace(go.Scatter(x=data['Date'], y=strategy_drawdown, mode='lines', name='dDCA Drawdown', line=dict(color='red')))
    fig_drawdown.update_layout(
        title="Drawdown from Peak: DCA vs. dDCA (Total Value)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        legend_title="Drawdown",
        **chart_settings
    )
    fig_drawdown.show()