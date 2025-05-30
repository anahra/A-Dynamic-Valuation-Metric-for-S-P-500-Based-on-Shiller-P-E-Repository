"""
Module for analyzing strategy performance across different time periods.
Provides functionality to compare investment strategies against benchmarks.
"""

import pandas as pd
import numpy as np
from .strat_test import run_strategy
from .analyze_sharpe_ratio import calculate_cashflows, analyze_risk_adjusted_returns

def analyze_by_decades(risk_data, start_year=1950, end_year=2025, initial_investment=0, monthly_investment=200):
    """
    Analyze strategy performance from the start of each decade until the present.
    
    Args:
        risk_data (pd.DataFrame): Risk metrics data
        start_year (int): Starting year for analysis
        end_year (int): Ending year for analysis (typically present year)
        initial_investment (float): Initial investment amount
        monthly_investment (float): Monthly investment amount
    
    Returns:
        tuple: (returns_df, risk_metrics_df) containing separate DataFrames for returns and risk metrics
    """
    decades = list(range(start_year, end_year - 9, 10))  # Stop 9 years before end to ensure at least one full decade
    returns_data = []
    risk_metrics_data = []

    for decade_start in decades:
        print(f"Analyzing period {decade_start}-{end_year}...")
        results = run_strategy(risk_data, start_year=decade_start, 
                             initial_investment=initial_investment, 
                             monthly_investment=monthly_investment)
        
        if not results.empty:            # Get risk-adjusted metrics for this period
            _, metrics = analyze_risk_adjusted_returns(results)
            
            # Calculate returns based on maximum negative cashflow
            benchmark_return = ((results['Benchmark_Portfolio'].iloc[-1] / 
                               abs(results['Benchmark_MaxNegativeCashflow'].iloc[-1]) - 1) * 100)
            strategy_return = ((results['Strategy_Portfolio'].iloc[-1] / 
                              abs(results['Strategy_MaxNegativeCashflow'].iloc[-1]) - 1) * 100)
            
            # Calculate total value (portfolio + positive cash) for drawdown calculation
            benchmark_total = results['Benchmark_Portfolio'] + results['Benchmark_Cashflow'].clip(lower=0)
            strategy_total = results['Strategy_Portfolio'] + results['Strategy_Cashflow'].clip(lower=0)

            # Calculate running peak and drawdown
            benchmark_peak = benchmark_total.cummax()
            strategy_peak = strategy_total.cummax()
            benchmark_drawdown = 100 * (benchmark_total - benchmark_peak) / benchmark_peak
            strategy_drawdown = 100 * (strategy_total - strategy_peak) / strategy_peak

            # Find maximum drawdown (most negative value)
            max_benchmark_drawdown = round(benchmark_drawdown.min(), 2)
            max_strategy_drawdown = round(strategy_drawdown.min(), 2)

            # Returns data
            returns_data.append({
                'Period': f"{decade_start}-{end_year}",
                'Years': end_year - decade_start,
                'Benchmark Return %': round(benchmark_return, 2),
                'Strategy Return %': round(strategy_return, 2),
                'Outperformance %': round(strategy_return - benchmark_return, 2),
                'Benchmark Max Drawdown %': max_benchmark_drawdown,
                'Strategy Max Drawdown %': max_strategy_drawdown
            })
            
            # Risk metrics data
            risk_metrics_data.append({
                'Period': f"{decade_start}-{end_year}",
                'Years': end_year - decade_start,
                'Benchmark Sharpe': round(metrics['Benchmark Sharpe Ratio'], 3),
                'Strategy Sharpe': round(metrics['Strategy Sharpe Ratio'], 3),
                'Benchmark IRR %': round(metrics['Benchmark IRR (Annual)'] * 100, 2),
                'Strategy IRR %': round(metrics['Strategy IRR (Annual)'] * 100, 2)
            })

    return pd.DataFrame(returns_data), pd.DataFrame(risk_metrics_data)
