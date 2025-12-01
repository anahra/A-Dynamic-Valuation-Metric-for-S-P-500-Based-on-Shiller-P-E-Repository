
import pandas as pd
import sys
import os

# Add the current directory to the path so we can import modules
sys.path.append(os.getcwd())

from data import shiller_pe_loader
from risk import shiller_pe_risk_web
from strategies import strat_test_web

def analyze_recent_sell():
    print("Loading data...")
    df = shiller_pe_loader.load_data("http://www.econ.yale.edu/~shiller/data/ie_data.xls")
    
    print("Computing risk metrics...")
    df = shiller_pe_risk_web.compute_risk_metrics(df)
    
    print("Running strategy simulation...")
    results = strat_test_web.run_strategy(df)
    
    # Filter for positive cashflows (sells)
    # Note: Strategy_Cashflow is cumulative? No, let's check the code again.
    # Line 28: strategy_cashflow = 0
    # Line 61: strategy_cashflow += sell_amount
    # Line 138: strategy_cashflow -= buy_amount
    # Line 147: data.loc[i, 'Strategy_Cashflow'] = strategy_cashflow
    # So Strategy_Cashflow IS cumulative.
    # Wait, if it's cumulative, I can't just check for > 0.
    # I need to check for *changes* in cashflow that are positive.
    # Or, I can look at the loop logic again.
    # The loop updates `strategy_cashflow` variable and then assigns it to the dataframe column.
    # So `Strategy_Cashflow` column tracks the *cumulative* cashflow (net money in/out).
    # A sell *increases* this value. A buy *decreases* it.
    # So I need to calculate the difference between rows.
    
    results['Cashflow_Change'] = results['Strategy_Cashflow'].diff()
    
    # Filter for positive changes (sells)
    sells = results[results['Cashflow_Change'] > 1] # Use > 1 to avoid floating point noise
    
    if not sells.empty:
        last_sell = sells.iloc[-1]
        print("\n--- Most Recent Sell Event ---")
        print(f"Date: {last_sell['Date'].strftime('%Y-%m-%d')}")
        print(f"Portfolio Value: ${last_sell['Strategy_Portfolio']:,.2f}")
        print(f"Sell Amount (Cashflow Change): ${last_sell['Cashflow_Change']:,.2f}")
        print(f"Cumulative Cashflow after sell: ${last_sell['Strategy_Cashflow']:,.2f}")
        print(f"Risk Level: {last_sell['Risk']:.4f}")
        print(f"Shiller PE: {last_sell['CAPE']:.2f}")
    else:
        print("\nNo sell events found.")

if __name__ == "__main__":
    analyze_recent_sell()
