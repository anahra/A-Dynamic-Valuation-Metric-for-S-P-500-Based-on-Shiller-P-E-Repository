import sys
import os
import pandas as pd

# Add the current directory to the path to ensure imports work
sys.path.append(os.getcwd())

from data.shiller_pe_loader import load_shiller_pe

def test_live_data():
    print("Testing live data loading...")
    try:
        data_pe, data_sp500 = load_shiller_pe()
        
        print("\n--- Data PE (Last 5 rows) ---")
        print(data_pe.tail())
        
        print("\n--- Data SP500 (Last 5 rows) ---")
        print(data_sp500.tail())
        
        last_date = data_pe['Date'].iloc[-1]
        second_last_date = data_pe['Date'].iloc[-2]
        
        print(f"\nLast Date: {last_date}")
        print(f"Second Last Date: {second_last_date}")
        
        # Basic checks
        if last_date > second_last_date:
            print("\nSUCCESS: Last date is more recent than the second to last date.")
        else:
            print("\nWARNING: Last date is NOT more recent. Live data might not have been appended (or it is the same day).")
            
        # Check if last date is recent (e.g. within last 5 days)
        if (pd.Timestamp.now() - last_date).days < 5:
             print("SUCCESS: Last date is recent.")
        else:
             print("WARNING: Last date seems old.")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_live_data()
