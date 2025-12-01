import sys
import os
import pandas as pd

# Add current directory to path
sys.path.append(os.getcwd())

from data.shiller_pe_loader import load_shiller_pe

print("Loading data using load_shiller_pe...")
data_pe, data_sp500 = load_shiller_pe()

print(f"PE Data Rows: {len(data_pe)}")
print(f"S&P 500 Data Rows: {len(data_sp500)}")

# Check for duplicates in returned data
def check_dupes(df, name):
    df['Month'] = df['Date'].dt.to_period('M')
    current_period = pd.Timestamp.now().to_period('M')
    
    # Check duplicates for historical months
    historical = df[df['Month'] < current_period]
    counts = historical.groupby('Month').size()
    dupes = counts[counts > 1]
    
    if not dupes.empty:
        print(f"FAIL: Found duplicates in historical {name}:", flush=True)
        print(dupes, flush=True)
    else:
        print(f"PASS: No duplicates in historical {name}", flush=True)
        
    # Check current month
    current = df[df['Month'] == current_period]
    print(f"INFO: {name} entries for current month ({current_period}): {len(current)}", flush=True)
    if not current.empty:
        print(current, flush=True)

print("Checking PE Data...", flush=True)
check_dupes(data_pe, "PE Data")
print("Checking S&P 500 Data...", flush=True)
check_dupes(data_sp500, "S&P 500 Data")

print("Performing Merge...", flush=True)
data = pd.merge(data_pe, data_sp500, on='Date', how='inner')
data.drop_duplicates(subset=['Date'], inplace=True)

print(f"Merged Data Rows: {len(data)}", flush=True)
check_dupes(data, "Merged Data")

# Check if values are different for the current month
current_period = pd.Timestamp.now().to_period('M')
current_month_data = data[data['Date'].dt.to_period('M') == current_period]

if len(current_month_data) >= 2:
    start_val = current_month_data.iloc[0]['S&P_500']
    latest_val = current_month_data.iloc[-1]['S&P_500']
    print(f"\nComparing Current Month Values ({current_period}):")
    print(f"Start of Month: {start_val}")
    print(f"Latest (Live):  {latest_val}")
    
    if start_val != latest_val:
        print("PASS: Latest value is different from Start of Month value (Live update working).")
    else:
        print("WARNING: Latest value is SAME as Start of Month value (Live update might not be working or market hasn't moved).")
else:
    print("WARNING: Less than 2 entries for current month.")
