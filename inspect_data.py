import pandas as pd
import os

data_dir = r"c:\Users\agusn\OneDrive - HEC Paris\Documentos\0_General\3_Educación\1_HEC\3_M2\1_Research Paper\A-Dynamic-Valuation-Metric-for-S-P-500-Based-on-Shiller-P-E\data"
pe_path = os.path.join(data_dir, "shiller_pe_data.csv")
sp500_path = os.path.join(data_dir, "sp500_data.csv")

def check_frequency(path, name):
    if not os.path.exists(path):
        print(f"{name} not found at {path}")
        return

    df = pd.read_csv(path, parse_dates=['Date'])
    print(f"--- {name} ---")
    print(f"Total rows: {len(df)}")
    
    # Check for multiple entries in the same month
    df['Month'] = df['Date'].dt.to_period('M')
    counts = df.groupby('Month').size()
    duplicates = counts[counts > 1]
    
    if not duplicates.empty:
        print(f"Found {len(duplicates)} months with multiple entries:")
        print(duplicates.head())
        # Show examples
        for month in duplicates.head(3).index:
            print(f"Entries for {month}:")
            print(df[df['Month'] == month])
    else:
        print("No months with multiple entries found.")

check_frequency(pe_path, "Shiller PE")
check_frequency(sp500_path, "S&P 500")
