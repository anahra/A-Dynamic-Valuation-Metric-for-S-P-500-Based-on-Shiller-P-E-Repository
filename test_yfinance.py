import yfinance as yf
import pandas as pd

print("Testing yfinance...")

try:
    print("\n--- Test 1: Ticker.history('^GSPC', period='5d') ---")
    ticker = yf.Ticker("^GSPC")
    hist = ticker.history(period="5d")
    print(hist)
except Exception as e:
    print(f"Test 1 Failed: {e}")

try:
    print("\n--- Test 2: yf.download('^GSPC', period='5d') ---")
    data = yf.download("^GSPC", period="5d")
    print(data)
except Exception as e:
    print(f"Test 2 Failed: {e}")

try:
    print("\n--- Test 3: Ticker.history('AAPL', period='5d') ---")
    ticker = yf.Ticker("AAPL")
    hist = ticker.history(period="5d")
    print(hist)
except Exception as e:
    print(f"Test 3 Failed: {e}")
