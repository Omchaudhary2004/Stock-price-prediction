import yfinance as yf
import time
ticker = "BTC-USD"
ticker = yf.Ticker(ticker)

while True:
    current_price = ticker.history(period="1d")['Close'].iloc[-1]
    print(f"Current Price: ${current_price:.2f}")
    time.sleep(1)