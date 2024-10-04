import yfinance as yf
import time

from ti import lists
#taking input from user for the stock
# ticker = input("Enter the Ticker of the stock:")
# ticker = "^NSEI"
gui = lists

# Create a Ticker object for Bitcoin
for i in gui:
    ticker = i
    ticker = yf.Ticker(ticker)

    # Get the current price
    # print(ticker.info)
    # print(ticker.history(period = "max"))
    current_price = ticker.history(period="1d")['Close'].iloc[-1]
    print(f"Current Price: ${current_price:.2f}")


# Replace 'TICKER' with the desired stock ticker symbol


# Download 15-minute data since the beginning
# data = yf.download(ticker, interval='15m', start='2007-9-1')  # Adjust the start date as needed

# print(data)
