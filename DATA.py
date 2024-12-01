import yfinance as yf
import pandas as pd
import numpy as np


# Getting the Live Prices
def Live_price(ticker):
# ticker = input("Enter the Ticker of the stock:")
# # Create a Ticker object for Bitcoin
    ticker = yf.Ticker(ticker)
    current_price = ticker.history(period="1d")['Close'].iloc[-1]
    print(f"Current Price: ${current_price:.2f}")
    print(current_price)
    return current_price


# it downloads the stock data straight from start 
def D_data(ticker):
    ticker = yf.Ticker(ticker)
    stock = ticker
    historical_data = stock.history(period="max")
    historical_data.to_csv(f"_historical_data.csv")

    # print(f"Data for  has been downloaded and saved as _historical_data.csv")
    # Display the data
    # print(historical_data)
    return

# it gives the dataframe which can be processed by the ml model 
def frame(ticker):
    ticker = yf.Ticker(ticker)
    current_price = ticker.history(period="1d")
    return current_price