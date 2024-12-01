import yfinance as yf
import pandas as pd
import numpy as np
import DATA as dt
import ML as ml
# Getting and downloading the data 
# stock = "^NSEI"
while True:
    stock = input("Enter the Ticker of the stock:")
    current_price = dt.Live_price(stock)
    dt.D_data(stock)


    current_frame = dt.frame(stock)
    list = current_frame.values.tolist()
    for i in list:
        j=i[1:]

    # predicting using randomforest
    price =  ml.predict_price([j])

    if price > current_price:
        print(f"So the market is going to open in gap up at {price} and you should put money it is currently at {current_price} ")
        print(f"{price-current_price}:Points Gap up")

    elif price < current_price:
        print(f"So the market is going to open in gap Down at {price} and you should put money it is currently at {current_price} ")
        print(f"{current_price-price}:Points GAP DOWN")
    else :
        print("price is gonna be Neutral")
    
    # Ask the user if they want to repeat or exit
    choice = input("Do you want to repeat or exit? (type 'Y' or 'N'): ").strip().lower()
    
    if choice == 'n':
        print("Goodbye!")
        break  # Exit the loop
    elif choice == 'y':
        print("Repeating the process...\n")
        continue  # Repeat the loop
    else:
        print("Invalid input. Please type 'repeat' or 'exit'.\n")

