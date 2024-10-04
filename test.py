import tkinter as tk
from PIL import ImageTk, Image
# Consider using `requests` for downloading images if necessary

# Function to handle adding tickers
def handle_ticker():
    ticker = ticker_input_lable.get()

    # Check if ticker already exists
    if ticker in lists:
        messagebox.showinfo("Ticker not added", "It already exists")
        return

    lists.append(ticker)

    # Create frame and label, updating grid positions
    frame_count = len(lists)  # Track the number of frames created
    col = frame_count % 3  # Determine column based on frame count
    row = (frame_count // 3) + 5  # Start below buttons and input fields

    frame = tk.Frame(root, bg="lightblue", width=500, height=400)
    frame.grid(row=row, column=col, padx=10, pady=10)
    label = tk.Label(frame, text=ticker, bg="lightblue")
    label.grid(row=0, column=0, padx=10, pady=10)

# Function to handle removing tickers (optional)
def remv_ticker():
    # Implement logic to remove ticker from `lists` and potentially
    # destroy the corresponding frame
    pass  # Placeholder for removal functionality

# Main application setup
lists = []  # List to store tickers

root = tk.Tk()
root.title("STOCK PRICES")

# Set icon (consider using `requests` to download remotely)
# root.iconbitmap("Red Illustrated Bull Stock Broker Logo.png")

# Set minimum window size
root.minsize(500, 500)

# Configure background color
root.configure(background="#131722")

# Load and resize logo image (error handling recommended)
try:
    img = Image.open("Red Illustrated Bull Stock Broker Logo.png")
    resize_img = img.resize((100, 100))
except FileNotFoundError:
    print("Logo image not found. Consider using 'requests' for remote images.")
    resize_img = None

# Display logo (conditional based on image availability)
if resize_img:
    imag = ImageTk.PhotoImage(resize_img)
    img_lable = tk.Label(root, image=imag)
    img_lable.grid(row=0, column=1, pady=(10, 10), padx=(10, 10))

# Text label for app title
text_lable = tk.Label(root, text="LIVE TICKER TRACKER", fg="white", bg="#131722")
text_lable.grid(row=1, column=1)
text_lable.config(font=("verdana"))

# Input field for ticker symbol
ticker_text_lable = tk.Label(root, text="Enter your ticker", fg="white", bg="#131722")
ticker_text_lable.grid(row=2, column=1, pady=(10, 5))
ticker_text_lable.config(font=("verdana", 10))

ticker_input_lable = tk.Entry(root, width=20)
ticker_input_lable.grid(row=3, column=1, ipady=3)

# Button to add ticker
ADD_button = tk.Button(root, text="Add Ticker", fg="black", bg="white", command=handle_ticker)
ADD_button.grid(row=4, column=1, pady=3)

# Optional input field for removing tickers
tickerRmv_text_lable = tk.Label(root, text="Enter ticker to remove", fg="white", bg="#131722")
tickerRmv_text_lable.grid(row=2, column=2, pady=(10, 5))

tickerRmv_input_lable = tk.Entry(root, width=20)
tickerRmv_input_lable.grid(row=3, column=2, ipady=3)

# Optional button to remove tickers
# RMV_button = tk.Button(root, text="Remove Ticker", fg="black", bg="white", command=remv_ticker)
# RMV_button.grid(row=4, column=2
from tkinter import *
from PIL import ImageTk,Image
# you have to pip install pillow (pip install Pillow)
from tkinter import messagebox

import yfinance as yf
 
# # creating function for the button 
# def handle_ticker():
#     i=0
#     j=5
#     ticker = ticker_input_lable.get()
#     if ticker in lists:
#         messagebox.showinfo("Ticker not added", "It already exists ")
#     else:
#         print(ticker)
#         lists.append(ticker)
#         print(lists)
#         if i<=3:
#             frame = Frame(root, bg="lightblue", width=500, height=400)
#             frame.grid(row=j,column=i,padx=10, pady=10)
#             label = Label(frame, text= ticker_input_lable.get(), bg="lightblue")
#             label.grid(row=j,column=i,padx=10, pady=10)
#             i+=1
#         else:
#             frame = Frame(root, bg="lightblue", width=500, height=400)
#             frame.grid(row=j,column=i,padx=10, pady=10)
#             label = Label(frame, text= ticker_input_lable.get(), bg="lightblue")
#             label.grid(row=j,column=i,padx=10, pady=10)
#             j+=1
#             i=1
            
        

# lists=[]
    
# def remv_ticker():
#     rmv_ticker = tickerRmv_input_lable.get()
#     print(rmv_ticker)
#     if rmv_ticker in lists:
#         print("removed")
#     else:
#         messagebox.showinfo("Ticker not added", "This ticker was not added so add it first to remove it. ")
    

# # creating object
# root = Tk()
# # changing title 
# root.title("STOCK PRICES")

# # changing icon
# root.iconbitmap("Red Illustrated Bull Stock Broker Logo.png")

# # setting minimum size of window 
# # root.geometry(500,500)
# root.minsize(500,500)

# # changing background colour
# root.configure(background="#131722")
# img = Image.open("Red Illustrated Bull Stock Broker Logo.png")
# resize_img =  img.resize((100,100))
# imag = ImageTk.PhotoImage(resize_img)
# img_lable = Label(root,image=imag)
# img_lable.grid(row=0,column=1,pady=(10,10),padx=(10,10))

# text_lable = Label(root,text="LIVE TICKER TRACKER",fg="white",bg="#131722")
# text_lable.grid(row=1,column=1)
# text_lable.config(font=("verdana"))

# # taking input ticker
# ticker_text_lable= Label(root,text="Enter your ticker",fg="white",bg="#131722")
# ticker_text_lable.grid(row=2,column=1,pady=(10,5))
# ticker_text_lable.config(font=("verdana",10))

# ticker_input_lable = Entry(root,width=20 )
# ticker_input_lable.grid(row=3,column=1,ipady=3)

# ADD_button = Button(root,text="Add tiker",fg="black",bg="white", command= handle_ticker)
# ADD_button.grid(row=4,column=1,pady=3)

# tickerRmv_text_lable= Label(root,text="Enter your ticker to remove",fg="white",bg="#131722")
# tickerRmv_text_lable.grid(row=2,column=2,pady=(10,5))
# tickerRmv_input_lable = Entry(root,width=20 )
# tickerRmv_input_lable.grid(row=3,column=2,ipady=3)
# RMV_button = Button(root,text="Remove tiker",fg="black",bg="white", command= remv_ticker)
# RMV_button.grid(row=4,column=2,pady=3)


# starting tthe window 
root.mainloop()