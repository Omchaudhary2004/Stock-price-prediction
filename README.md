# Stock Gap Prediction Model

This project is a **stock market gap prediction tool** built using machine learning. The tool uses historical stock data to predict whether the stock will open **Gap Up**, **Gap Down**, or **Neutral** on the next trading day.

---

## 🚀 Features

- **Prediction using Random Forest Classifier**: Utilizes historical stock data to make predictions.
- **Input Stock Ticker**: Accepts stock tickers (from [Yahoo Finance](https://finance.yahoo.com)) as input for predictions.
- **Automatic Data Fetching**: Retrieves historical data using the `yfinance` library.
- **Easy-to-Run Main Script**: Simply run the main file and follow the prompts.

---

## 🛠️ Libraries and Dependencies

The project leverages the following Python libraries:
- `sklearn`: For the Random Forest Classifier and evaluation metrics.
- `pandas`: For data handling and manipulation.
- `numpy`: For numerical operations.
- `yfinance`: To fetch historical stock data from Yahoo Finance.

---

## 📂 Project Structure
The project consists of three main Python files:

- **`main.py`:** Entry point, handles user interaction, and calls functions from other modules.
- **`ml.py`:** Implements the Random Forest model for prediction.
- **`data.py`:** Handles data retrieval, preprocessing, and price-related operations using yfinance.


---

## ⚡ How to Run
### Run this command
```bash
python main.py
```
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/stock-gap-prediction.git
cd stock-gap-prediction
