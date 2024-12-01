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

- **`main.py`**: The entry point for the application. Prompts the user for a stock ticker and runs the prediction.
- **`models/`**: Contains the Random Forest model and related utilities.
- **`data/`**: (Optional) Can store preprocessed or raw historical data if required.
- **`README.md`**: Documentation for the repository.

---

## ⚡ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/stock-gap-prediction.git
cd stock-gap-prediction
