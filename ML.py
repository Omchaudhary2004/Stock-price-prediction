import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings 
def predict_price(data):
    df = pd.read_csv("_historical_data.csv")
    x = df[[ 'High', 'Low', 'Close', 'Volume', 'Dividends','Stock Splits']]
    y = df['Open']
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=1)
    model_rf= RandomForestRegressor(random_state=1)
    model_rf.fit(x_train,y_train)
    y_pred_rf = model_rf.predict(data)
    warnings.filterwarnings('ignore')
    # print(y_pred_rf)
    return y_pred_rf
