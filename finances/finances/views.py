# views.py
from django.shortcuts import render
from django.http import JsonResponse
# from finances.logic.data import Datacollection  # Import your function
import json
import time
#############################################################
import urllib.parse
import pandas as pd
import requests
import requests
from dateutil.relativedelta import relativedelta
from datetime import datetime
from datetime import date
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
#############################################################

apikey = '12780cc6-a7e4-466c-9488-3de58d39ada5'
secretkey = 'ute2ha7ln6'
uris = 'https://api.upstox.com/v2/login'
uri = f'https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={apikey}&redirect_uri={uris}'
print(uri)

def index(request):
    return render(request, 'index.html')

# def run_option_chain(request):
#     # Call your function
#     result = Datacollection.fetch_and_save_option_chain()
    
#     # Return the result as JSON
#     return JsonResponse({'result': result})

def logins(request):
    return render(request,'login.html')

def about(request):
    return render(request , 'about.html')

def marketoverview(request):
    return render(request , 'marketoverview.html')

def markettai(request):
    if request.method == 'POST' and request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # Get form data from AJAX request
        data = json.loads(request.body)
        code = data.get('code')
        modem = data.get('model')
        timeframe = data.get('timeframe')
        
        # Your processing code here
        # This is where you would run your LSTM model with the given parameters
        
        ###################################################################
        def logins(code):
            apikey = '12780cc6-a7e4-466c-9488-3de58d39ada5'
            secretkey = 'ute2ha7ln6'
            
            uris = 'https://api.upstox.com/v2/login'
            uri = f'https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={apikey}&redirect_uri={uris}'
            # print(uri)
            # code = input('enter your code:')
            
            url = 'https://api.upstox.com/v2/login/authorization/token'

            headers ={
                'accept': 'application/json' ,
                'Content-Type':'application/x-www-form-urlencoded' 
            }

            data = {
                'code=<Your-Auth-Code-Here>&client_id=<Your-API-Key-Here>&client_secret=<Your-API-Secret-Here>&redirect_uri=<Your-Redirect-URI-Here>&grant_type=authorization_code'
            }
            url = 'https://api.upstox.com/v2/login/authorization/token'
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/x-www-form-urlencoded',
            }

            data = {
                'code': code,
                'client_id': apikey,
                'client_secret': secretkey,
                'redirect_uri': uris,
                'grant_type': 'authorization_code',
            }

            response = requests.post(url, headers=headers, data=data)

            print(response.status_code)
            print(response.json())
            access_token = response.json()['access_token']
            return access_token






        def minute_data(access_token):
            to_date = str(date.today()) #todays date 
            today = datetime.now()
            from_date = (today - relativedelta(months=5)).strftime("%Y-%m-%d")

            print(from_date)

            index = "NSE_INDEX|Nifty 50"
            url = f'https://api.upstox.com/v2/historical-candle/{index}/1minute/{to_date}/{from_date}'

            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {access_token}'
            }

            response = requests.get(url, headers=headers)

            # Check the response status
            if response.status_code == 200:
                # Do something with the response data (e.g., print it)
                print(response.json())
                data = response.json()
                data = data['data']['candles']
                columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Volume_2']
                df = pd.DataFrame(data, columns=columns)
                return(df)
            else:
                # Print an error message if the request was not successful
                print(f"Error: {response.status_code} - {response.text}")





        def year_data(access_token):
            to_date = str(date.today()) #todays date 
            today = datetime.now()
            from_date = (today - relativedelta(months=5))
            year_date = (from_date - relativedelta(year=2)).strftime("%Y-%m-%d")
            print(year_date)

            index = "NSE_INDEX|Nifty 50"
            url = f'https://api.upstox.com/v2/historical-candle/{index}/day/{to_date}/{year_date}'

            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {access_token}'
            }

            response = requests.get(url, headers=headers)

            # Check the response status
            if response.status_code == 200:
                # Do something with the response data (e.g., print it)
                print(response.json())
                
                
            else:
                # Print an error message if the request was not successful
                print(f"Error: {response.status_code} - {response.text}")


            data = response.json()
            data = data['data']['candles']
            columns = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume', 'Volume_2']
            df_year = pd.DataFrame(data, columns=columns)
            return(df_year)


        def concat_ohcv_dataframes(df_5months, df_1day):
            """
            Concatenates two OHLCV DataFrames, ensuring no overlapping dates.

            Args:
                df_5months: DataFrame containing 5 months of 1-minute data.
                df_1day: DataFrame containing 1 day of data from the start of NIFTY history.

            Returns:
                A concatenated DataFrame with no overlapping dates.
            """

            # Ensure 'Datetime' column is datetime type and index
            df_5months['Datetime'] = pd.to_datetime(df_5months['Datetime'])
            df_1day['Datetime'] = pd.to_datetime(df_1day['Datetime'])

            df_5months = df_5months.set_index('Datetime')
            df_1day = df_1day.set_index('Datetime')

            # Find the earliest date in df_5months
            earliest_5months = df_5months.index.min()

            # Filter df_1day to exclude dates that are also in df_5months
            df_1day_filtered = df_1day[df_1day.index < earliest_5months]

            # Concatenate the filtered DataFrames
            concatenated_df = pd.concat([df_1day_filtered, df_5months]).sort_index()

            # Reset index to make 'Datetime' a column again
            concatenated_df = concatenated_df.reset_index()

            return concatenated_df


        def import_dataframe_to_sqlite(dataframe, database_name, table_name):
            """
            Imports a pandas DataFrame into an SQLite database, creating the table if it doesn't exist.

            Args:
                dataframe: The pandas DataFrame to import.
                database_name: The name of the SQLite database file.
                table_name: The name of the table to create or append to.
            """

            try:
                conn = sqlite3.connect(database_name)
                cursor = conn.cursor()

                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume INTEGER,
                        volume_2 INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """)

                # Prepare data for insertion
                data_to_insert = []
                for index, row in dataframe.iterrows():
                    data_to_insert.append((
                        row['Open'],
                        row['High'],
                        row['Low'],
                        row['Close'],
                        row['Volume'],
                        row['Volume_2'],
                    ))

                # Insert data into the table
                cursor.executemany(f"""
                    INSERT INTO {table_name} (open, high, low, close, volume, volume_2)
                    VALUES (?, ?, ?, ?, ?, ?);
                """, data_to_insert)

                conn.commit()
                print(f"DataFrame successfully imported into {database_name}.{table_name}")

            except sqlite3.Error as e:
                print(f"An error occurred: {e}")

            finally:
                if conn:
                    conn.close()
        ###################################################################
        def dataf_to_windowed_df(datass):
            data = []
            n = 3
            for i in range(n ,len(datass)):
                target_date = datass['Datetime'].iloc[i]

                t3 = datass['Close'].iloc[i-n]
                t2 = datass['Close'].iloc[i-(n-1)]
                t1 = datass['Close'].iloc[i-(n-2)]
                target =datass['Close'].iloc[i]
                data.append([target_date,t3,t2,t1,target])

            datapd = pd.DataFrame(data , columns=['target_date','t3','t2','t1','target'])
            return datapd


        # ========================== 1. Preprocess Data ==========================

        # def preprocess_data(df):
        #     """
        #     Prepares the dataset for LSTM by normalizing the features.

        #     Args:
        #         df (pd.DataFrame): DataFrame with columns ['t3', 't2', 't1', 'target'].

        #     Returns:
        #         tuple: (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
        #     """
        #     df = df.copy()
        #     df.drop(columns=['target_date'], inplace=True, errors='ignore')  # Drop date if exists

        #     # Normalize data using MinMaxScaler
        #     scaler = MinMaxScaler(feature_range=(0, 1))
        #     scaled_data = scaler.fit_transform(df)

        #     # Split features (X) and target (y)
        #     X, y = scaled_data[:, :-1], scaled_data[:, -1]  # X = (t3, t2, t1), y = target

        #     # Reshape X for LSTM (samples, time steps, features)
        #     X = X.reshape(X.shape[0], X.shape[1], 1)

        #     # Train, validation, and test split (70% train, 15% validation, 15% test)
        #     split_train = int(0.7 * len(X))
        #     split_val = int(0.85 * len(X))

        #     X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
        #     y_train, y_val, y_test = y[:split_train], y[split_train:split_val], y[split_val:]

        #     return X_train, y_train, X_val, y_val, X_test, y_test, scaler


        # ========================== 2. Build LSTM Model ==========================

        def build_lstm_model():
            """
            Builds and compiles an LSTM model for stock price prediction.

            Returns:
                Sequential: Compiled LSTM model.
            """
            model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(3, 1)),
                Dropout(0.2),
                LSTM(64),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)  # Predicting 1 value (target close)
            ])

            model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0005), metrics=['mae'])
            return model


        # ========================== 3. Train Model ==========================

        def train_model(model, X_train, y_train, X_val, y_val):
            """
            Trains the LSTM model using early stopping.

            Args:
                model (Sequential): LSTM model.
                X_train (ndarray): Training features.
                y_train (ndarray): Training labels.
                X_val (ndarray): Validation features.
                y_val (ndarray): Validation labels.

            Returns:
                Sequential: Trained model.
            """
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=200,
                batch_size=32,
                callbacks=[early_stop]
            )

            # Save the trained model
            model.save("nifty_lstm_model.h5")

            # # Plot training loss
            # plt.figure(figsize=(10, 5))
            # plt.plot(history.history['loss'], label="Train Loss")
            # plt.plot(history.history['val_loss'], label="Validation Loss")
            # plt.legend()
            # plt.title("LSTM Model Training Loss")
            # plt.show()

            return model


        # ========================== 4. Predict Next Day Closing Price ==========================

        def predict_close(model, last_3_closes, scaler):
            """
            Predicts today's closing price based on the last 3 days' closing prices.

            Args:
                model (Sequential): Trained LSTM model.
                last_3_closes (list): List of last 3 closing prices [T-3, T-2, T-1].
                scaler (MinMaxScaler): Scaler used for feature transformation.

            Returns:
                float: Predicted closing price for today.
            """
            input_data = np.array(last_3_closes).reshape(1, 3)  # Reshape for input
            input_data_scaled = scaler.transform(np.hstack((input_data, [[0]])))[:, :-1]  # Normalize features only
            
            input_data_scaled = input_data_scaled.reshape(1, 3, 1)  # Reshape for LSTM
            predicted_scaled = model.predict(input_data_scaled)
            
            # Convert back to original price
            predicted_close = scaler.inverse_transform(np.hstack((input_data, predicted_scaled)))[:, -1][0]
            
            return predicted_close
#################################################################################################
            

        def preprocess_data(df):
            df = df.copy()  # Avoid modifying the original dataframe

            # Drop datetime columns (assuming your datetime column is named 'timestamp')
            if 'timestamp' in df.columns:
                df = df.drop(columns=['timestamp'])

            # Ensure only numeric columns are used
            df = df.select_dtypes(include=[np.number])

            # Initialize the MinMaxScaler
            scaler = MinMaxScaler()

            # Fit and transform the numeric data
            scaled_data = scaler.fit_transform(df)

            # Convert scaled data back to DataFrame
            scaled_df = pd.DataFrame(scaled_data, columns=df.columns)

            # Split data (adjust slicing based on your needs)
            train_size = int(0.7 * len(scaled_df))
            val_size = int(0.15 * len(scaled_df))

            X_train, y_train = scaled_df.iloc[:train_size, :-1], scaled_df.iloc[:train_size, -1]
            X_val, y_val = scaled_df.iloc[train_size:train_size + val_size, :-1], scaled_df.iloc[train_size:train_size + val_size, -1]
            X_test, y_test = scaled_df.iloc[train_size + val_size:, :-1], scaled_df.iloc[train_size + val_size:, -1]

            return X_train, y_train, X_val, y_val, X_test, y_test, scaler

################################################################################################
        access_token = logins(code)
        print(access_token)
        mindata = minute_data(access_token)
        yeardat = year_data(access_token)
        df = concat_ohcv_dataframes(mindata , yeardat)
        dataf = df[['Datetime','Close']]
        df = dataf_to_windowed_df(dataf)
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_data(df)

        # Build and train model
        lstm_model = build_lstm_model()
        lstm_model = train_model(lstm_model, X_train, y_train, X_val, y_val)

        # Load trained model for prediction
        trained_model = load_model("nifty_lstm_model.h5")

        # Example Prediction
        recent_closes = [22397.2, 22470.5, 22497.9]  # Last 3 days' close
        predicted_price = predict_close(trained_model, recent_closes, scaler)

        print(f"Predicted Today's Close: {predicted_price}")
        
        time.sleep(3)  # Simulating a 3-second calculation
        
        # Example result (replace with your actual result calculation)
        if modem == "Lstm_model_1":
            result = f"Model 1 prediction for {code} using {timeframe} timeframe: 125.67 and predected price is {predicted_price} "
        else:
            result = f"Model 2 prediction for {code} using {timeframe} timeframe: 130.42 and predected price is {predicted_price}"
        
        # Return result as JSON
        return JsonResponse({
            'status': 'success',
            'result': result,
            'code': code,
            'model': modem,
            'timeframe': timeframe
        })
        
    # For regular GET requests, just render the form
    return render(request, 'markettai.html')