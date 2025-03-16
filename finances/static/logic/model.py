import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import datasql as ds
import numpy as np
apikey = '12780cc6-a7e4-466c-9488-3de58d39ada5'
secretkey = 'ute2ha7ln6'
uris = 'https://api.upstox.com/v2/login'
uri = f'https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={apikey}&redirect_uri={uris}'
print(uri)

access_token = ds.logins()
mindata = ds.minute_data(access_token)
yeardat = ds.year_data(access_token)
df = ds.concat_ohcv_dataframes(mindata , yeardat)
dataf = df[['Datetime','Close']]




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

def preprocess_data(df):
    """
    Prepares the dataset for LSTM by normalizing the features.

    Args:
        df (pd.DataFrame): DataFrame with columns ['t3', 't2', 't1', 'target'].

    Returns:
        tuple: (X_train, y_train, X_val, y_val, X_test, y_test, scaler)
    """
    df = df.copy()
    df.drop(columns=['target_date'], inplace=True, errors='ignore')  # Drop date if exists

    # Normalize data using MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Split features (X) and target (y)
    X, y = scaled_data[:, :-1], scaled_data[:, -1]  # X = (t3, t2, t1), y = target

    # Reshape X for LSTM (samples, time steps, features)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train, validation, and test split (70% train, 15% validation, 15% test)
    split_train = int(0.7 * len(X))
    split_val = int(0.85 * len(X))

    X_train, X_val, X_test = X[:split_train], X[split_train:split_val], X[split_val:]
    y_train, y_val, y_test = y[:split_train], y[split_train:split_val], y[split_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler


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

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.legend()
    plt.title("LSTM Model Training Loss")
    plt.show()

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


# ========================== 5. Run Pipeline ==========================
df = dataf_to_windowed_df(dataf)
# Example DataFrame
# df = pd.DataFrame({
#     'target_date': ['2025-03-10', '2025-03-07', '2025-03-06', '2025-03-05'],
#     't3': [22397.2, 22470.5, 22497.9, 22460.3],
#     't2': [22470.5, 22497.9, 22460.3, 22552.5],
#     't1': [22497.9, 22460.3, 22552.5, 22544.7],
#     'target': [22460.3, 22552.5, 22544.7, 22337.3]
# })

# Preprocess Data
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
