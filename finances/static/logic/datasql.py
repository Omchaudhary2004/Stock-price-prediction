import urllib.parse
import pandas as pd
import requests
import requests
from dateutil.relativedelta import relativedelta
from datetime import datetime
from datetime import date
import sqlite3


apikey = '12780cc6-a7e4-466c-9488-3de58d39ada5'
secretkey = 'ute2ha7ln6'
uris = 'https://api.upstox.com/v2/login'
uri = f'https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={apikey}&redirect_uri={uris}'
print(uri)

def logins(code):
    apikey = '12780cc6-a7e4-466c-9488-3de58d39ada5'
    secretkey = 'ute2ha7ln6'
    
    uris = 'https://api.upstox.com/v2/login'
    # uri = f'https://api.upstox.com/v2/login/authorization/dialog?response_type=code&client_id={apikey}&redirect_uri={uris}'
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

# Example usage (assuming you have a DataFrame named 'concatenated_result'):
# import_dataframe_to_sqlite(concatenated_result, 'my_database.db', 'nifty_historic_data')


# access_token = logins()
# minut = minute_data(access_token)
# df_year = year_data(access_token)
# finaldf = concat_ohcv_dataframes(minut , df_year)
# import_dataframe_to_sqlite(finaldf , "stock_datass.db" , 'nifty_ohcv')