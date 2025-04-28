import requests
import pandas as pd
import time
import os

def get_financialdatamodelingprep_data():
    API_KEY = 'Got it here https://site.financialmodelingprep.com/developer/docs/dashboard'

    symbols = ['NVDA', 'TSLA', 'SPY']
    start_date = '2024-12-11'
    end_date = '2025-03-16'

    base_url = 'https://financialmodelingprep.com/api/v3/historical-price-full/'

    all_dfs = []
    for symbol in symbols:
        url = f"{base_url}{symbol}?from={start_date}&to={end_date}&apikey={API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if 'historical' in data:
                df = pd.DataFrame(data['historical'])
                df.to_csv(f"stock_price_related/{symbol}_historical_data.csv", index=False)
                df['symbol'] = symbol
                all_dfs.append(df)
            else:
                raise ValueError()
        else:
            raise ValueError()
    
    all_dfs = pd.concat(all_dfs, ignore_index=True)
    all_dfs.to_csv("stock_price_related/financialmodelingprep/all_historical_data.csv", index=False)

def get_finhub_data():
    API_KEY = '3c21662b208b40f69e4e7a4a13b9d075'  # Replace with your Twelve Data API key

    symbols = ['NVDA', 'TSLA', 'SPY']
    start_date = '2024-12-11'
    end_date = '2025-03-16'
    interval = '1h'

    base_url = 'https://api.twelvedata.com/time_series'

    all_dfs = []
    for symbol in symbols:
        params = {
            'symbol': symbol,
            'interval': interval,
            'start_date': start_date,
            'end_date': end_date,
            'apikey': API_KEY,
            'format': 'JSON'
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'values' in data:
                df = pd.DataFrame(data['values'])
                df['datetime'] = pd.to_datetime(df['datetime'])
                df = df.sort_values('datetime')
                df['symbol'] = symbol
                df.to_csv(f"stock_price_related/{symbol}_hourly_data.csv", index=False)
                all_dfs.append(df)
            else:
                print(f"No data returned for {symbol}. Response: {data}")
        else:
            print(f"Failed to fetch data for {symbol}. Status code: {response.status_code}")
        time.sleep(1) 

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv("stock_price_related/all_hourly_data.csv", index=False)
        print("Data extraction complete. Files saved in 'stock_price_related' directory.")
    else:
        print("No data was fetched for the given symbols and date range.")    

if __name__ == "__main__":
    get_finhub_data()