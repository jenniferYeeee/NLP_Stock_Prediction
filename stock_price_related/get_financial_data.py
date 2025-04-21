import requests
import pandas as pd

def main():
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
    all_dfs.to_csv("stock_price_related/all_historical_data.csv", index=False)
    

if __name__ == "__main__":
    main()