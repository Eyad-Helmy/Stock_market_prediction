import os
import yfinance as yf
import pandas as pd

def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    save_dir: str = "data"
):
    os.makedirs(save_dir, exist_ok=True)

    file_name = f"{ticker}_{start_date}_{end_date}.csv"
    file_path = os.path.join(save_dir, file_name)

    if os.path.exists(file_path):
        print(f"[data_loader] Loading cached data from '{file_path}' ...")
        df = pd.read_csv(file_path, index_col="Date")
        print(f"[data_loader] Loaded {len(df)} rows")
        return df

    print(f"[data_loader] Downloading {ticker} from Yahoo finance ({start_date} -> {end_date}) ...")
    df = yf.download(ticker, start_date, end_date, progress=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    if df.empty:
        raise ValueError(f"No data returned from {ticker} between start and end date selected date")
    
    df.to_csv(file_path)
    return df