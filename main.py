# main orchastrator

from data_loader import download_stock_data

CONFIG = {
    "ticker": "AAPL",
    "start_date": "2018-01-01",
    "end_date": "2024-01-01",
    "window_size": 30,
    "batch_size": 32,
    "num_epochs": 50,
    "feature_cols": ["Close", "MA_10", "MA_50", "Daily_Return", "Volatility", "Lag_1", "Lag_2"],
    "target_col": "Close",
}

def main():
    print("\n" + "=" * 50)
    print("   STOCK MARKET PREDICTION — AAPL LSTM")
    print("=" * 50 + "\n")

    print("STEP 1: LOADING DATA:")
    df_raw = download_stock_data(CONFIG['ticker'], CONFIG['start_date'], CONFIG['end_date'])    # cache dir is automatically set to "data"
    print(f"Raw data shape: {df_raw.shape}")
    
    #TODO engineer new features (the ones in CONFIG['feature_cols'] except for close since it already exists) and preprocessiing /DONE
    #TODO split rows to training and testing /DONE
    #TODO scaling data on train set only then transform both sets   /DONE
    #TODO create 3D sequences (LSTM-compatable format) on both sets seperatly  /DONE
    #TODO build model (window size, number of features)
    #TODO train model
    #TODO evaluate
    #TODO visualize
    pass

if __name__ == "__main__":
    main()