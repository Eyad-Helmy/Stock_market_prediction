# main orchastrator

from data_loader import download_stock_data
from preprocessor import add_features, split_data, scale_data, create_sequences 

CONFIG = {
    "ticker": "AAPL",
    "start_date": "2018-01-01",
    "end_date": "2024-01-01",
    "window_size": 30,
    "batch_size": 32,
    "num_epochs": 50,
    "feature_cols": ["Close", "MA_10", "MA_50", "Daily_Return", "Volatility", "Lag_1", "Lag_2"],
    "target_col": "Close",
    "test_size": 0.2
}

def main():
    print("\n" + "=" * 50)
    print("   STOCK MARKET PREDICTION — AAPL LSTM")
    print("=" * 50 + "\n")

    print("STEP 1: LOADING DATA:")
    df_raw = download_stock_data(CONFIG['ticker'], CONFIG['start_date'], CONFIG['end_date'])    # cache dir is automatically set to "data"
    print(f"        Raw data shape: {df_raw.shape}\n")
    
    #TODO engineer new features (the ones in CONFIG['feature_cols'] except for close since it already exists) and preprocessiing /DONE
    print("STEP 2: FEATURE ENGINEERING:")
    df = add_features(df_raw)
    print(f"        Enriched data shape: {df.shape}\n")

    #TODO split rows to training and testing /DONE
    print("STEP 3: Splitting into train/test:")
    train_set, test_set = split_data(df, CONFIG["test_size"])
    print(f"        Train set shape: {train_set.shape}")
    print(f"        Test set shape: {test_set.shape}")

    #TODO scaling data on train set only then transform both sets   /DONE
    print("STEP 4: Scaling data...")
    X_train, y_train, X_test, y_test, feature_scaler, target_scaler = scale_data(
        train_set, test_set, CONFIG["feature_cols"], CONFIG["target_col"]
    )

    #TODO create 3D sequences (LSTM-compatable format) on both sets seperatly  /DONE
    print("STEP 5: Creating sequences...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, window_size=CONFIG["window_size"])
    X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  window_size=CONFIG["window_size"])

    #TODO build model (window size, number of features)     /tobeunderstood
    #TODO train model   /tobeunderstood
    #TODO evaluate
    #TODO visualize 
    pass

if __name__ == "__main__":
    main()