# Pipeline order:
    # 1. add_features()     — engineer new columns
    # 2. split_data()        — split rows chronologically BEFORE scaling
    # 3. scale_data()         — fit scaler on train only, transform both
    # 4. create_sequences()   — build windows separately per split

# Why do we need these extra columns?
    # ====================================
    # The raw Close price alone tells the model "the price today."
    # It does NOT tell the model:
    #     - Is the price trending up or down over the last 2 weeks? (MA_10)
    #     - Is there a longer-term trend behind that?              (MA_50)
    #     - How much did the price MOVE today vs yesterday?        (Daily_Return)
    #     - How erratic has the price been lately?                 (Volatility)
    #     - What were the exact prices 1, 2, 3 days ago?          (Lag features)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds features and cleans data from null values resulted from the feature engineering
    """
    df = df.copy()

    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(window=10).std()
    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)
    df["Lag_3"] = df["Close"].shift(3)

    before_cleaning = len(df)
    df.dropna(inplace=True)
    after_cleaning = len(df)
    print(f"[preprocessor] Added features, {before_cleaning} rows -> {after_cleaning} rows after dropping NaNs.")

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2) -> tuple:
    split_index = int(len(df) * (1 - test_size))   # for dataset of size 10,split_index = 10 * 0.8 = 8
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df

def scale_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    target_col: str
) -> tuple:
    
    feature_scaler = MinMaxScaler(feature_range=(0,1))
    target_scaler = MinMaxScaler(feature_range=(0,1))   # seperate because we will be only reversing this to show output

    # fit the scaling formula only to the training data not on all to avoid leaking test information
    X_train = feature_scaler.fit_transform(train_df[feature_cols].values)
    y_train = target_scaler.fit_transform(train_df[[target_col]].values)

    # transform the test data using the same training scaling formulas so the test set remains unseen
    X_test = feature_scaler.transform(test_df[feature_cols].values)
    y_test = target_scaler.transform(test_df[[target_col]].values)

    print(f"[preprocessor] scale_data: X_train = {X_train.shape}, X_test = {X_test.shape}")

    return X_train, y_train, X_test, y_test, feature_scaler, target_scaler

def create_sequences(X: pd.DataFrame, y: pd.DataFrame, window_size: int = 30):
    X_seq, y_seq = [], []

    for i in range(window_size, len(X)):
        X_seq.append(X[i - window_size : i])
        y_seq.append(y[i])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    print(f"[preprocessor] create_sequences: X_seq = {X_seq.shape}, y_seq = {y_seq.shape}")

    return X_seq, y_seq