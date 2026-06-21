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

def add_features(df: pd.DataFrame):
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


