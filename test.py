# for testing stuff, idky i uploaded this
import yfinance as yf
import pandas as pd
from preprocessor import add_features, split_data, scale_data, create_sequences

pd.set_option("display.max_columns", None)  #display all columns

df = yf.download("AAPL", "2018-01-01", "2019-01-30")
df = pd.DataFrame(df)
df.columns = df.columns.get_level_values(0)
# print(isinstance(df.columns, pd.MultiIndex))

df = add_features(df)
train, test = split_data(df)

feature_cols = [
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    "MA_10",
    "MA_50",
    "Daily_Return",
    "Volatility",
    "Lag_1",
    "Lag_2",
    "Lag_3",
]
target_col = "Close"

X_train, y_train, X_test, y_test, feature_scaler, target_scaler = scale_data(train, test, feature_cols, target_col)

print("TRAIN / TEST SUMMARY")
print("------------------")
print(f"Train shape: {train.shape}")
print(f"Test shape:  {test.shape}\n")
print("Train sample:")
print(train.head(5).to_string(index=False))
print("\nTest sample:")
print(test.head(5).to_string(index=False))
print("\nSCALED DATA SUMMARY")
print("-------------------")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_test shape:  {y_test.shape}\n")
print("First 5 scaled X_train rows:")
print(pd.DataFrame(X_train, columns=feature_cols).head().to_string(index=False))
print("\nFirst 5 scaled y_train values:")
print(pd.DataFrame(y_train, columns=[target_col]).head().to_string(index=False))

# create sequences from scaled data
window_size = 30
X_seq, y_seq = create_sequences(X_train, y_train, window_size=window_size)

print("\nSEQUENCES SUMMARY")
print("-----------------")
print(f"window_size: {window_size}")
print(f"X_seq shape: {X_seq.shape}")
print(f"y_seq shape: {y_seq.shape}\n")
if X_seq.shape[0] > 0:
    print("First X_seq sample (shape):", X_seq[0].shape)
    print("First y_seq sample:", y_seq[0])