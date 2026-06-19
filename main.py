# main orchastrator

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
    #TODO download stock data from yahoo finance api
    #TODO engineer new features (the ones in CONFIG['feature_cols'] except for close since it already exists) and preprocessiing
    #TODO scaling data and then spliting
    #TODO create 3D sequences (LSTM-compatable format)
    #TODO build model (window size, number of features)
    #TODO train model
    #TODO evaluate
    #TODO visualize
    pass