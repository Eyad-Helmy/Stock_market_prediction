# Stock Prediction Pipeline

This repository contains a simple stock prediction pipeline for Apple (AAPL) using an LSTM model.

## Project Structure

- `main.py` - orchestration pipeline: data loading, feature engineering, splitting, scaling, sequence creation, model building, training, evaluation, and visualization.
- `data_loader.py` - downloads stock data from Yahoo Finance and caches CSV files to `data/`.
- `preprocessor.py` - feature creation, train/test splitting, scaling, and LSTM sequence preparation.
- `model.py` - LSTM model definition and training logic.
- `evaluate.py` - model evaluation, inverse scaling, and metric reporting.
- `visualize.py` - plots predictions and training loss.
- `data/` - cached stock CSV files.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Run the pipeline with:

```bash
python main.py
```

The script will:

1. Download the configured stock data for `AAPL`
2. Engineer features
3. Split and scale the data
4. Build and train an LSTM model
5. Evaluate predictions on the test set
6. Save visual outputs to `outputs/`

## Outputs

- `outputs/aapl_predictions.png` - actual vs predicted closing prices
- `outputs/aapl_training_loss.png` - training and validation loss curves
- `models/aapl_lstm.h5` - trained Keras model

## Notes

The LSTM uses a rolling window of 30 days and predicts the next closing price.
You can update `CONFIG` in `main.py` to change the ticker, date range, or training settings.
