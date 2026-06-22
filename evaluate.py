import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

def evaluate(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    target_scaler: MinMaxScaler
) -> tuple:
    
    y_pred_scaled = model.predict(X_test, verbose=0)
 
    # Convert both predictions and ground truth back to dollars
    y_pred_dollars = target_scaler.inverse_transform(y_pred_scaled)
    y_true_dollars = target_scaler.inverse_transform(y_test)
 
    # Compute metrics on dollar-scale values
    mae  = mean_absolute_error(y_true_dollars, y_pred_dollars)
    rmse = np.sqrt(mean_squared_error(y_true_dollars, y_pred_dollars))
 
    print("\n" + "=" * 40)
    print("        EVALUATION RESULTS")
    print("=" * 40)
    print(f"  MAE  (avg dollar error) : ${mae:.2f}")
    print(f"  RMSE (penalised error)  : ${rmse:.2f}")
    print("=" * 40 + "\n")
 
    return y_pred_dollars, y_true_dollars