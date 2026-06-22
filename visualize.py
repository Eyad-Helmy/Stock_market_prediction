import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
 
 
def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ticker: str = "AAPL",
    save_path: str = "predictions.png"
) -> None:
    
    fig, ax = plt.subplots(figsize=(14, 6))
 
    ax.plot(y_true.flatten(), label="Actual Price",    color="#1B7F5E", linewidth=2.0)
    ax.plot(y_pred.flatten(), label="Predicted Price", color="#C94040",
            linewidth=1.5, linestyle="--")
 
    ax.set_title(f"{ticker} — Actual vs Predicted Closing Price (Test Set)",
                 fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Trading Days (test period)", fontsize=11)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] Saved prediction chart → '{save_path}'")

def plot_training_loss(
    history,
    save_path: str = "training_loss.png"
) -> None:
    
    fig, ax = plt.subplots(figsize=(10, 5))
 
    ax.plot(history.history["loss"],     label="Training loss",   color="#2B6CB0", linewidth=2.0)
    ax.plot(history.history["val_loss"], label="Validation loss", color="#C05621",
            linewidth=1.5, linestyle="--")
 
    ax.set_title("Model Loss over Epochs", fontsize=14, fontweight="bold", pad=14)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
 
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[visualize] Saved training loss chart → '{save_path}'")