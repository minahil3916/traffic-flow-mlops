
import numpy as np
import pandas as pd
from pathlib import Path


def evaluate_baseline(X, y, flow_mean, flow_std):

    # Predict future flow using the most recent observed flow
    predictions = X[:, -1, :, 0]

    # Convert normalized values back to original flow scale
    predictions = predictions * flow_std + flow_mean
    actual = y * flow_std + flow_mean

    # Calculate evaluation metrics
    mae = np.mean(np.abs(actual - predictions))
    rmse = np.sqrt(np.mean((actual - predictions) ** 2))

    return mae, rmse


def run_baseline():

    # Load generated sequences
    sequences = np.load("data/processed/traffic_sequences.npz")

    X_val = sequences["X_val"]
    y_val = sequences["y_val"]

    X_test = sequences["X_test"]
    y_test = sequences["y_test"]

    # Load original training data to calculate flow statistics
    splits = np.load("data/processed/traffic_splits.npz")
    train_data = splits["train"]

    flow_mean = train_data[:, :, 0].mean()
    flow_std = train_data[:, :, 0].std()

    print("\n========== PERSISTENCE BASELINE ==========")

    # Evaluate on validation data
    val_mae, val_rmse = evaluate_baseline(
        X_val, y_val, flow_mean, flow_std
    )

    # Evaluate on test data
    test_mae, test_rmse = evaluate_baseline(
        X_test, y_test, flow_mean, flow_std
    )

    print("\nValidation Results:")
    print("MAE:", round(val_mae, 4))
    print("RMSE:", round(val_rmse, 4))

    print("\nTesting Results:")
    print("MAE:", round(test_mae, 4))
    print("RMSE:", round(test_rmse, 4))

    # Save evaluation results
    results = pd.DataFrame({
        "Dataset": ["Validation", "Testing"],
        "MAE": [val_mae, test_mae],
        "RMSE": [val_rmse, test_rmse]
    })

    Path("reports").mkdir(exist_ok=True)
    results.to_csv("reports/baseline_results.csv", index=False)

    print("\nResults saved to reports/baseline_results.csv")


if __name__ == "__main__":
    run_baseline()
