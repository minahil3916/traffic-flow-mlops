import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

def prepare_training_data(X, y, sample_size=150000):

    # Treat each sensor at each time window as one training example
    total_examples = X.shape[0] * X.shape[2]

    # Randomly select examples with a fixed seed
    rng = np.random.default_rng(42)
    selected = rng.choice(
        total_examples,
        size=min(sample_size, total_examples),
        replace=False
    )

    # Convert flattened indices into sequence and sensor indices
    sequence_indices = selected // X.shape[2]
    sensor_indices = selected % X.shape[2]

    # Extract historical readings for selected sensors
    X_sample = X[sequence_indices, :, sensor_indices, :]
    X_sample = X_sample.reshape(len(selected), -1)

    # Extract corresponding future flow targets
    y_sample = y[sequence_indices, sensor_indices]

    return X_sample, y_sample


def evaluate_model(model, X, y, flow_mean, flow_std):

    predictions = []
    actual_values = []

    # Predict in batches to control memory usage
    for start in range(0, len(X), 32):

        end = start + 32

        batch_X = X[start:end]

        batch_X = batch_X.transpose(0, 2, 1, 3)
        batch_X = batch_X.reshape(-1, 36)

        batch_predictions = model.predict(batch_X)

        predictions.extend(batch_predictions)

        actual_values.extend(y[start:end].reshape(-1))

    predictions = np.array(predictions)
    actual_values = np.array(actual_values)

    # Convert normalized flow values to original scale
    predictions = predictions * flow_std + flow_mean
    actual_values = actual_values * flow_std + flow_mean

    mae = mean_absolute_error(actual_values, predictions)
    rmse = np.sqrt(mean_squared_error(actual_values, predictions))

    return mae, rmse



def run_random_forest():

    # Configure MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Traffic Flow Forecasting")

    # Load generated sequences
    sequences = np.load("data/processed/traffic_sequences.npz")

    X_train = sequences["X_train"]
    y_train = sequences["y_train"]

    X_val = sequences["X_val"]
    y_val = sequences["y_val"]

    X_test = sequences["X_test"]
    y_test = sequences["y_test"]

    # Prepare training samples
    X_sample, y_sample = prepare_training_data(X_train, y_train)

    print("\nTraining samples:", X_sample.shape)
    print("Target samples:", y_sample.shape)

    # Define Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    print("\n========== RANDOM FOREST TRAINING ==========")

    # Start MLflow experiment run
    with mlflow.start_run(run_name="Random_Forest_Baseline"):

        # Train model
        model.fit(X_sample, y_sample)

        print("Model training completed!")

        # Log model parameters
        mlflow.log_params({
            "model": "RandomForestRegressor",
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_leaf": 5,
            "sample_size": 150000,
            "lookback": 12,
            "forecast_horizon": 3
        })

        # Load original training data for inverse normalization
        splits = np.load("data/processed/traffic_splits.npz")
        train_data = splits["train"]

        flow_mean = train_data[:, :, 0].mean()
        flow_std = train_data[:, :, 0].std()

        # Evaluate on validation data
        val_mae, val_rmse = evaluate_model(
            model, X_val, y_val, flow_mean, flow_std
        )

        # Evaluate on test data
        test_mae, test_rmse = evaluate_model(
            model, X_test, y_test, flow_mean, flow_std
        )

        print("\n========== RANDOM FOREST RESULTS ==========")

        print("\nValidation:")
        print("MAE:", round(val_mae, 4))
        print("RMSE:", round(val_rmse, 4))

        print("\nTesting:")
        print("MAE:", round(test_mae, 4))
        print("RMSE:", round(test_rmse, 4))

        # Log evaluation metrics in MLflow
        mlflow.log_metrics({
            "validation_mae": val_mae,
            "validation_rmse": val_rmse,
            "test_mae": test_mae,
            "test_rmse": test_rmse
        })

        # Save trained model locally
        model_dir = Path("artifacts")
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, model_dir / "random_forest.joblib")

        print("\nModel saved to artifacts/random_forest.joblib")

        # Log model in MLflow
        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest",
            skops_trusted_types=["sklearn.tree._tree.Tree"]
        )

        # Save evaluation results
        results = pd.DataFrame({
            "Dataset": ["Validation", "Testing"],
            "MAE": [val_mae, test_mae],
            "RMSE": [val_rmse, test_rmse]
        })

        Path("reports").mkdir(exist_ok=True)

        results.to_csv(
            "reports/random_forest_results.csv",
            index=False
        )

        # Save results as an MLflow artifact
        mlflow.log_artifact("reports/random_forest_results.csv")

        print("\nResults saved to reports/random_forest_results.csv")
        print("Experiment logged successfully in MLflow!")


if __name__ == "__main__":
    run_random_forest()
