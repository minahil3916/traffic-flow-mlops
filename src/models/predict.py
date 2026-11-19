
import numpy as np
import joblib


def load_model():

    model = joblib.load("artifacts/random_forest.joblib")

    return model


def predict_traffic(model, input_sequence):

    # Expected input: (12, 307, 3)

    # Rearrange data so each sensor becomes one prediction example
    input_data = input_sequence.transpose(1, 0, 2)

    # Flatten 12 historical readings into 36 features
    input_data = input_data.reshape(307, 36)

    # Predict normalized future flow
    predictions = model.predict(input_data)

    return predictions


if __name__ == "__main__":

    # Load trained model
    model = load_model()

    # Load test sequences
    sequences = np.load("data/processed/traffic_sequences.npz")

    # Select the first test sequence
    input_sequence = sequences["X_test"][0]

    # Generate predictions
    predictions = predict_traffic(model, input_sequence)

    # Convert predictions back to original flow scale
    splits = np.load("data/processed/traffic_splits.npz")
    train_data = splits["train"]

    flow_mean = train_data[:, :, 0].mean()
    flow_std = train_data[:, :, 0].std()

    predictions = predictions * flow_std + flow_mean

    print("\n========== TRAFFIC PREDICTIONS ==========")

    print("Number of sensors:", len(predictions))
        # Get actual future flow values
    actual = sequences["y_test"][0]

    actual = actual * flow_std + flow_mean

    print("\nFirst 10 sensor comparisons:")

    for i in range(10):
        print(
            f"Sensor {i}: "
            f"Predicted = {predictions[i]:.2f}, "
            f"Actual = {actual[i]:.2f}"
        )

    print("\nAverage predicted flow:", predictions.mean())
    print("Average actual flow:", actual.mean())

    print("\nPrediction completed successfully!")
