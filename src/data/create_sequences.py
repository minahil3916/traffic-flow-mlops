
import numpy as np
from pathlib import Path

def create_sequences(data, lookback=12, horizon=3):
    """
    Convert traffic data into input-output sequences.

    Inputs:
        data: Array of shape (time_steps, sensors, features)
        lookback: Number of historical time steps
        horizon: Number of steps ahead to predict

    Returns:
        X: Historical readings
        y: Future flow values
    """

    X = []
    y = []

    total_steps = len(data)

    for i in range(total_steps - lookback - horizon + 1):

        # Historical readings for all sensors and features
        input_sequence = data[i:i + lookback]

        # Future flow values for all sensors
        target = data[i + lookback + horizon - 1, :, 0]

        X.append(input_sequence)
        y.append(target)

    return np.array(X), np.array(y)


def create_evaluation_sequences(previous_data, current_data, lookback=12, horizon=3):

    # Keep enough historical observations for the first target
    context_size = lookback + horizon - 1

    context = previous_data[-context_size:]

    # Combine historical context with current split
    combined = np.concatenate((context, current_data), axis=0)

    X = []
    y = []

    for i in range(len(current_data)):

        input_sequence = combined[i:i + lookback]

        target = combined[i + lookback + horizon - 1, :, 0]

        X.append(input_sequence)
        y.append(target)

    return np.array(X), np.array(y)


if __name__ == "__main__":

    # Load normalized dataset splits
    dataset = np.load("data/processed/traffic_splits_scaled.npz")

    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    # Generate training sequences
    X_train, y_train = create_sequences(
        train_data,
        lookback=12,
        horizon=3
    )

    # Generate validation sequences using training history
    X_val, y_val = create_evaluation_sequences(
        train_data,
        val_data,
        lookback=12,
        horizon=3
    )

    # Generate testing sequences using validation history
    X_test, y_test = create_evaluation_sequences(
        val_data,
        test_data,
        lookback=12,
        horizon=3
    )

    print("\n========== SEQUENCE GENERATION ==========")

    print("\nTraining:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("\nValidation:")
    print("X_val:", X_val.shape)
    print("y_val:", y_val.shape)

    print("\nTesting:")
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("\n========== SEQUENCE VERIFICATION ==========")

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))
    print("Testing samples:", len(X_test))

    print("\nAll sequences generated successfully!")

        # Save generated sequences
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "traffic_sequences.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test
    )

    print("\nSequences saved successfully!")
    print("File: data/processed/traffic_sequences.npz")