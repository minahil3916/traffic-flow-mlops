
import numpy as np
import pandas as pd
from pathlib import Path


def normalize_dataset():
    dataset = np.load("data/processed/traffic_splits.npz")

    # Use float64 for accurate statistics
    train = dataset["train"].astype(np.float64)
    validation = dataset["validation"].astype(np.float64)
    test = dataset["test"].astype(np.float64)

    print("\n========== NORMALIZATION ==========")

    # Calculate statistics using training data only
    mean = train.mean(axis=(0, 1), keepdims=True)
    std = train.std(axis=(0, 1), keepdims=True)

    std = np.where(std == 0, 1.0, std)

    print("Training means:", mean.flatten())
    print("Training standard deviations:", std.flatten())

    # Apply the same training statistics to every split
    train_scaled = ((train - mean) / std).astype(np.float32)
    validation_scaled = ((validation - mean) / std).astype(np.float32)
    test_scaled = ((test - mean) / std).astype(np.float32)

    print("\n========== NORMALIZATION VERIFICATION ==========")
    print("Training means:", train_scaled.mean(axis=(0, 1), dtype=np.float64))
    print("Training std:", train_scaled.std(axis=(0, 1), dtype=np.float64))

    # Save normalized splits
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "traffic_splits_scaled.npz",
        train=train_scaled,
        validation=validation_scaled,
        test=test_scaled
    )

    # Save statistics for future inference
    stats = pd.DataFrame({
        "Feature": ["Flow", "Occupancy", "Speed"],
        "Training_Mean": mean.flatten(),
        "Training_Std": std.flatten()
    })

    stats.to_csv("reports/normalization_statistics.csv", index=False)

    print("\nNormalization completed successfully!")
    print("Normalized data saved in data/processed/")
    print("Statistics saved in reports/")


if __name__ == "__main__":
    normalize_dataset()
