
import numpy as np
import pandas as pd


def inspect_dataset():
    # Load the original dataset
    file_path = "data/raw/pems04.npz"
    dataset = np.load(file_path)

    data = dataset["data"]

    print("\n========== DATASET OVERVIEW ==========")
    print("Shape:", data.shape)
    print("Data type:", data.dtype)
    print("Total values:", data.size)

    # Measurement names
    channels = ["Flow", "Occupancy", "Speed"]

    print("\n========== DATA QUALITY ==========")

    for i, channel in enumerate(channels):
        values = data[:, :, i]

        print(f"\n--- {channel} ---")

        print("Minimum:", np.min(values))
        print("Maximum:", np.max(values))
        print("Mean:", np.mean(values))
        print("Standard deviation:", np.std(values))

        print("Missing values:", np.isnan(values).sum())
        print("Infinite values:", np.isinf(values).sum())
        print("Zero values:", np.sum(values == 0))
        print("Negative values:", np.sum(values < 0))

    # Check each sensor
    print("\n========== SENSOR INSPECTION ==========")

    sensor_means = np.mean(data[:, :, 0], axis=0)

    print("Number of sensors:", data.shape[1])
    print("Average flow across sensors:", np.mean(sensor_means))
    print("Lowest sensor average flow:", np.min(sensor_means))
    print("Highest sensor average flow:", np.max(sensor_means))

    print("\nInspection completed successfully!")


if __name__ == "__main__":
    inspect_dataset()
