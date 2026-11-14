
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def explore_dataset():

    # Load original dataset
    dataset = np.load("data/raw/pems04.npz")
    data = dataset["data"]

    channels = ["Flow", "Occupancy", "Speed"]

    # Create output directory
    os.makedirs("reports/figures", exist_ok=True)

    print("========== EDA SUMMARY ==========")

    # Convert all readings into a DataFrame
    flattened_data = data.reshape(-1, 3)

    df = pd.DataFrame(flattened_data, columns=channels)

    # Descriptive statistics
    print("\nDescriptive Statistics:")
    print(df.describe().round(3))

    # Zero percentages
    print("\nZero Reading Percentages:")
    for i, channel in enumerate(channels):
        zero_percentage = np.mean(data[:, :, i] == 0) * 100
        print(f"{channel}: {zero_percentage:.2f}%")

    # Correlation
    correlation = df.corr()

    print("\nCorrelation Matrix:")
    print(correlation.round(3))

    # 1. Traffic flow over time for sensor 0
    sensor_id = 0

    # First seven days (288 readings per day)
    readings = 7 * 288

    flow = data[:readings, sensor_id, 0]

    plt.figure(figsize=(12, 5))
    plt.plot(flow)
    plt.title("Traffic Flow Over Time - Sensor 0")
    plt.xlabel("5-minute intervals since dataset start")
    plt.ylabel("Traffic Flow")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("reports/figures/traffic_flow_over_time.png")
    plt.close()

    # 2. Distribution of traffic flow
    # Sample readings to keep plotting efficient
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        len(df),
        size=min(200000, len(df)),
        replace=False
    )

    flow_sample = df.iloc[sample_indices]["Flow"]

    plt.figure(figsize=(10, 5))
    plt.hist(flow_sample, bins=50, edgecolor="black")
    plt.title("Distribution of Traffic Flow")
    plt.xlabel("Traffic Flow")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("reports/figures/traffic_flow_distribution.png")
    plt.close()

    # 3. Average traffic flow per sensor
    sensor_means = np.mean(data[:, :, 0], axis=0)

    plt.figure(figsize=(12, 5))
    plt.bar(range(data.shape[1]), sensor_means)
    plt.title("Average Traffic Flow Across Sensors")
    plt.xlabel("Sensor ID")
    plt.ylabel("Average Traffic Flow")
    plt.tight_layout()
    plt.savefig("reports/figures/sensor_average_flow.png")
    plt.close()

    # 4. Correlation heatmap
    plt.figure(figsize=(7, 5))
    plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)

    plt.xticks(range(3), channels)
    plt.yticks(range(3), channels)

    for i in range(3):
        for j in range(3):
            plt.text(
                j, i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center"
            )

    plt.colorbar(label="Correlation")
    plt.title("Correlation Between Traffic Features")
    plt.tight_layout()
    plt.savefig("reports/figures/feature_correlation.png")
    plt.close()

    # Save descriptive statistics
    df.describe().to_csv("reports/eda_statistics.csv")

    print("\nEDA completed successfully!")
    print("Charts saved in reports/figures/")
    print("Statistics saved in reports/eda_statistics.csv")


if __name__ == "__main__":
    explore_dataset()
