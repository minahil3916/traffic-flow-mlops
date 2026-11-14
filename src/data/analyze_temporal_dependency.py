
import numpy as np
import pandas as pd


def analyze_temporal_dependency():
    data = np.load("data/raw/pems04.npz")["data"]

    # Extract flow values: time steps x sensors
    flow = data[:, :, 0]

    # Lags represent previous time intervals
    lags = [1, 2, 3, 6, 12, 288]

    results = []

    print("\n========== TEMPORAL DEPENDENCY ANALYSIS ==========")

    for lag in lags:
        sensor_correlations = []

        for sensor in range(flow.shape[1]):
            previous = flow[:-lag, sensor]
            current = flow[lag:, sensor]

            correlation = np.corrcoef(previous, current)[0, 1]

            if np.isfinite(correlation):
                sensor_correlations.append(correlation)

        average_correlation = np.mean(sensor_correlations)

        results.append({
            "Lag": lag,
            "Time_Difference_Minutes": lag * 5,
            "Average_Sensor_Correlation": average_correlation
        })

        print(
            f"Lag {lag:3d} ({lag * 5:4d} minutes): "
            f"{average_correlation:.4f}"
        )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        "reports/temporal_dependency.csv",
        index=False
    )

    print("\nResults saved to reports/temporal_dependency.csv")
    print("\nAnalysis completed successfully!")


if __name__ == "__main__":
    analyze_temporal_dependency()
