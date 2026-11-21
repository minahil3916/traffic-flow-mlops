
import numpy as np
import pandas as pd
from pathlib import Path


def prepare_features():
    # Load the original PEMS04 dataset
    data = np.load("data/raw/pems04.npz")["data"]

    total_steps, total_sensors, _ = data.shape

    # PEMS04 does not provide actual timestamps.
    # Create a consistent synthetic timeline at 5-minute intervals.
    timestamps = pd.date_range(
        start="2026-01-01",
        periods=total_steps,
        freq="5min"
    )

    # Create one row for each sensor at each time step.
    df = pd.DataFrame({
        "event_timestamp": np.repeat(timestamps, total_sensors),
        "sensor_id": np.tile(
            np.arange(total_sensors),
            total_steps
        ),
        "flow": data[:, :, 0].reshape(-1).astype(np.float32),
        "occupancy": data[:, :, 1].reshape(-1).astype(np.float32),
        "speed": data[:, :, 2].reshape(-1).astype(np.float32),
    })

    # Save the feature data as Parquet.
    output_dir = Path("feature_store/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "traffic_features.parquet"
    df.to_parquet(output_path, index=False)

    print("Feast feature data prepared successfully!")
    print(f"Total records: {len(df):,}")
    print(f"Total sensors: {total_sensors}")
    print(f"Total time steps: {total_steps:,}")
    print(f"Saved to: {output_path}")
    print("\nFirst five records:")
    print(df.head())


if __name__ == "__main__":
    prepare_features()
