
import numpy as np
from pathlib import Path


def split_dataset():
    # Load original dataset
    data = np.load("data/raw/pems04.npz")["data"]

    print("\n========== ORIGINAL DATASET ==========")
    print("Dataset shape:", data.shape)

    total_steps = data.shape[0]

    # Chronological split: 70%, 15%, 15%
    train_end = int(total_steps * 0.70)
    val_end = int(total_steps * 0.85)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    print("\n========== DATA SPLIT ==========")
    print("Training shape:", train_data.shape)
    print("Validation shape:", val_data.shape)
    print("Testing shape:", test_data.shape)

    print("\n========== SPLIT INDICES ==========")
    print(f"Training:   0 to {train_end - 1}")
    print(f"Validation: {train_end} to {val_end - 1}")
    print(f"Testing:    {val_end} to {total_steps - 1}")

    # Verify that all observations are included
    assert len(train_data) + len(val_data) + len(test_data) == total_steps

    # Verify chronological boundaries
    assert train_end < val_end < total_steps

    # Save processed splits
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "traffic_splits.npz",
        train=train_data,
        validation=val_data,
        test=test_data
    )

    print("\nDataset splitting completed successfully!")
    print("Saved to data/processed/traffic_splits.npz")


if __name__ == "__main__":
    split_dataset()
