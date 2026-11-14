
import numpy as np
import pandas as pd


def analyze_patterns():
    data = np.load("data/raw/pems04.npz")["data"]

    # Dataset contains 5-minute readings
    readings_per_day = 288
    number_of_days = data.shape[0] // readings_per_day

    # Keep complete days only
    data = data[:number_of_days * readings_per_day]

    # Reshape: days, readings per day, sensors, channels
    daily_data = data.reshape(
        number_of_days,
        readings_per_day,
        data.shape[1],
        data.shape[2]
    )

    flow = daily_data[:, :, :, 0]

    print("\n========== TEMPORAL ANALYSIS ==========")

    # Average traffic flow for each day
    daily_average = flow.mean(axis=(1, 2))

    daily_summary = pd.DataFrame({
        "Day_Index": np.arange(1, number_of_days + 1),
        "Average_Flow": daily_average
    })

    print("\nDaily Average Traffic Flow:")
    print(daily_summary.round(2).to_string(index=False))

    print("\nBusiest Day Index:", np.argmax(daily_average) + 1)
    print("Busiest Day Average:", round(daily_average.max(), 2))

    print("\nQuietest Day Index:", np.argmin(daily_average) + 1)
    print("Quietest Day Average:", round(daily_average.min(), 2))

    # Average traffic at each 5-minute position in the day
    time_profile = flow.mean(axis=(0, 2))

    peak_slot = np.argmax(time_profile)
    quiet_slot = np.argmin(time_profile)

    print("\n========== WITHIN-DAY PATTERN ==========")
    print("Total 5-minute intervals per day:", readings_per_day)
    print("Peak interval index:", peak_slot)
    print("Peak average flow:", round(time_profile[peak_slot], 2))
    print("Quiet interval index:", quiet_slot)
    print("Quiet average flow:", round(time_profile[quiet_slot], 2))

    # Average flow for each sensor
    sensor_average = data[:, :, 0].mean(axis=0)

    sensor_summary = pd.DataFrame({
        "Sensor_ID": np.arange(data.shape[1]),
        "Average_Flow": sensor_average
    })

    print("\n========== SENSOR ANALYSIS ==========")
    print("\nTop 5 Sensors by Average Flow:")
    print(
        sensor_summary.nlargest(5, "Average_Flow")
        .round(2).to_string(index=False)
    )

    print("\nBottom 5 Sensors by Average Flow:")
    print(
        sensor_summary.nsmallest(5, "Average_Flow")
        .round(2).to_string(index=False)
    )

    # Save results
    daily_summary.to_csv(
        "reports/daily_flow_summary.csv", index=False
    )

    pd.DataFrame({
        "Interval_Index": np.arange(readings_per_day),
        "Average_Flow": time_profile
    }).to_csv(
        "reports/time_of_day_flow_profile.csv", index=False
    )

    sensor_summary.to_csv(
        "reports/sensor_flow_summary.csv", index=False
    )

    print("\nAnalysis completed successfully!")
    print("Results saved in reports/")


if __name__ == "__main__":
    analyze_patterns()
