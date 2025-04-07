import os
import pandas as pd
import numpy as np
from datetime import timedelta

np.random.seed(17)
os.chdir("./csv_files")

files = [
    "CPU Utilization-data-2025-04-06 13_46_22.csv",
    "Disk I_O-data-as-joinbyfield-2025-04-06 13_49_45.csv",
    "Disk Utilization-data-2025-04-06 13_49_09.csv",
    "Load Average-data-as-joinbyfield-2025-04-06 13_47_48.csv",
    "Memory Utilization-data-2025-04-06 13_48_28.csv",
    "Network I_O-data-as-joinbyfield-2025-04-06 13_50_53.csv",
    "Network Traffic-data-as-joinbyfield-2025-04-06 13_50_17.csv"
]

def merge_csv_files(file_list):
    dataframes = []
    for file in file_list:
        df = pd.read_csv(file)
        dataframes.append(df)
    merged_df = dataframes[0]
    for df in dataframes[1:]:
        merged_df = pd.merge(merged_df, df, on="Time", how="outer")
    merged_df["Time"] = pd.to_datetime(merged_df["Time"])
    if "Label" not in merged_df.columns:
        merged_df["Label"] = "normal"
    else:
        merged_df["Label"].fillna("normal", inplace=True)
    return merged_df

merged_df = merge_csv_files(files)

def set_label_for_interval(df, start_str, end_str, label):
    start = pd.to_datetime(start_str)
    end = pd.to_datetime(end_str)
    mask = (df["Time"] >= start) & (df["Time"] <= end)
    df.loc[mask, "Label"] = label
    return df

# DDoS attacks
merged_df = set_label_for_interval(merged_df, "2025-04-03 20:53:00", "2025-04-03 20:56:00", "ddos")
merged_df = set_label_for_interval(merged_df, "2025-04-03 20:58:00", "2025-04-03 21:04:00", "ddos")
merged_df = set_label_for_interval(merged_df, "2025-04-03 21:16:00", "2025-04-03 21:41:00", "ddos")
merged_df = set_label_for_interval(merged_df, "2025-04-04 15:25:00", "2025-04-04 15:40:00", "ddos")
# CPU load
merged_df = set_label_for_interval(merged_df, "2025-04-04 16:22:00", "2025-04-04 16:24:00", "cpu_load_anomaly")
merged_df = set_label_for_interval(merged_df, "2025-04-04 16:26:00", "2025-04-04 16:31:00", "cpu_load_anomaly")

# Here we generate synthetic data
def generate_synthetic_data(start_date, end_date, freq="5T"):
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(date_range)
    synthetic = pd.DataFrame({
        "Time": date_range,
        "CPU Utilization": np.random.normal(0.05, 0.005, n).clip(0, 1),
        "Memory Utilization": np.random.normal(0.065, 0.005, n).clip(0, 1),
        "Load[5m]": np.random.normal(0.68, 0.1, n).clip(0),
        "Load[1m]": np.random.normal(0.94, 0.1, n).clip(0),
        "Load[15m]": np.random.normal(0.66, 0.1, n).clip(0),
        "Transmit Total (Network I/O)": np.random.normal(62662, 3000, n).clip(0),
        "Receive Total (Network I/O)": np.random.normal(10019, 800, n).clip(0),
        "Read (Disk I/O)": np.random.normal(0, 5, n).clip(0),
        "Write (Disk I/O)": np.random.normal(332794, 10000, n).clip(0),
        "Disk Utilization": np.random.normal(0.022, 0.002, n).clip(0, 1),
        "Receive Errors (Network Traffic)": np.random.poisson(0.1, n),
        "Receive Total (Network Traffic)": np.random.normal(51.6, 2, n),
        "Transmit Errors (Network Traffic)": np.random.poisson(0.1, n),
        "Receive Dropped (Network Traffic)": np.random.poisson(0.05, n),
        "Transmit Dropped (Network Traffic)": np.random.poisson(0.05, n),
        "Transmit Total (Network Traffic)": np.random.normal(52.6, 2, n),
        "Label": "normal"
    })
    return synthetic
# We need more data, thus we generate a synthetic data till the 2025-05-31
start_synth = merged_df["Time"].max() + timedelta(minutes=5)
end_synth = pd.to_datetime("2025-05-31 23:59:59")
synthetic_df = generate_synthetic_data(start_synth, end_synth)

# Inject anomalies
def inject_random_anomalies(df):
    df = df.copy()
    n = len(df)
    
    # DDoS 2%
    ddos_mask = np.random.random(n) < 0.15
    if ddos_mask.any():
        df.loc[ddos_mask, "Transmit Total (Network I/O)"] *= np.random.uniform(5, 10, ddos_mask.sum())
        df.loc[ddos_mask, "Receive Total (Network I/O)"] *= np.random.uniform(5, 10, ddos_mask.sum())
        df.loc[ddos_mask, "Label"] = "ddos"
    
    # CPU 1%
    cpu_mask = np.random.random(n) < 0.12
    if cpu_mask.any():
        df.loc[cpu_mask, "CPU Utilization"] = np.random.uniform(0.9, 1.0, cpu_mask.sum())
        df.loc[cpu_mask, "Label"] = "cpu_load_anomaly"
    
    # User peak 1%
    user_peak_mask = np.random.random(n) < 0.15
    if user_peak_mask.any():
        df.loc[user_peak_mask, "Load[5m]"] *= np.random.uniform(1.5, 2.0, user_peak_mask.sum())
        df.loc[user_peak_mask, "Load[1m]"] *= np.random.uniform(1.5, 2.0, user_peak_mask.sum())
        df.loc[user_peak_mask, "Load[15m]"] *= np.random.uniform(1.5, 2.0, user_peak_mask.sum())
        df.loc[user_peak_mask, "Transmit Total (Network I/O)"] *= np.random.uniform(1.5, 2.0, user_peak_mask.sum())
        df.loc[user_peak_mask, "Label"] = "user_peak"
    
    # Server unavailability 1%
    unavailable_mask = np.random.random(n) < 0.08
    if unavailable_mask.any():
        cols_to_zero = [
            "CPU Utilization", "Memory Utilization", "Load[5m]", "Load[1m]", "Load[15m]",
            "Transmit Total (Network I/O)", "Receive Total (Network I/O)",
            "Read (Disk I/O)", "Write (Disk I/O)"
        ]
        df.loc[unavailable_mask, cols_to_zero] = 0
        df.loc[unavailable_mask, "Label"] = "server_unavailability"
    
    # Memory leak
    num_blocks = 100
    block_size = 10
    for _ in range(num_blocks):
        if n > block_size:
            start_idx = np.random.randint(0, n - block_size)
            leak_indices = range(start_idx, start_idx + block_size)
            df.loc[leak_indices, "Memory Utilization"] = np.linspace(0.80, 0.98, block_size)
            df.loc[leak_indices, "Label"] = "memory_leak"
    
    return df

synthetic_df = inject_random_anomalies(synthetic_df)

final_df = pd.concat([merged_df, synthetic_df], ignore_index=True)
final_df = final_df.sort_values("Time")
final_df = final_df.loc[:, ~final_df.columns.duplicated()]
print(final_df)

final_df.to_csv("combined_with_labels_and_synthetic.csv", index=False)
print("Dataset created. Numer of lines:", len(final_df))
