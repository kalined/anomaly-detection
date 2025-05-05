import os
from datetime import timedelta

import numpy as np
import pandas as pd

np.random.seed(17)

os.chdir("./csv_files")

files = [
    "CPU Utilization-data-2025-04-06 13_46_22.csv",
    "Disk I_O-data-as-joinbyfield-2025-04-06 13_49_45.csv",
    "Disk Utilization-data-2025-04-06 13_49_09.csv",
    "Load Average-data-as-joinbyfield-2025-04-06 13_47_48.csv",
    "Memory Utilization-data-2025-04-06 13_48_28.csv",
    "Network I_O-data-as-joinbyfield-2025-04-06 13_50_53.csv",
    "Network Traffic-data-as-joinbyfield-2025-04-06 13_50_17.csv",
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


# sujungiame duoemnis
merged_df = merge_csv_files(files)


def set_label_for_interval(df, start_str, end_str, label):
    start = pd.to_datetime(start_str)
    end = pd.to_datetime(end_str)
    mask = (df["Time"] >= start) & (df["Time"] <= end)
    df.loc[mask, "Label"] = label
    return df


merged_df = set_label_for_interval(
    merged_df, "2025-04-03 20:53:00", "2025-04-03 20:56:00", "ddos"
)
merged_df = set_label_for_interval(
    merged_df, "2025-04-03 20:58:00", "2025-04-03 21:04:00", "ddos"
)
merged_df = set_label_for_interval(
    merged_df, "2025-04-03 21:16:00", "2025-04-03 21:41:00", "ddos"
)
merged_df = set_label_for_interval(
    merged_df, "2025-04-04 15:25:00", "2025-04-04 15:40:00", "ddos"
)
merged_df = set_label_for_interval(
    merged_df, "2025-04-04 16:22:00", "2025-04-04 16:24:00", "cpu_load_anomaly"
)
merged_df = set_label_for_interval(
    merged_df, "2025-04-04 16:26:00", "2025-04-04 16:31:00", "cpu_load_anomaly"
)


# Generuoja retus, labai didelius "Read (Disk I/O)" outlier'us
def generate_read_disk(n):
    # maza dispersija (kad duomenys butu aciau vidurkio, kuris yra labai mazas)
    base_values = np.random.normal(0, 5, n).clip(0)
    #  1% pakeiciame i dideles reiksmes
    num_outliers = max(1, int(0.01 * n))
    outlier_indices = np.random.choice(n, num_outliers, replace=False)
    base_values[outlier_indices] = np.random.uniform(1e7, 1.5e7, num_outliers)
    return base_values


def generate_synthetic_data_stable(start_date, end_date, freq="5T"):
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    n = len(date_range)
    hours = date_range.hour

    # Aukstesni vidurkiai darbo laiku ir zemesni ne darbo laiku
    cpu_mean = np.where(
        (hours >= 8) & (hours < 19), 0.40, 0.04
    )  # darbo laikas: 10%, ne darbo: 4%
    mem_mean = np.where(
        (hours >= 8) & (hours < 19), 0.50, 0.06
    )  # RAM: 10% darbo, 6% ne darbo
    load_mean = np.where(
        (hours >= 8) & (hours < 19), 5.0, 0.6
    )  # sis apkrova: aukstesne darbo metu
    write_mean = np.where((hours >= 8) & (hours < 19), 500000, 410000)
    # disk_utilization_mean = np.where((hours >= 8) & (hours < 19), 0.030, 0.0020)
    net_transmit_mean = np.where((hours >= 8) & (hours < 19), 70000, 55000)
    net_receive_mean = np.where((hours >= 8) & (hours < 19), 11000, 8500)

    synthetic = pd.DataFrame(
        {
            "Time": date_range,
            "CPU Utilization": np.random.normal(cpu_mean, 0.005, n).clip(0, 1),
            "Memory Utilization": np.random.normal(mem_mean, 0.005, n).clip(0, 1),
            "Load[5m]": np.random.normal(load_mean, 0.1, n).clip(0),
            "Load[1m]": np.random.normal(load_mean * 1.1, 0.1, n).clip(0),
            "Load[15m]": np.random.normal(load_mean * 0.95, 0.1, n).clip(0),
            "Write (Disk I/O)": np.random.normal(write_mean, 10000, n).clip(0),
            # "Disk Utilization":
            # np.random.normal(disk_utilization_mean, 0.025, n).clip(0, 1),
            "Disk Utilization": np.random.normal(0.035, 0.025, n).clip(0, 1).round(4),
            "Transmit Total (Network I/O)": np.random.normal(
                net_transmit_mean, 3000, n
            ).clip(0),
            "Receive Total (Network I/O)": np.random.normal(
                net_receive_mean, 800, n
            ).clip(0),
            "Read (Disk I/O)": generate_read_disk(n),
            "Receive Errors (Network Traffic)": np.random.poisson(0.1, n).clip(0),
            "Receive Total (Network Traffic)": np.random.normal(67.5, 57.5, n).clip(0),
            "Transmit Errors (Network Traffic)": np.random.poisson(0.1, n).clip(0),
            "Receive Dropped (Network Traffic)": np.random.poisson(0.05, n).clip(0),
            "Transmit Dropped (Network Traffic)": np.random.poisson(0.05, n).clip(0),
            "Transmit Total (Network Traffic)": np.random.normal(70, 55, n).clip(0),
            "Label": "normal",
        }
    )
    return synthetic


# Sintetiniai duomenys: nuo pradzios 5 min iki 2025-05-31 23:59:59
start_synth = merged_df["Time"].max() + timedelta(minutes=5)
end_synth = pd.to_datetime("2025-05-31 23:59:59")
synthetic_df = generate_synthetic_data_stable(start_synth, end_synth, freq="5T")


# Funkcija iterpti papildomas anomalijas sintetineje dalyje
def inject_random_anomalies(df):
    df = df.copy()
    n = len(df)

    # DDoS anomalija: padidinti tinklo I/O rodiklius
    ddos_mask = np.random.random(n) < 0.15
    if ddos_mask.any():
        df.loc[ddos_mask, "Transmit Total (Network I/O)"] *= np.random.uniform(
            5, 10, ddos_mask.sum()
        )
        df.loc[ddos_mask, "Receive Total (Network I/O)"] *= np.random.uniform(
            5, 10, ddos_mask.sum()
        )
        df.loc[ddos_mask, "Label"] = "ddos"

    # CPU apkrovos anomalija
    cpu_mask = np.random.random(n) < 0.12
    if cpu_mask.any():
        df.loc[cpu_mask, "CPU Utilization"] = np.random.uniform(
            0.9, 1.0, cpu_mask.sum()
        )
        df.loc[cpu_mask, "Label"] = "cpu_load_anomaly"

    # User peak anomalijos
    user_peak_mask = np.random.random(n) < 0.15
    if user_peak_mask.any():
        df.loc[user_peak_mask, "Load[5m]"] *= np.random.uniform(
            1.5, 2.0, user_peak_mask.sum()
        )
        df.loc[user_peak_mask, "Load[1m]"] *= np.random.uniform(
            1.5, 2.0, user_peak_mask.sum()
        )
        df.loc[user_peak_mask, "Load[15m]"] *= np.random.uniform(
            1.5, 2.0, user_peak_mask.sum()
        )
        df.loc[user_peak_mask, "Transmit Total (Network I/O)"] *= np.random.uniform(
            1.5, 2.0, user_peak_mask.sum()
        )
        df.loc[user_peak_mask, "Label"] = "user_peak"

    # Server unavailability anomalija
    unavailable_mask = np.random.random(n) < 0.08
    if unavailable_mask.any():
        cols_to_zero = [
            "CPU Utilization",
            "Memory Utilization",
            "Load[5m]",
            "Load[1m]",
            "Load[15m]",
            "Transmit Total (Network I/O)",
            "Receive Total (Network I/O)",
            "Read (Disk I/O)",
            "Write (Disk I/O)",
        ]
        df.loc[unavailable_mask, cols_to_zero] = 0
        df.loc[unavailable_mask, "Label"] = "server_unavailability"

    # Memory leak anomalija
    num_blocks = 100
    block_size = 10
    for _ in range(num_blocks):
        if n > block_size:
            start_idx = np.random.randint(0, n - block_size)
            leak_indices = range(start_idx, start_idx + block_size)
            df.loc[leak_indices, "Memory Utilization"] = np.linspace(
                0.80, 0.98, block_size
            )
            df.loc[leak_indices, "Label"] = "memory_leak"

    return df


synthetic_df = inject_random_anomalies(synthetic_df)

final_df = pd.concat([merged_df, synthetic_df], ignore_index=True)
final_df = final_df.sort_values("Time")
final_df = final_df.loc[:, ~final_df.columns.duplicated()]

rounding_dict = {
    "CPU Utilization": 4,
    "Memory Utilization": 4,
    "Load[5m]": 2,
    "Load[1m]": 2,
    "Load[15m]": 2,
    "Write (Disk I/O)": 0,
    "Disk Utilization": 4,
    "Transmit Total (Network I/O)": 0,
    "Receive Total (Network I/O)": 0,
    "Read (Disk I/O)": 0,
    "Receive Errors (Network Traffic)": 0,
    "Receive Total (Network Traffic)": 1,
    "Transmit Errors (Network Traffic)": 0,
    "Receive Dropped (Network Traffic)": 0,
    "Transmit Dropped (Network Traffic)": 0,
    "Transmit Total (Network Traffic)": 1,
}

for col, decimals in rounding_dict.items():
    if col in final_df.columns:
        final_df[col] = final_df[col].round(decimals)

final_df.to_csv("combined_with_labels_and_synthetic5.csv", index=False)
print("Dataset created. Number of lines:", len(final_df))
