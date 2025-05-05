import pandas as pd


def preprocess_data(data: str) -> pd.DataFrame:

    df = pd.read_csv(data)

    df.fillna(df.mean(numeric_only=True), inplace=True)
    df["Time"] = pd.to_datetime(df["Time"])

    df["Hour"] = df["Time"].dt.hour
    df["Minute"] = df["Time"].dt.minute

    def string_to_integer(string):
        if string == "normal":
            return 0
        if string == "user_peak":
            return 1
        if string == "ddos":
            return 2
        if string == "cpu_load_anomaly":
            return 3
        if string == "server_unavailability":
            return 4
        if string == "memory_leak":
            return 5

    df["Label"] = df["Label"].apply(string_to_integer)

    df = df.drop(columns=["Time"])
    df_columns = df.columns.tolist()
    new_order = df_columns[-2:] + df_columns[:-2]
    df = df[new_order]

    return df
