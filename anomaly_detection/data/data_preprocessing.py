import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import autocorrelation_plot

def preprocess_data(data: str) -> pd.DataFrame:

    df = pd.read_csv(data)

    #print("Dataframe information: \n",df.info())
    #print("Check null values: \n", df.isnull().sum())
    #print("Count null values: \n", df.isnull().sum().sum())

    df1 = df[df.isna().any(axis=1)]
    #print("View rows with null values: ", df1)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    #print("Count null values after fillna function: \n", df.isnull().sum().sum())
    df['Time'] = pd.to_datetime(df['Time'])
    #print("Dataframe information: \n",df.info())

    df['Hour'] = df['Time'].dt.hour
    df['Minute'] = df['Time'].dt.minute

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


    #print(df.tail(15))

    #print("Data shape:", df.shape)
    #print("Dataframe information: \n",df.info())

    return df