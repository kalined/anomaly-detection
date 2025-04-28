import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from anomaly_detection.data import data_preprocessing
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")

else:
    mps_device = torch.device("mps")

print(mps_device)

class MetricsDataset(Dataset):
    def __init__(self, data, N):
        df = data
        self.columns = df.columns.to_list()
        self.values = df.values.astype(np.float32)
        self.seq_len = N
        self.num_samples = len(self.values) - N
        self.num_features = self.values.shape[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        print("Testas")
        print(self.num_features)
        print(self.columns)
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range (0 to {self.num_samples-1})")
        X = self.values[idx : idx + self.seq_len]
        print("X values:")
        print(X)
        y = self.values[idx + self.seq_len]
        print("Y values:")
        print(y)
        # shapes: X (N, num_features), y (num_features,)
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

N = 30  # Number of past days to consider

def main():
    data_path = "data/csv_files/combined_with_labels_and_synthetic.csv"
    df = data_preprocessing.preprocess_data(data_path).drop(columns=['Label'])

    dataset_len = len(df)
    train_end = int(dataset_len * 0.8)
    valid_end = int(dataset_len * 0.9)

    train_df = df.iloc[:train_end]
    valid_df = df.iloc[train_end:valid_end]
    test_df  = df.iloc[valid_end:]

    print(train_df.shape)
    print(valid_df.shape)
    print(test_df.shape)

    train_dataset = MetricsDataset(train_df, N)
    valid_dataset = MetricsDataset(valid_df, N)
    test_dataset = MetricsDataset(test_df, N)

    x, y = train_dataset[0]
    print(x.shape)
    print(y.shape)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    x, y = next(iter(train_loader))
    print(x.shape)
    print(y.shape)
    #lstm_model = MetricsDataset(df, N)
    #print(lstm_model.__len__())
    #print(lstm_model.__getitem__(1))
    #train_data = df[]

    #rf = MetricsDataset(data_path, N)
    #rf.__getitem__(100)
    #print(rf.__len__())
    #rf.model_decision_tree()
    #rf.model_logistic_regression()
    #rf.model_svm()
    #rf.model_random_forest()
    #print(df.tail(15))

if __name__ == '__main__':
    main()