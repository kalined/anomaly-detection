import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from anomaly_detection.data import data_preprocessing

# Check that MPS is available
if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print(
            "MPS not available because the current PyTorch"
            " install was not built with MPS enabled."
        )
    else:
        print(
            "MPS not available because the current MacOS version is not 12.3+ "
            "and/or you do not have an MPS-enabled device on this machine."
        )
else:
    mps_device = torch.device("mps")
print(f"Using device: {mps_device}")


# Dataset wrapper for multivariate time series
class MetricsDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_len: int):
        self.values = df.values.astype(np.float32)
        self.seq_len = seq_len
        self.num_samples = len(self.values) - seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        x = self.values[idx : idx + self.seq_len]
        y = self.values[idx + self.seq_len]
        return torch.tensor(x), torch.tensor(y)


class LSTMModel(nn.Module):
    def __init__(self, input_size=18, hidden_size=128, output_size=18, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, dropout=0.2, batch_first=True
        )
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(
            x.device
        )
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(
            x.device
        )
        out, _ = self.lstm(x, (h0, c0))
        return self.linear(out[:, -1, :])


# Plot function: one subplot per metric
def plot_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, names: list[str]):
    n = y_true.shape[1]
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True)
    axes = axes.flatten()
    for i, name in enumerate(names):
        ax = axes[i]
        ax.plot(y_true[:, i], label="Actual", alpha=0.7)
        ax.plot(y_pred[:, i], linestyle="--", label="Predicted", alpha=0.7)
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=8)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()


# Main training and evaluation loop
def main():
    # Parameters
    seq_len = 30
    batch_size = 32
    epochs = 10
    data_path = "data/csv_files/combined_with_labels_and_synthetic.csv"

    # Load and preprocess
    df = data_preprocessing.preprocess_data(data_path).drop(columns=["Label"])
    names = df.columns.to_list()
    length = len(df)
    train_end = int(length * 0.8)
    valid_end = int(length * 0.9)

    train_df = df.iloc[:train_end]
    valid_df = df.iloc[train_end:valid_end]
    test_df = df.iloc[valid_end:]

    scaler = StandardScaler().fit(train_df)
    joblib.dump(scaler, "lstm_scaler.pkl")
    train_df = pd.DataFrame(scaler.transform(train_df), columns=names)
    valid_df = pd.DataFrame(scaler.transform(valid_df), columns=names)
    test_df = pd.DataFrame(scaler.transform(test_df), columns=names)

    # Datasets and loaders
    train_ds = MetricsDataset(train_df, seq_len)
    valid_ds = MetricsDataset(valid_df, seq_len)
    test_ds = MetricsDataset(test_df, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model, loss, optimizer
    device = mps_device
    model = LSTMModel(input_size=len(names)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    # Training
    loss_train_history = []
    loss_valid_history = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)
            loss = loss_fn(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        loss_train_history.append(train_loss)

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                valid_loss += loss_fn(preds, y_batch).item()
        valid_loss /= len(valid_loader)
        loss_valid_history.append(valid_loss)

        print(
            f"Epoch {epoch+1}/{epochs}"
            "Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}"
        )

    # Evaluation
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).cpu().numpy()
            all_preds.append(preds)
            all_trues.append(y_batch.numpy())
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_trues)

    mae = mean_absolute_error(y_true.ravel(), y_pred.ravel())
    print(f"Test MAE: {mae:.4f}")
    mse = mean_squared_error(y_true.ravel(), y_pred.ravel())
    print(f"Test MSE: {mse:.4f}")

    print(loss_train_history)
    print(loss_valid_history)

    plt.plot(loss_train_history, label="training_loss")
    plt.plot(loss_valid_history, label="validation_loss")
    plt.plot(mae, label="mae")
    plt.plot(mse, label="mse")
    plt.legend()
    plt.show()

    # Build DataFrames with suffixes
    df_true = pd.DataFrame(y_true, columns=names).add_suffix("_true")
    df_pred = pd.DataFrame(y_pred, columns=names).add_suffix("_pred")

    df_true_invers_scaling = scaler.inverse_transform(df_true)
    df_pred_invers_scaling = scaler.inverse_transform(df_pred)

    # Print first N rows
    N_print = 10
    print(f"\n=== First {N_print} samples: True and Predicted DF ===")
    print(df_true["CPU Utilization_true"].head(N_print))
    print(df_pred["CPU Utilization_pred"].head(N_print))

    print(df_true_invers_scaling)

    # Plot all metrics
    plot_all_metrics(df_true_invers_scaling, df_pred_invers_scaling, names)


if __name__ == "__main__":
    main()
