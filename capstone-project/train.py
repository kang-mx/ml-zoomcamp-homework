import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

# Config 
WINDOW_SIZE = 30
EPOCHS = 50
BATCH_SIZE = 64
HIDDEN_DIM = 128
DROPOUT = 0.0
LR = 0.0005
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Column definitions
COLUMNS = [
    "unit_number", "cycle",
    "op_setting_1", "op_setting_2", "op_setting_3",
    "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5",
    "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10",
    "sensor_11", "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_16", "sensor_17", "sensor_18", "sensor_19", "sensor_20", "sensor_21",
]

# Load data
def load_data():
    df_train = pd.read_csv("train_FD001.txt", sep=r'\s+', header=None, names=COLUMNS)
    df_test  = pd.read_csv("test_FD001.txt",  sep=r'\s+', header=None, names=COLUMNS)
    df_rul   = pd.read_csv("RUL_FD001.txt",   sep=r'\s+', header=None, names=["RUL"])
    return df_train, df_test, df_rul

# Feature engineering
def prepare_data(df_train, df_test, df_rul):
    # Add RUL to train
    max_cycles = df_train.groupby("unit_number")["cycle"].max()
    df_train["RUL"] = df_train.apply(
        lambda row: max_cycles[row["unit_number"]] - row["cycle"], axis=1
    )

    # Add RUL to test
    df_test["RUL"] = (
        df_rul["RUL"].values
        + df_test.groupby("unit_number")["cycle"].transform("max")
        - df_test["cycle"]
    )

    # Remove low variance sensors
    sensor_cols = [c for c in df_train.columns if c.startswith("sensor_")]
    low_var = df_train[sensor_cols].std()
    low_var_cols = low_var[low_var < 1e-3].index.tolist()
    df_train.drop(columns=low_var_cols, inplace=True)
    df_test.drop(columns=low_var_cols,  inplace=True)

    # Remove operational settings
    op_cols = [c for c in df_train.columns if c.startswith("op_setting_")]
    df_train.drop(columns=op_cols, inplace=True)
    df_test.drop(columns=op_cols,  inplace=True)

    # Normalize
    feature_cols = [c for c in df_train.columns if c.startswith("sensor_") or c == "cycle"]
    scaler = MinMaxScaler()
    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_test[feature_cols]  = scaler.transform(df_test[feature_cols])
    joblib.dump(scaler, "scaler.pkl")
    print("Scaler saved to scaler.pkl")

    return df_train, df_test, feature_cols

# Sequence creation
def create_sequences(df, feature_cols, window_size):
    sequences, targets = [], []
    for engine in df["unit_number"].unique():
        engine_data = df[df["unit_number"] == engine].sort_values("cycle")
        data_array  = engine_data[feature_cols].values
        rul_array   = engine_data["RUL"].values
        for i in range(len(data_array) - window_size):
            sequences.append(data_array[i:i+window_size])
            targets.append(rul_array[i + window_size])
    return np.array(sequences), np.array(targets)

# Dataset
class RULDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(DEVICE)
    def __len__(self):  return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# Model
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# Train
def train(model, train_loader, optimizer, criterion):
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} done")

# Main
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    df_train, df_test, df_rul = load_data()
    df_train, df_test, feature_cols = prepare_data(df_train, df_test, df_rul)

    engines = df_train["unit_number"].unique()
    train_units, val_units = train_test_split(engines, test_size=0.2, random_state=RANDOM_STATE)
    train_df = df_train[df_train["unit_number"].isin(train_units)]
    val_df   = df_train[df_train["unit_number"].isin(val_units)]

    X_train_seq, y_train_seq = create_sequences(train_df, feature_cols, WINDOW_SIZE)
    X_val_seq,   y_val_seq   = create_sequences(val_df,   feature_cols, WINDOW_SIZE)

    train_loader = DataLoader(RULDataset(X_train_seq, y_train_seq), batch_size=BATCH_SIZE, shuffle=True)

    model     = LSTMRegressor(input_dim=X_train_seq.shape[2], hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    train(model, train_loader, optimizer, criterion)

    # Evaluate on validation set
    model.eval()
    X_val_t = torch.tensor(X_val_seq, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        y_pred = model(X_val_t).cpu().numpy()

    rmse = np.sqrt(mean_squared_error(y_val_seq, y_pred))
    mae  = mean_absolute_error(y_val_seq, y_pred)
    print(f"\nValidation RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    torch.save(model.state_dict(), "lstm_rul_final.pth")
    print("Model saved to lstm_rul_final.pth")