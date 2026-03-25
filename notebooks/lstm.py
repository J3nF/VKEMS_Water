"""
LSTM Water Leakage Predictor
============================
Predicts water leakage from environmental features:
  - Air temperature
  - Precipitation
  - Soil moisture
  - Soil temperature

Data: weekly measurements over ~20 years (~1,040 time steps)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ─── Configuration ────────────────────────────────────────────────────────────

FEATURE_COLS = ["temperature_2m", "soil_temperature_100_to_255cm", "soil_moisture_100_to_255cm", "precipitation"]
TARGET_COL   = "water_leakage"

SEQ_LEN      = 12      # look-back window (weeks)
BATCH_SIZE   = 32
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.2
LEARNING_RATE = 1e-3
EPOCHS       = 50
TEST_SIZE    = 0.2     # fraction of data held out for testing
VAL_SIZE     = 0.1     # fraction of training data used for validation
SEED         = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset.
    Each sample:  X  →  (SEQ_LEN, n_features)
                  y  →  scalar (leakage at the next time step)
    """
    def __init__(self, features: np.ndarray, targets: np.ndarray, seq_len: int):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets  = torch.tensor(targets,  dtype=torch.float32)
        self.seq_len  = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_len]          # (SEQ_LEN, n_features)
        y = self.targets[idx + self.seq_len]                  # scalar
        return x, y


# ─── Model ────────────────────────────────────────────────────────────────────

class LeakageLSTM(nn.Module):
    """
    Stacked LSTM followed by a fully-connected regression head.
    """
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)          # out: (batch, seq_len, hidden_size)
        last    = out[:, -1, :]        # take the last time step
        return self.head(last).squeeze(-1)


# ─── Training helpers ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        total_loss += criterion(pred, y).item() * len(y)
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.concatenate(preds), np.concatenate(trues)


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run(df: pd.DataFrame):
    """
    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns listed in FEATURE_COLS and TARGET_COL,
        sorted chronologically (one row per week).
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ── 1. Scale features and target separately ──────────────────────────────
    feat_scaler   = StandardScaler()
    target_scaler = StandardScaler()

    features = feat_scaler.fit_transform(df[FEATURE_COLS].values)
    targets  = target_scaler.fit_transform(
                    df[[TARGET_COL]].values).ravel()

    # ── 2. Chronological train / val / test split ─────────────────────────────
    n          = len(features)
    n_test     = int(n * TEST_SIZE)
    n_val      = int((n - n_test) * VAL_SIZE)
    n_train    = n - n_test - n_val

    f_train, f_val, f_test = (features[:n_train],
                               features[n_train:n_train+n_val],
                               features[n_train+n_val:])
    t_train, t_val, t_test = (targets[:n_train],
                               targets[n_train:n_train+n_val],
                               targets[n_train+n_val:])

    print(f"Splits — train: {n_train}, val: {n_val}, test: {n_test}")

    # ── 3. Datasets & loaders ────────────────────────────────────────────────
    train_ds = TimeSeriesDataset(f_train, t_train, SEQ_LEN)
    val_ds   = TimeSeriesDataset(f_val,   t_val,   SEQ_LEN)
    test_ds  = TimeSeriesDataset(f_test,  t_test,  SEQ_LEN)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # ── 4. Model, optimiser, scheduler ───────────────────────────────────────
    model     = LeakageLSTM(len(FEATURE_COLS), HIDDEN_SIZE,
                             NUM_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # ── 5. Training loop ─────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_dl, optimizer, criterion)
        val_loss, _, _ = evaluate(model, val_dl, criterion)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"train MSE: {train_loss:.4f} | val MSE: {val_loss:.4f}")

    # ── 6. Evaluate on test set ───────────────────────────────────────────────
    model.load_state_dict(best_state)
    test_loss, preds_scaled, trues_scaled = evaluate(model, test_dl, criterion)

    # Inverse-transform back to original units
    preds = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    trues = target_scaler.inverse_transform(trues_scaled.reshape(-1, 1)).ravel()

    mae  = np.mean(np.abs(preds - trues))
    rmse = np.sqrt(np.mean((preds - trues) ** 2))
    ss_res = np.sum((trues - preds) ** 2)
    ss_tot = np.sum((trues - trues.mean()) ** 2)
    r2   = 1 - ss_res / ss_tot

    print(f"\n── Test Results ──────────────────────────────")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  R²   : {r2:.4f}")

    # ── 7. Plots ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(history["train_loss"], label="Train MSE")
    axes[0].plot(history["val_loss"],   label="Val MSE")
    axes[0].set_title("Training / Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (scaled)")
    axes[0].legend()

    axes[1].plot(trues, label="Actual",    alpha=0.8)
    axes[1].plot(preds, label="Predicted", alpha=0.8, linestyle="--")
    axes[1].set_title("Test Set Predictions")
    axes[1].set_xlabel("Week")
    axes[1].set_ylabel("Water Leakage")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("lstm_results.png", dpi=150)
    plt.show()
    print("Plot saved to lstm_results.png")

    # ── 8. Save model ─────────────────────────────────────────────────────────
    torch.save({
        "model_state":    best_state,
        "feat_scaler":    feat_scaler,
        "target_scaler":  target_scaler,
        "config": {
            "input_size":  len(FEATURE_COLS),
            "hidden_size": HIDDEN_SIZE,
            "num_layers":  NUM_LAYERS,
            "dropout":     DROPOUT,
            "seq_len":     SEQ_LEN,
        },
    }, "leakage_lstm.pt")
    print("Model saved to leakage_lstm.pt")

    return model, feat_scaler, target_scaler


# ─── Inference helper ─────────────────────────────────────────────────────────

def predict(checkpoint_path: str, recent_df: pd.DataFrame) -> float:
    """
    Load a saved model and predict leakage for the NEXT week.

    Parameters
    ----------
    checkpoint_path : str
        Path to the .pt file saved by run().
    recent_df : pd.DataFrame
        At least SEQ_LEN rows of feature data (most recent last),
        with the same FEATURE_COLS columns.

    Returns
    -------
    float : predicted water leakage in original units.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg  = ckpt["config"]

    model = LeakageLSTM(cfg["input_size"], cfg["hidden_size"],
                        cfg["num_layers"],  cfg["dropout"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    feat_scaler   = ckpt["feat_scaler"]
    target_scaler = ckpt["target_scaler"]

    window = recent_df[FEATURE_COLS].values[-cfg["seq_len"]:]
    x      = feat_scaler.transform(window)
    x_t    = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # (1, seq, feat)

    with torch.no_grad():
        pred_scaled = model(x_t).item()

    return target_scaler.inverse_transform([[pred_scaled]])[0][0]


# ─── Quick smoke-test with synthetic data ─────────────────────────────────────

if __name__ == "__main__":
    csv_file = 'full_weekly_data.csv'
    df = pd.read_csv(csv_file, parse_dates=['date'])
    df.set_index('date', inplace=True)
    run(df)