"""Training script for Kolam Image-to-Image Autoencoder Baseline.

Trains on data/splits/train.json using combined MSE and BCE loss.
Saves checkpoint to checkpoints/best_autoencoder.pt.
"""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from kolam_r.baselines.autoencoder import KolamAutoencoder
from kolam_r.dataset.loader import KolamDataset


class AutoencoderTorchDataset(Dataset):
    def __init__(self, json_path: str | Path) -> None:
        self.ds = KolamDataset(json_path)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_np = self.ds.load_image(idx).astype(np.float32) / 255.0
        return torch.from_numpy(img_np).unsqueeze(0)


def train_autoencoder(
    train_json: str | Path = "data/splits/train.json",
    val_json: str | Path = "data/splits/val.json",
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 1e-3,
    save_path: str | Path = "checkpoints/best_autoencoder.pt",
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Autoencoder on {device} for {epochs} epochs...")

    train_ds = AutoencoderTorchDataset(train_json)
    val_ds = AutoencoderTorchDataset(val_json)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = KolamAutoencoder(latent_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(batch)
        t_loss /= len(train_ds)
        train_losses.append(t_loss)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = criterion(recon, batch)
                v_loss += loss.item() * len(batch)
        v_loss /= len(val_ds)
        val_losses.append(v_loss)

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model_state_dict": model.state_dict(), "val_loss": v_loss}, save_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss (MSE): {t_loss:.5f} | Val Loss (MSE): {v_loss:.5f}")

    print(f"Training complete. Best Val Loss: {best_val_loss:.5f}. Checkpoint saved to {save_path}")


if __name__ == "__main__":
    train_autoencoder()
