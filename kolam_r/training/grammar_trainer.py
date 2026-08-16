"""Training loop and validation engine for Vision-to-Grammar Model."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from kolam_r.dataset.loader import KolamDataset
from kolam_r.grammar.tokenizer import GrammarTokenizer
from kolam_r.grammar.vocabulary import PAD_IDX, VOCAB_TOKENS
from kolam_r.lsystem.rules import RULES_BY_ID
from kolam_r.models.seq2seq_grammar import VisionToGrammarModel
from kolam_r.schema import VALID_RULES
from kolam_r.training.grammar_losses import GrammarRecoveryLoss


class GrammarDataset(Dataset):
    """Wraps a KolamDataset and appends tokenized ground truth grammar sequences."""

    def __init__(self, base_dataset: KolamDataset, max_seq_len: int = 110) -> None:
        self.base = base_dataset
        self.tokenizer = GrammarTokenizer()
        self.max_seq_len = max_seq_len

        # Precompute grammar token IDs for all valid rules
        self.rule_tokens_cache: dict[int, torch.Tensor] = {}
        for idx, rule_id in enumerate(VALID_RULES):
            rule_obj = RULES_BY_ID[rule_id]
            g_str = self.tokenizer.rule_to_grammar_string(rule_obj)
            token_ids = self.tokenizer.encode(g_str, max_length=max_seq_len, add_special_tokens=True)
            self.rule_tokens_cache[idx] = torch.tensor(token_ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, torch.Tensor, dict[str, Any]]:
        img, meta = self.base[idx]
        rule_idx = int(meta["rule_id"])
        tokens = self.rule_tokens_cache[rule_idx]
        return img, tokens, meta


class GrammarTrainer:
    """Trainer for Vision-to-Grammar model."""

    def __init__(
        self,
        model: VisionToGrammarModel,
        train_dataset: KolamDataset,
        val_dataset: KolamDataset,
        criterion: GrammarRecoveryLoss | None = None,
        batch_size: int = 32,
        lr: float = 5e-4,
        weight_decay: float = 1e-4,
        device: str | None = None,
    ) -> None:
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device)
        self.criterion = (criterion or GrammarRecoveryLoss()).to(self.device)

        self.train_ds = GrammarDataset(train_dataset)
        self.val_ds = GrammarDataset(val_dataset)

        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=40, eta_min=1e-5
        )

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_seq_loss": [],
            "val_seq_loss": [],
            "val_token_acc": [],
            "val_seq_match": [],
        }

    def _collate_batch(
        self, batch: tuple[Any, Any, dict[str, Any]]
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        imgs_raw, tokens_raw, targets_raw = batch

        if isinstance(imgs_raw, np.ndarray):
            imgs = torch.from_numpy(imgs_raw).float()
        else:
            imgs = imgs_raw.float()
        imgs = imgs.to(self.device)

        tokens = tokens_raw.to(self.device)

        targets: dict[str, torch.Tensor] = {}
        for k, v in targets_raw.items():
            if isinstance(v, (list, tuple)):
                v = torch.tensor(v)
            elif isinstance(v, np.ndarray):
                v = torch.from_numpy(v)
            targets[k] = v.to(self.device)

        return imgs, tokens, targets

    def train_epoch(self) -> tuple[float, float]:
        self.model.train()
        total_loss, total_seq_loss = 0.0, 0.0

        for batch in self.train_loader:
            imgs, tokens, targets = self._collate_batch(batch)

            self.optimizer.zero_grad()
            outputs = self.model(imgs, target_tokens=tokens)
            loss_dict = self.criterion(outputs, target_tokens=tokens, target_meta=targets)
            loss = loss_dict["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_seq_loss += loss_dict["loss_seq"].item() * imgs.size(0)

        n = len(self.train_loader.dataset)
        return total_loss / n, total_seq_loss / n

    def validate_epoch(self) -> tuple[float, float, float, float]:
        self.model.eval()
        total_loss, total_seq_loss = 0.0, 0.0
        correct_tokens, total_tokens = 0, 0
        exact_matches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                imgs, tokens, targets = self._collate_batch(batch)
                outputs = self.model(imgs, target_tokens=tokens)
                loss_dict = self.criterion(outputs, target_tokens=tokens, target_meta=targets)

                total_loss += loss_dict["total_loss"].item() * imgs.size(0)
                total_seq_loss += loss_dict["loss_seq"].item() * imgs.size(0)

                # Compute token accuracy (excluding padding)
                pred_tokens = outputs["token_logits"][:, :-1, :].argmax(dim=-1)
                target_shift = tokens[:, 1:]
                non_pad_mask = (target_shift != PAD_IDX)

                correct_tokens += ((pred_tokens == target_shift) & non_pad_mask).sum().item()
                total_tokens += non_pad_mask.sum().item()

                # Autoregressive generation for validation sequence exact match
                gen_tokens, _ = self.model.generate(imgs, max_length=self.model.max_seq_len)
                for gen, tgt in zip(gen_tokens, tokens):
                    # Compare tokens up to EOS
                    gen_list = gen.tolist()
                    tgt_list = tgt.tolist()
                    gen_clean = [t for t in gen_list if t not in {PAD_IDX, 1}][:len([t for t in tgt_list if t not in {PAD_IDX, 1}])]
                    tgt_clean = [t for t in tgt_list if t not in {PAD_IDX, 1}]
                    if gen_clean == tgt_clean:
                        exact_matches += 1

        n = len(self.val_loader.dataset)
        token_acc = correct_tokens / max(total_tokens, 1)
        seq_match = exact_matches / n
        return total_loss / n, total_seq_loss / n, token_acc, seq_match

    def train(
        self,
        num_epochs: int = 40,
        checkpoint_dir: Path | str | None = None,
        verbose: bool = True,
    ) -> Path:
        chk_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        chk_dir.mkdir(parents=True, exist_ok=True)
        best_model_path = chk_dir / "best_grammar_model.pt"

        best_val_loss = float("inf")
        best_epoch = 0

        if verbose:
            print(f"Training Vision-to-Grammar Model on device: {self.device} | {num_epochs} epochs")
            print("-" * 75)

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            tr_loss, tr_seq = self.train_epoch()
            va_loss, va_seq, tok_acc, seq_match = self.validate_epoch()
            self.scheduler.step()
            dt = time.time() - t0

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(va_loss)
            self.history["train_seq_loss"].append(tr_seq)
            self.history["val_seq_loss"].append(va_seq)
            self.history["val_token_acc"].append(tok_acc)
            self.history["val_seq_match"].append(seq_match)

            is_best = va_loss < best_val_loss
            if is_best:
                best_val_loss = va_loss
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": va_loss,
                        "token_acc": tok_acc,
                        "seq_match": seq_match,
                    },
                    best_model_path,
                )

            if verbose and (epoch % 5 == 0 or epoch == num_epochs or is_best):
                marker = "*" if is_best else " "
                print(
                    f"Epoch {epoch:2d}/{num_epochs:2d} [{dt:.1f}s]{marker} | "
                    f"TrLoss: {tr_loss:.4f} (Seq: {tr_seq:.4f}) | "
                    f"ValLoss: {va_loss:.4f} (Seq: {va_seq:.4f}, TokAcc: {tok_acc*100:.1f}%, SeqMatch: {seq_match*100:.1f}%)"
                )

        if verbose:
            print("-" * 75)
            print(f"Training complete. Best model checkpoint: {best_model_path} (Epoch {best_epoch}, Val Loss: {best_val_loss:.4f})")

        return best_model_path

    def plot_training_curves(self, output_path: Path | str) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        epochs = range(1, len(self.history["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

        # Loss Curve
        ax = axes[0]
        ax.plot(epochs, self.history["train_loss"], label="Train Total Loss", color="#2563eb", lw=2)
        ax.plot(epochs, self.history["val_loss"], label="Val Total Loss", color="#dc2626", lw=2, linestyle="--")
        ax.set_title("Total Multi-Task Loss", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Sequence Cross-Entropy Loss
        ax = axes[1]
        ax.plot(epochs, self.history["train_seq_loss"], label="Train Seq Loss", color="#10b981", lw=2)
        ax.plot(epochs, self.history["val_seq_loss"], label="Val Seq Loss", color="#8b5cf6", lw=2, linestyle="--")
        ax.set_title("Autoregressive Sequence Loss", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross Entropy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Validation Metrics
        ax = axes[2]
        ax.plot(epochs, [v * 100 for v in self.history["val_token_acc"]], label="Token Accuracy", color="#3b82f6", lw=2)
        ax.plot(epochs, [v * 100 for v in self.history["val_seq_match"]], label="Exact Seq Match", color="#f59e0b", lw=2)
        ax.set_title("Validation Sequence Metrics", fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return output_path
