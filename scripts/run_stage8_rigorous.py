"""Comprehensive, unified Stage 8 ablation benchmark script.
Trains all models to 30 epochs with CosineAnnealingLR, logs per-epoch loss, syntactic validity, and exact match,
computes fresh per-condition unpadded token accuracy and Oracle/Predicted depth search, and generates artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kolam_r.ablations.model_variants import (
    VisionToGrammarResNet,
    VisionToGrammarSequenceOnly,
)
from kolam_r.dataset.loader import KolamDataset
from kolam_r.grammar.executor import GrammarExecutor
from kolam_r.grammar.tokenizer import GrammarTokenizer
from kolam_r.lsystem.rules import RULES_BY_ID
from kolam_r.models.seq2seq_grammar import VisionToGrammarModel
from kolam_r.training.grammar_trainer import GrammarDataset


class SubsampledGrammarDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: KolamDataset, fraction: float = 1.0, motif_filter: str | None = None, seed: int = 42) -> None:
        self.tokenizer = GrammarTokenizer()
        self.max_seq_len = 110
        self.base = base_dataset

        self.rule_tokens_cache: dict[int, torch.Tensor] = {}
        for idx, rule_id in enumerate(["R01", "R02", "R03", "R04", "R05", "R06"]):
            rule_obj = RULES_BY_ID[rule_id]
            g_str = self.tokenizer.rule_to_grammar_string(rule_obj)
            token_ids = self.tokenizer.encode(g_str, max_length=110, add_special_tokens=True)
            self.rule_tokens_cache[idx] = torch.tensor(token_ids, dtype=torch.long)

        valid_indices = []
        for idx in range(len(base_dataset)):
            rec = base_dataset.records[idx]
            if motif_filter is None or rec.get("motif") == motif_filter:
                valid_indices.append(idx)

        if fraction < 1.0:
            rng = np.random.default_rng(seed)
            n_sub = max(1, int(len(valid_indices) * fraction))
            self.indices = sorted(rng.choice(valid_indices, size=n_sub, replace=False).tolist())
        else:
            self.indices = valid_indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple[np.ndarray, torch.Tensor, dict]:
        real_idx = self.indices[i]
        img, meta = self.base[real_idx]
        rule_idx = int(meta["rule_id"])
        tokens = self.rule_tokens_cache[rule_idx]
        return img, tokens, meta


def collate_fn(batch: list) -> dict[str, torch.Tensor]:
    GRID_MAP = {3: 0, 5: 1, 7: 2, 9: 3}
    imgs = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    tokens = torch.stack([b[1] for b in batch])
    symmetries = torch.tensor([b[2]["symmetry"] for b in batch], dtype=torch.long)
    depths = torch.tensor([b[2]["recursion_depth"] - 1 for b in batch], dtype=torch.long)
    grid_sizes = torch.tensor([GRID_MAP.get(int(b[2]["grid_size"]), 1) for b in batch], dtype=torch.long)
    angles = torch.tensor([float(b[2]["angle"]) / 90.0 for b in batch], dtype=torch.float)

    return {
        "image": imgs,
        "tokens": tokens,
        "symmetry": symmetries,
        "depth": depths,
        "grid_size": grid_sizes,
        "angle": angles,
    }


def evaluate_batch_validity_and_match(
    model: nn.Module,
    val_dataset: KolamDataset,
    device: str = "cpu",
    max_eval: int = 54,
) -> tuple[float, float]:
    """Quick validation-epoch evaluation for syntactic validity and exact match rate."""
    model.eval()
    tokenizer = GrammarTokenizer()
    executor = GrammarExecutor()

    valid_count = 0
    match_count = 0
    n = min(len(val_dataset), max_eval)

    with torch.no_grad():
        for i in range(n):
            rec = val_dataset.records[i]
            img = val_dataset.load_image(i)
            img_t = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            tgt_seq = tokenizer.rule_to_grammar_string(RULES_BY_ID[rec["production_rule_id"]])

            gen_t, _ = model.generate(img_t)
            gen_str = tokenizer.decode(gen_t[0].tolist())
            parsed = executor.parse_grammar_string(gen_str)

            if parsed.is_valid:
                valid_count += 1
            if gen_str == tgt_seq:
                match_count += 1

    return float(valid_count / n), float(match_count / n)


def train_and_track_variant(
    name: str,
    model: nn.Module,
    train_ds: torch.utils.data.Dataset,
    val_ds: torch.utils.data.Dataset,
    val_kolam_ds: KolamDataset,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 5e-4,
    has_aux: bool = True,
    device: str = "cpu",
) -> tuple[dict, list[float], list[float], list[float], list[float]]:
    loader_train = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    loader_val = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    train_losses, val_losses = [], []
    val_syntactic_validities, val_exact_matches = [], []
    best_loss = float("inf")
    best_weights = None

    print(f"\n--- Training {name} ({len(train_ds)} samples, {epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for batch in loader_train:
            imgs = batch["image"].to(device)
            tokens = batch["tokens"].to(device)
            tgt_in = tokens[:, :-1]
            tgt_out = tokens[:, 1:]

            optimizer.zero_grad()
            out = model(imgs, tgt_in)

            if isinstance(out, dict):
                logits = out["token_logits"]
                aux_preds = {
                    "symmetry": out.get("logits_symmetry"),
                    "grid_size": out.get("logits_grid_size"),
                    "depth": out.get("logits_depth_direct"),
                    "pred_angle": out.get("pred_angle_norm"),
                }
            else:
                logits, aux_preds = out

            loss_seq = criterion_ce(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total_loss = loss_seq

            if has_aux and aux_preds:
                loss_aux = 0.0
                if aux_preds.get("symmetry") is not None:
                    loss_aux += F.cross_entropy(aux_preds["symmetry"], batch["symmetry"].to(device))
                if aux_preds.get("grid_size") is not None:
                    loss_aux += F.cross_entropy(aux_preds["grid_size"], batch["grid_size"].to(device))
                if aux_preds.get("depth") is not None:
                    loss_aux += F.cross_entropy(aux_preds["depth"], batch["depth"].to(device))
                if aux_preds.get("pred_angle") is not None:
                    ang_pred = aux_preds["pred_angle"]
                    if ang_pred.ndim > 1:
                        ang_pred = ang_pred.squeeze(-1)
                    loss_aux += F.mse_loss(ang_pred, batch["angle"].to(device))
                total_loss = loss_seq + 0.2 * loss_aux

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_loss += total_loss.item() * len(imgs)

        scheduler.step()
        t_loss /= len(train_ds)
        train_losses.append(t_loss)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for batch in loader_val:
                imgs = batch["image"].to(device)
                tokens = batch["tokens"].to(device)
                tgt_in = tokens[:, :-1]
                tgt_out = tokens[:, 1:]
                out = model(imgs, tgt_in)
                logits = out["token_logits"] if isinstance(out, dict) else out[0]
                l = criterion_ce(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
                v_loss += l.item() * len(imgs)
        v_loss /= len(val_ds)
        val_losses.append(v_loss)

        # Track per-epoch syntactic validity and exact match
        v_syn, v_mat = evaluate_batch_validity_and_match(model, val_kolam_ds, device=device)
        val_syntactic_validities.append(v_syn)
        val_exact_matches.append(v_mat)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train: {t_loss:.4f} | Val: {v_loss:.4f} | Syntax: {v_syn*100:.1f}% | Match: {v_mat*100:.1f}%")

        if v_loss < best_loss:
            best_loss = v_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_weights, train_losses, val_losses, val_syntactic_validities, val_exact_matches


def evaluate_final_unpadded(
    model: nn.Module,
    dataset: KolamDataset,
    device: str = "cpu",
    is_seq_only: bool = False,
) -> dict:
    model.eval()
    tokenizer = GrammarTokenizer()
    executor = GrammarExecutor()

    symmetries = ["C1", "C2", "C4", "D1", "D2", "D4"]
    grid_sizes = [3, 5, 7, 9]

    total_samples = len(dataset)
    correct_grammar = 0
    correct_pred_depth = 0
    correct_oracle_depth = 0
    syntactic_valid = 0

    total_tokens_unpadded = 0
    matching_tokens_unpadded = 0

    for idx in range(total_samples):
        rec = dataset.records[idx]
        img_raw = dataset.load_image(idx)
        img_t = torch.from_numpy(img_raw.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        true_r = rec["production_rule_id"]
        true_d = int(rec["recursion_depth"])

        tgt_seq = tokenizer.rule_to_grammar_string(RULES_BY_ID[true_r])
        tgt_tokens = [t for t in tokenizer.encode(tgt_seq, max_length=110, add_special_tokens=True) if t != tokenizer.vocab.pad_idx]

        with torch.no_grad():
            gen_tok_tensor, aux_preds = model.generate(img_t)

        gen_tokens = gen_tok_tensor[0].cpu().tolist()
        gen_str = tokenizer.decode(gen_tokens)
        parsed_pred = executor.parse_grammar_string(gen_str)
        parsed_oracle = executor.parse_grammar_string(tgt_seq)

        # Unpadded token comparison
        min_len = min(len(tgt_tokens), len(gen_tokens))
        for t_idx in range(min_len):
            if tgt_tokens[t_idx] == gen_tokens[t_idx]:
                matching_tokens_unpadded += 1
        total_tokens_unpadded += max(len(tgt_tokens), len(gen_tokens))

        if parsed_pred.is_valid:
            syntactic_valid += 1

        if gen_str == tgt_seq:
            correct_grammar += 1

        if not is_seq_only and "pred_symmetry" in aux_preds:
            sym_idx = int(aux_preds["pred_symmetry"][0].cpu() if aux_preds["pred_symmetry"].ndim > 0 else aux_preds["pred_symmetry"].cpu())
            grid_idx = int(aux_preds["pred_grid_size"][0].cpu() if aux_preds["pred_grid_size"].ndim > 0 else aux_preds["pred_grid_size"].cpu())
            ang_val = float(aux_preds["pred_angle"][0].cpu() if aux_preds["pred_angle"].ndim > 0 else aux_preds["pred_angle"].cpu())
            sym = symmetries[min(sym_idx, len(symmetries) - 1)]
            grid = grid_sizes[min(grid_idx, len(grid_sizes) - 1)]
            angle = ang_val
        else:
            sym = rec.get("symmetry", "C1")
            grid = int(rec.get("grid_size", 5))
            angle = float(rec.get("angle", 90.0))

        # 1. Predicted Grammar Depth Search
        if parsed_pred.is_valid:
            d_hat_pred, _ = executor.search_depth_ncc(
                parsed_pred,
                target_image=img_raw,
                angle=angle,
                symmetry=sym,
                grid_size=grid,
                motif=rec.get("motif", "M1"),
            )
        else:
            d_hat_pred = -1
        if d_hat_pred == true_d:
            correct_pred_depth += 1

        # 2. Oracle Grammar Depth Search (Fresh per sample and condition)
        d_hat_oracle, _ = executor.search_depth_ncc(
            parsed_oracle,
            target_image=img_raw,
            angle=float(rec.get("angle", 90.0)),
            symmetry=rec.get("symmetry", "C1"),
            grid_size=int(rec.get("grid_size", 5)),
            motif=rec.get("motif", "M1"),
        )
        if d_hat_oracle == true_d:
            correct_oracle_depth += 1

    per_token_acc = float(matching_tokens_unpadded / total_tokens_unpadded)
    exact_match_rate = float(correct_grammar / total_samples)

    return {
        "exact_grammar_match_rate": exact_match_rate,
        "per_token_accuracy": per_token_acc,
        "syntactic_validity_rate": float(syntactic_valid / total_samples),
        "protocol_b_depth_accuracy_predicted": float(correct_pred_depth / total_samples),
        "protocol_b_depth_accuracy_oracle": float(correct_oracle_depth / total_samples),
    }


def main() -> None:
    device = "cpu"
    splits_root = Path("data/splits")
    base_train = KolamDataset(splits_root / "train.json")
    base_val = KolamDataset(splits_root / "val.json")

    ds_train_100 = GrammarDataset(base_train)
    ds_train_50 = SubsampledGrammarDataset(base_train, fraction=0.5, seed=42)
    ds_train_m1 = SubsampledGrammarDataset(base_train, fraction=1.0, motif_filter="M1")
    ds_val = GrammarDataset(base_val)

    test_iid = KolamDataset(splits_root / "test_iid.json")
    test_heldout = KolamDataset(splits_root / "test_heldout_depth.json")

    # 1. Full Multi-Task Model (Canonical)
    m_full = VisionToGrammarModel(d_model=256).to(device)
    ckpt_full = torch.load("checkpoints/best_grammar_model.pt", map_location="cpu", weights_only=True)
    m_full.load_state_dict(ckpt_full["model_state_dict"])
    res_full_iid = evaluate_final_unpadded(m_full, test_iid, device=device)
    res_full_held = evaluate_final_unpadded(m_full, test_heldout, device=device)

    # 2. Sequence-Only Model (30 Epochs)
    m_seq = VisionToGrammarSequenceOnly(d_model=256).to(device)
    w_seq, t_seq, v_seq, syn_seq, mat_seq = train_and_track_variant(
        "Sequence-Only", m_seq, ds_train_100, ds_val, base_val, epochs=30, has_aux=False, device=device
    )
    m_seq.load_state_dict(w_seq)
    res_seq_iid = evaluate_final_unpadded(m_seq, test_iid, device=device, is_seq_only=True)

    # 3. ResNet-18 Model (30 Epochs)
    m_res = VisionToGrammarResNet(d_model=256).to(device)
    w_res, t_res, v_res, syn_res, mat_res = train_and_track_variant(
        "ResNet-18", m_res, ds_train_100, ds_val, base_val, epochs=30, has_aux=True, device=device
    )
    m_res.load_state_dict(w_res)
    res_res_iid = evaluate_final_unpadded(m_res, test_iid, device=device)

    # 4. 50% Data Scale (30 Epochs)
    m_50 = VisionToGrammarModel(d_model=256).to(device)
    w_50, t_50, v_50, syn_50, mat_50 = train_and_track_variant(
        "50% Data 4-Motifs", m_50, ds_train_50, ds_val, base_val, epochs=30, has_aux=True, device=device
    )
    m_50.load_state_dict(w_50)
    res_50_iid = evaluate_final_unpadded(m_50, test_iid, device=device)

    # 5. Single-Motif M1 Model (30 Epochs)
    m_m1 = VisionToGrammarModel(d_model=256).to(device)
    w_m1, t_m1, v_m1, syn_m1, mat_m1 = train_and_track_variant(
        "Single-Motif M1", m_m1, ds_train_m1, ds_val, base_val, epochs=30, has_aux=True, device=device
    )
    m_m1.load_state_dict(w_m1)

    ds_m1_seen = KolamDataset(splits_root / "test_iid.json")
    ds_m1_seen.records = [test_iid.records[i] for i, r in enumerate(test_iid.records) if r.get("motif") == "M1"]
    ds_m1_unseen = KolamDataset(splits_root / "test_iid.json")
    ds_m1_unseen.records = [test_iid.records[i] for i, r in enumerate(test_iid.records) if r.get("motif") != "M1"]

    res_m1_seen = evaluate_final_unpadded(m_m1, ds_m1_seen, device=device)
    res_m1_unseen = evaluate_final_unpadded(m_m1, ds_m1_unseen, device=device)

    # Compile Final JSON Artifact
    final_output = {
        "full_multitask_test_iid": res_full_iid,
        "full_multitask_test_heldout": res_full_held,
        "sequence_only_test_iid": res_seq_iid,
        "resnet_backbone_test_iid": res_res_iid,
        "subsampled_50pct_data_test_iid": res_50_iid,
        "single_motif_m1_seen_eval": res_m1_seen,
        "single_motif_m1_unseen_transfer": res_m1_unseen,
        "per_epoch_trajectories": {
            "Sequence-Only": {
                "train_loss": t_seq,
                "val_loss": v_seq,
                "val_syntactic_validity": syn_seq,
                "val_exact_match": mat_seq,
            },
            "ResNet-18": {
                "train_loss": t_res,
                "val_loss": v_res,
                "val_syntactic_validity": syn_res,
                "val_exact_match": mat_res,
            },
            "50% Data (4 Motifs)": {
                "train_loss": t_50,
                "val_loss": v_50,
                "val_syntactic_validity": syn_50,
                "val_exact_match": mat_50,
            },
            "Single-Motif M1": {
                "train_loss": t_m1,
                "val_loss": v_m1,
                "val_syntactic_validity": syn_m1,
                "val_exact_match": mat_m1,
            },
        },
    }

    with open("results/ablation_evaluation.json", "w") as f:
        json.dump(final_output, f, indent=2)

    # Plot Multi-Panel Clean Figures matching exact 30 epoch data points
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=200, constrained_layout=True)
    epochs_30 = list(range(1, 31))

    # Panel 1: Sequence-Only
    axes[0, 0].plot(epochs_30, t_seq, label="Train Loss", color="crimson", linestyle="--")
    axes[0, 0].plot(epochs_30, v_seq, label="Val Loss", color="darkred")
    axes[0, 0].set_title("Ablation 1: Sequence-Only (30 Epochs)", fontsize=10, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch", fontsize=8)
    axes[0, 0].set_ylabel("Loss", fontsize=8)
    axes[0, 0].set_xlim(1, 30)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: ResNet-18
    axes[0, 1].plot(epochs_30, t_res, label="Train Loss", color="blue", linestyle="--")
    axes[0, 1].plot(epochs_30, v_res, label="Val Loss", color="navy")
    axes[0, 1].set_title("Ablation 2: ResNet-18 (30 Epochs)", fontsize=10, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch", fontsize=8)
    axes[0, 1].set_ylabel("Loss", fontsize=8)
    axes[0, 1].set_xlim(1, 30)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Data Scale & Motif Loss
    axes[1, 0].plot(epochs_30, v_50, label="50% Data (4 Motifs) Val", color="orange")
    axes[1, 0].plot(epochs_30, v_m1, label="25% Data (M1-Only) Val", color="purple")
    axes[1, 0].set_title("Ablation 3: Val Loss vs Data Scale (30 Epochs)", fontsize=10, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch", fontsize=8)
    axes[1, 0].set_ylabel("Validation Loss", fontsize=8)
    axes[1, 0].set_xlim(1, 30)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Per-Epoch Exact Match Trajectory (Single-Motif M1 vs 50% Data)
    axes[1, 1].plot(epochs_30, [x * 100 for x in mat_50], label="50% Data (4 Motifs)", color="orange")
    axes[1, 1].plot(epochs_30, [x * 100 for x in mat_m1], label="Single-Motif M1", color="purple")
    axes[1, 1].set_title("Ablation 3: Exact Match Rate (%) over 30 Epochs", fontsize=10, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch", fontsize=8)
    axes[1, 1].set_ylabel("Val Exact Match (%)", fontsize=8)
    axes[1, 1].set_xlim(1, 30)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("KOLAM-R Stage 8 — Systematic Ablation Trajectories (30 Epochs)", fontsize=13, fontweight="bold")
    fig.savefig("results/ablation_curves.png", bbox_inches="tight")
    plt.close(fig)

    print("Finished successfully.")


if __name__ == "__main__":
    main()
