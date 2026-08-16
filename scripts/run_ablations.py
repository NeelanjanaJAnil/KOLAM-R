"""Rigorous training, curve logging, and evaluation runner for Stage 8 Ablations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from kolam_r.ablations.model_variants import (
    VisionToGrammarResNet,
    VisionToGrammarSequenceOnly,
)
from kolam_r.ablations.search_metrics import search_depth_multi_metric
from kolam_r.dataset.loader import KolamDataset
from kolam_r.grammar.executor import GrammarExecutor, ParsedGrammar
from kolam_r.grammar.tokenizer import GrammarTokenizer
from kolam_r.grammar.vocabulary import EOS_IDX, PAD_IDX
from kolam_r.lsystem.rules import RULES_BY_ID
from kolam_r.models.seq2seq_grammar import VisionToGrammarModel
from kolam_r.schema import VALID_RULES
from kolam_r.training.grammar_trainer import GrammarDataset

GRID_SIZE_MAP = {3: 0, 5: 1, 7: 2, 9: 3}


class SubsampledGrammarDataset(Dataset):
    def __init__(self, base_dataset: KolamDataset, fraction: float = 1.0, motif_filter: str | None = None, seed: int = 42) -> None:
        self.tokenizer = GrammarTokenizer()
        self.max_seq_len = 110
        self.base = base_dataset

        self.rule_tokens_cache: dict[int, torch.Tensor] = {}
        for idx, rule_id in enumerate(VALID_RULES):
            rule_obj = RULES_BY_ID[rule_id]
            g_str = self.tokenizer.rule_to_grammar_string(rule_obj)
            token_ids = self.tokenizer.encode(g_str, max_length=110, add_special_tokens=True)
            self.rule_tokens_cache[idx] = torch.tensor(token_ids, dtype=torch.long)

        # Filter by motif if specified
        valid_indices = []
        for idx in range(len(base_dataset)):
            rec = base_dataset.records[idx]
            if motif_filter is None or rec.get("motif") == motif_filter:
                valid_indices.append(idx)

        # Subsample by fraction if needed
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
    imgs = torch.from_numpy(np.stack([b[0] for b in batch])).float()
    tokens = torch.stack([b[1] for b in batch])
    
    symmetries = torch.tensor([b[2]["symmetry"] for b in batch], dtype=torch.long)
    motifs = torch.tensor([b[2]["motif"] for b in batch], dtype=torch.long)
    depths = torch.tensor([b[2]["recursion_depth"] - 1 for b in batch], dtype=torch.long)
    grid_sizes = torch.tensor([GRID_SIZE_MAP.get(int(b[2]["grid_size"]), 1) for b in batch], dtype=torch.long)
    angles = torch.tensor([float(b[2]["angle"]) / 90.0 for b in batch], dtype=torch.float)

    return {
        "image": imgs,
        "tokens": tokens,
        "symmetry": symmetries,
        "motif": motifs,
        "depth": depths,
        "grid_size": grid_sizes,
        "angle": angles,
    }


def train_and_log_variant(
    name: str,
    model: nn.Module,
    train_ds: Dataset,
    val_ds: Dataset,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 5e-4,
    has_aux: bool = True,
    device: str = "cpu",
) -> tuple[dict, list[float], list[float]]:
    loader_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    loader_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)

    train_losses = []
    val_losses = []
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
            optimizer.step()
            t_loss += total_loss.item() * len(imgs)

        t_loss /= len(train_ds)
        train_losses.append(t_loss)

        # Validation
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

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {t_loss:.4f} | Val Loss: {v_loss:.4f}")

        if v_loss < best_loss:
            best_loss = v_loss
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_weights)
    return best_weights, train_losses, val_losses


_RENDER_CACHE: dict[tuple, np.ndarray] = {}

def cached_render_candidate(axiom: str, prod_tuple: tuple, d: int, angle: float, symmetry: str, grid_size: int, motif: str, max_safe_depth: int) -> np.ndarray:
    key = (axiom, prod_tuple, d, round(angle, 1), symmetry, grid_size, motif)
    if key in _RENDER_CACHE:
        return _RENDER_CACHE[key]

    from kolam_r.lsystem.engine import LSystemEngine
    from kolam_r.lsystem.rules import ProductionRule
    from kolam_r.renderer.image_renderer import render_kolam
    from kolam_r.symmetry.transforms import apply_symmetry
    from kolam_r.turtle.interpreter import TurtleInterpreter

    prod_dict = dict(prod_tuple)
    rule = ProductionRule(
        rule_id="DYNAMIC",
        name="Dynamic Grammar",
        axiom=axiom,
        productions=prod_dict,
        default_angle=angle,
        description="Dynamic grammar for depth search",
        max_safe_depth=max_safe_depth,
        connectivity="closed",
        source="Synthesized",
    )
    try:
        expanded = engine.expand_rule(rule, depth=d)
        if len(expanded) > 3000:
            img = np.zeros((64, 64), dtype=np.uint8)
        else:
            res = interp.interpret(expanded, angle=angle)
            if len(res.segments) > 1500:
                img = np.zeros((64, 64), dtype=np.uint8)
            else:
                sym_segs = apply_symmetry(res.segments, symmetry=symmetry)
                img = render_kolam(segments=sym_segs, image_size=64, motif=motif, grid_size=grid_size)
    except Exception:
        img = np.zeros((64, 64), dtype=np.uint8)

    _RENDER_CACHE[key] = img
    return img


def fast_search_depth(
    parsed: ParsedGrammar,
    target_image: np.ndarray,
    angle: float,
    symmetry: str = "C1",
    grid_size: int = 5,
    motif: str = "M1",
    max_safe_depth: int = 4,
    search_metric: str = "ncc",
) -> tuple[int, float]:
    if not parsed.is_valid:
        return 1, 0.0

    from kolam_r.reconstruction.metrics import (
        compute_iou,
        compute_mse,
        compute_ncc,
        compute_ssim,
    )

    prod_tuple = tuple(sorted(parsed.productions.items()))
    best_d = 1
    best_score = -float("inf") if search_metric != "mse" else float("inf")

    for d in range(1, max_safe_depth + 1):
        candidate = cached_render_candidate(
            axiom=parsed.axiom,
            prod_tuple=prod_tuple,
            d=d,
            angle=angle,
            symmetry=symmetry,
            grid_size=grid_size,
            motif=motif,
            max_safe_depth=max_safe_depth,
        )

        if search_metric == "ncc":
            score = compute_ncc(candidate, target_image)
            if score > best_score:
                best_score = score
                best_d = d
        elif search_metric == "ssim":
            score = compute_ssim(candidate, target_image)
            if score > best_score:
                best_score = score
                best_d = d
        elif search_metric == "iou":
            score = compute_iou(candidate, target_image)
            if score > best_score:
                best_score = score
                best_d = d
        elif search_metric == "mse":
            score = compute_mse(candidate, target_image)
            if score < best_score:
                best_score = score
                best_d = d

    return best_d, float(best_score)


def evaluate_model_on_dataset(
    model: nn.Module,
    dataset: KolamDataset,
    device: str = "cpu",
    is_sequence_only: bool = False,
) -> dict:
    model.eval()
    tokenizer = GrammarTokenizer()
    executor = GrammarExecutor()

    correct_grammar = 0
    correct_depth = 0
    syntactic_valid = 0

    symmetries = ["C1", "C2", "C4", "D1", "D2", "D4"]
    grid_sizes = [3, 5, 7, 9]

    for idx in range(len(dataset)):
        rec = dataset.records[idx]
        img_raw = dataset.load_image(idx)
        img_t = torch.from_numpy(img_raw.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
        true_r = rec["production_rule_id"]
        true_d = int(rec["recursion_depth"])

        tgt_seq = tokenizer.rule_to_grammar_string(RULES_BY_ID[true_r])

        with torch.no_grad():
            gen_tokens, aux_preds = model.generate(img_t)

        token_ids = gen_tokens[0].cpu().tolist()
        gen_str = tokenizer.decode(token_ids)
        parsed = executor.parse_grammar_string(gen_str)

        if parsed.is_valid:
            syntactic_valid += 1

        if gen_str == tgt_seq:
            correct_grammar += 1

        if not is_sequence_only and "pred_symmetry" in aux_preds:
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

        disc_d, _ = fast_search_depth(
            parsed,
            target_image=img_raw,
            angle=angle,
            symmetry=sym,
            grid_size=grid,
            motif=rec.get("motif", "M1"),
            search_metric="ncc",
        )
        if disc_d == true_d:
            correct_depth += 1

    total = len(dataset)
    return {
        "exact_grammar_match_rate": float(correct_grammar / total),
        "protocol_b_depth_accuracy": float(correct_depth / total),
        "syntactic_validity_rate": float(syntactic_valid / total),
    }


def main() -> None:
    print("=" * 75)
    print("KOLAM-R Stage 8 — Comprehensive & Verified Ablation Benchmark")
    print("=" * 75)

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

    history_log = {}

    # 1. Train Sequence-Only Variant (100% Data, 4 Motifs)
    m_seq_only = VisionToGrammarSequenceOnly(d_model=256).to(device)
    _, seq_train_loss, seq_val_loss = train_and_log_variant("Sequence-Only (No Aux Loss)", m_seq_only, ds_train_100, ds_val, epochs=20, has_aux=False, device=device)
    history_log["Sequence-Only"] = {"train": seq_train_loss, "val": seq_val_loss}

    # 2. Train ResNet Backbone Variant (100% Data, 4 Motifs)
    m_resnet = VisionToGrammarResNet(d_model=256).to(device)
    _, resnet_train_loss, resnet_val_loss = train_and_log_variant("ResNet-18 Backbone", m_resnet, ds_train_100, ds_val, epochs=20, has_aux=True, device=device)
    history_log["ResNet-18"] = {"train": resnet_train_loss, "val": resnet_val_loss}

    # 3. Train 50% Data Volume Model (50% Data, Full 4-Motif Diversity)
    m_50pct = VisionToGrammarModel(d_model=256).to(device)
    _, p50_train_loss, p50_val_loss = train_and_log_variant("50% Data Scale (4 Motifs)", m_50pct, ds_train_50, ds_val, epochs=20, has_aux=True, device=device)
    history_log["50% Data"] = {"train": p50_train_loss, "val": p50_val_loss}

    # 4. Train Single-Motif Model (M1-Only, 25% Data)
    m_m1 = VisionToGrammarModel(d_model=256).to(device)
    _, m1_train_loss, m1_val_loss = train_and_log_variant("Single-Motif (M1-Only)", m_m1, ds_train_m1, ds_val, epochs=20, has_aux=True, device=device)
    history_log["Single-Motif M1"] = {"train": m1_train_loss, "val": m1_val_loss}

    # 5. Load Canonical Full Model (100% Data, 4 Motifs)
    m_full = VisionToGrammarModel(d_model=256).to(device)
    ckpt_full = torch.load("checkpoints/best_grammar_model.pt", map_location="cpu", weights_only=True)
    m_full.load_state_dict(ckpt_full["model_state_dict"])

    # Benchmark Evaluations
    print("\n--- Running Systematic Evaluations ---")
    eval_full_iid = evaluate_model_on_dataset(m_full, test_iid, device=device)
    eval_full_held = evaluate_model_on_dataset(m_full, test_heldout, device=device)

    eval_seq_iid = evaluate_model_on_dataset(m_seq_only, test_iid, device=device, is_sequence_only=True)
    eval_seq_held = evaluate_model_on_dataset(m_seq_only, test_heldout, device=device, is_sequence_only=True)

    eval_res_iid = evaluate_model_on_dataset(m_resnet, test_iid, device=device)
    eval_res_held = evaluate_model_on_dataset(m_resnet, test_heldout, device=device)

    eval_50pct_iid = evaluate_model_on_dataset(m_50pct, test_iid, device=device)
    eval_50pct_held = evaluate_model_on_dataset(m_50pct, test_heldout, device=device)

    # Cross-Motif Evaluations for Single-Motif Model
    m1_subset_indices = [i for i, r in enumerate(test_iid.records) if r.get("motif") == "M1"]
    unseen_subset_indices = [i for i, r in enumerate(test_iid.records) if r.get("motif") != "M1"]

    ds_m1_seen = KolamDataset(splits_root / "test_iid.json")
    ds_m1_seen.records = [test_iid.records[i] for i in m1_subset_indices]
    ds_m1_unseen = KolamDataset(splits_root / "test_iid.json")
    ds_m1_unseen.records = [test_iid.records[i] for i in unseen_subset_indices]

    eval_m1_seen = evaluate_model_on_dataset(m_m1, ds_m1_seen, device=device)
    eval_m1_unseen = evaluate_model_on_dataset(m_m1, ds_m1_unseen, device=device)

    # Depth Search Sensitivity: Tested on (A) Oracle Grammar and (B) Predicted Grammar
    print("Evaluating Protocol B Search Metric Sensitivity (Oracle vs Predicted Grammar)...")
    tokenizer = GrammarTokenizer()
    executor = GrammarExecutor()

    search_sensitivity_oracle = {}
    search_sensitivity_predicted = {}

    for sm in ["ncc", "ssim", "iou", "mse"]:
        corr_oracle = 0
        corr_pred = 0
        for idx in range(len(test_iid)):
            rec = test_iid.records[idx]
            img_raw = test_iid.load_image(idx)
            true_d = int(rec["recursion_depth"])
            r_id = rec["production_rule_id"]

            # Oracle Grammar Search
            oracle_seq = tokenizer.rule_to_grammar_string(RULES_BY_ID[r_id])
            oracle_parsed = executor.parse_grammar_string(oracle_seq)
            d_o, _ = fast_search_depth(oracle_parsed, img_raw, float(rec.get("angle", 90.0)), rec.get("symmetry", "C1"), int(rec.get("grid_size", 5)), rec.get("motif", "M1"), search_metric=sm)
            if d_o == true_d:
                corr_oracle += 1

            # Predicted Grammar Search (using Full Model predictions)
            img_t = torch.from_numpy(img_raw.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                gen_toks, aux = m_full.generate(img_t)
            gen_str = tokenizer.decode(gen_toks[0].cpu().tolist())
            pred_parsed = executor.parse_grammar_string(gen_str)
            d_p, _ = fast_search_depth(pred_parsed, img_raw, float(rec.get("angle", 90.0)), rec.get("symmetry", "C1"), int(rec.get("grid_size", 5)), rec.get("motif", "M1"), search_metric=sm)
            if d_p == true_d:
                corr_pred += 1

        search_sensitivity_oracle[sm] = float(corr_oracle / len(test_iid))
        search_sensitivity_predicted[sm] = float(corr_pred / len(test_iid))

    # Compile Final JSON Results
    results_compiled = {
        "ablation_1_auxiliary_loss": {
            "full_multitask_model": {"test_iid": eval_full_iid, "test_heldout_depth": eval_full_held},
            "sequence_only_model": {"test_iid": eval_seq_iid, "test_heldout_depth": eval_seq_held},
        },
        "ablation_2_visual_encoder": {
            "convnet_4layer_canonical": {"test_iid": eval_full_iid, "test_heldout_depth": eval_full_held},
            "resnet_backbone": {"test_iid": eval_res_iid, "test_heldout_depth": eval_res_held},
        },
        "ablation_3_data_scale_and_motif": {
            "full_100pct_data_4motifs": eval_full_iid,
            "subsampled_50pct_data_4motifs": eval_50pct_iid,
            "single_motif_m1_seen_eval": eval_m1_seen,
            "single_motif_m1_unseen_motifs_transfer": eval_m1_unseen,
        },
        "ablation_4_search_metrics": {
            "oracle_grammar": search_sensitivity_oracle,
            "predicted_grammar_end_to_end": search_sensitivity_predicted,
        },
        "training_trajectories": history_log,
    }

    out_json = Path("results/ablation_evaluation.json")
    with open(out_json, "w") as f:
        json.dump(results_compiled, f, indent=2)

    # Plot Multi-Panel Loss Curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=200, constrained_layout=True)

    # Panel 1: Sequence Only vs Full Multi-Task
    epochs_20 = list(range(1, 21))
    axes[0, 0].plot(epochs_20, history_log["Sequence-Only"]["train"], label="Seq-Only Train", color="crimson", linestyle="--")
    axes[0, 0].plot(epochs_20, history_log["Sequence-Only"]["val"], label="Seq-Only Val", color="darkred")
    axes[0, 0].set_title("Ablation 1: Sequence-Only Loss Trajectory", fontsize=10, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch", fontsize=8)
    axes[0, 0].set_ylabel("Loss", fontsize=8)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: ResNet-18 Backbone
    axes[0, 1].plot(epochs_20, history_log["ResNet-18"]["train"], label="ResNet Train", color="blue", linestyle="--")
    axes[0, 1].plot(epochs_20, history_log["ResNet-18"]["val"], label="ResNet Val", color="navy")
    axes[0, 1].set_title("Ablation 2: ResNet-18 Loss Trajectory", fontsize=10, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch", fontsize=8)
    axes[0, 1].set_ylabel("Loss", fontsize=8)
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Data Scale & Motif Diversity
    axes[1, 0].plot(epochs_20, history_log["50% Data"]["val"], label="50% Data (4 Motifs) Val", color="orange")
    axes[1, 0].plot(epochs_20, history_log["Single-Motif M1"]["val"], label="25% Data (M1-Only) Val", color="purple")
    axes[1, 0].set_title("Ablation 3: Data Scale & Motif Generalization", fontsize=10, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch", fontsize=8)
    axes[1, 0].set_ylabel("Validation Loss", fontsize=8)
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 4: Protocol B Metric Sensitivity
    metrics = ["NCC", "SSIM", "IoU", "MSE"]
    oracle_scores = [search_sensitivity_oracle[m.lower()] * 100 for m in metrics]
    pred_scores = [search_sensitivity_predicted[m.lower()] * 100 for m in metrics]
    x_pos = np.arange(len(metrics))
    width = 0.35
    axes[1, 1].bar(x_pos - width/2, oracle_scores, width, label="Oracle Grammar", color="teal", alpha=0.8)
    axes[1, 1].bar(x_pos + width/2, pred_scores, width, label="Predicted Grammar", color="coral", alpha=0.8)
    axes[1, 1].set_title("Ablation 4: Protocol B Search Metric Accuracy (%)", fontsize=10, fontweight="bold")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].set_ylabel("Depth Accuracy (%)", fontsize=8)
    axes[1, 1].set_ylim(0, 115)
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("KOLAM-R Stage 8 — Systematic Ablation Trajectories & Invariances", fontsize=13, fontweight="bold")
    fig.savefig("results/ablation_curves.png", bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 75)
    print("STAGE 8 COMPREHENSIVE ABLATION BENCHMARK SUMMARY")
    print("=" * 75)
    print(f"1. Loss Grounding: Full Multi-Task ({eval_full_iid['exact_grammar_match_rate']*100:.1f}%) vs Seq-Only ({eval_seq_iid['exact_grammar_match_rate']*100:.1f}%)")
    print(f"2. Visual Encoder: ConvNet ({eval_full_iid['exact_grammar_match_rate']*100:.1f}%) vs ResNet-18 ({eval_res_iid['exact_grammar_match_rate']*100:.1f}%)")
    print(f"3. Data Scale vs Motif: 100% Data ({eval_full_iid['exact_grammar_match_rate']*100:.1f}%) | 50% Data 4-Motifs ({eval_50pct_iid['exact_grammar_match_rate']*100:.1f}%) | Single-Motif Unseen ({eval_m1_unseen['exact_grammar_match_rate']*100:.1f}%)")
    print(f"4. Search Metric (Oracle): NCC {search_sensitivity_oracle['ncc']*100:.1f}% | SSIM {search_sensitivity_oracle['ssim']*100:.1f}% | IoU {search_sensitivity_oracle['iou']*100:.1f}% | MSE {search_sensitivity_oracle['mse']*100:.1f}%")
    print(f"   Search Metric (Pred):   NCC {search_sensitivity_predicted['ncc']*100:.1f}% | SSIM {search_sensitivity_predicted['ssim']*100:.1f}% | IoU {search_sensitivity_predicted['iou']*100:.1f}% | MSE {search_sensitivity_predicted['mse']*100:.1f}%")

    print("\nArtifacts Saved:")
    print("  - results/ablation_curves.png")
    print("  - results/ablation_evaluation.json")


if __name__ == "__main__":
    main()
