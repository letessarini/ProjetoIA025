from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from copy import deepcopy
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency for import-time flexibility
    torch = None
    nn = None


def _no_grad_decorator():
    if torch is None:
        def identity(func):
            return func
        return identity
    return torch.no_grad()


# Use a safe decorator so module import does not crash when torch is unavailable.
if torch is not None:
    _no_grad = torch.no_grad()
else:
    def _no_grad(func):
        return func


@dataclass
class TrainingConfig:
    model_name: str = "unet"
    device: str = "auto"
    epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    limit: Optional[int] = None
    seed: int = 20261022
    output_dir: str = "results"
    checkpoint_name: str = "last_model.pt"
    use_valid_mask: bool = True
    train_eval_ratio: float = 0.8
    model_config: Optional[Dict[str, Any]] = None
    thresholds: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    loss_name: str = "weighted_bce"
    loss_config: Optional[Dict[str, Any]] = None
    oversample_positive_patches: bool = True
    positive_oversample_ratio: float = 0.7
    min_positive_fraction: float = 1e-6
    input_channels: Optional[int] = None
    train_patch_limit: Optional[int] = None
    val_patch_limit: Optional[int] = None
    eval_patch_limit: Optional[int] = None
    split_strategy: str = "random_patch"
    split_seed: int = 20261022
    selection_metric: str = "f2"
    threshold_bins: Optional[int] = None
    early_stopping_patience: Optional[int] = None
    cosine_schedule: bool = False
    mixed_precision: bool = False
    gradient_clip: Optional[float] = None
    min_positive_pixels: Optional[int] = None
    evaluate_train_set: bool = True
    save_validation_predictions: bool = True

    @property
    def resolved_device(self) -> str:
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"


class PositiveOversamplingDataset:
    """Sample positive patches more often during training to match the legacy flood-segmentation logic."""

    def __init__(self, patches: Sequence[Dict[str, Any]], positive_ratio: float = 0.7, min_positive_fraction: float = 1e-6, min_positive_pixels: Optional[int] = None):
        self.patches = list(patches)
        self.positive_ratio = max(0.0, min(float(positive_ratio), 1.0))
        self.min_positive_fraction = max(0.0, float(min_positive_fraction))
        self.positive_index = [
            self._is_positive_patch(p, self.min_positive_fraction, min_positive_pixels)
            for p in self.patches
        ]

    @staticmethod
    def _is_positive_patch(patch, min_positive_fraction=1e-6, min_positive_pixels=None):
        if "y" not in patch:
            with np.load(patch["path"], allow_pickle=False) as data:
                return PositiveOversamplingDataset._is_positive_patch(data, min_positive_fraction, min_positive_pixels)
        y = np.asarray(patch.get("y", []), dtype=np.float32)
        valid = np.asarray(
            patch.get("valido", np.ones_like(y)),
            dtype=np.float32
        ) > 0.5

        if y.size == 0 or valid.sum() == 0:
            return False

        count = ((y > 0.5) & valid).sum()
        if min_positive_pixels is not None:
            return bool(count >= min_positive_pixels)
        positive_fraction = count / valid.sum()

        return bool(positive_fraction > max(0.0, min_positive_fraction))

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx):
        return _patch_arrays(self.patches[idx])

    def sampler_weights(self):
        if not self.patches:
            return np.array([], dtype=np.float32)

        positive_count = sum(self.positive_index)
        negative_count = len(self.patches) - positive_count

        if positive_count == 0 or negative_count == 0:
            return np.ones(len(self.patches), dtype=np.float32)

        positive_weight = self.positive_ratio / positive_count
        negative_weight = (1.0 - self.positive_ratio) / negative_count

        return np.asarray(
            [
                positive_weight if is_positive else negative_weight
                for is_positive in self.positive_index
            ],
            dtype=np.float32,
        )


class PatchDataset:
    """Simple dataset wrapper built from patch dictionaries."""

    def __init__(self, patches: Sequence[Dict[str, Any]]):
        self.patches = list(patches)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        return _patch_arrays(self.patches[idx])


def _patch_arrays(patch):
    if "X" not in patch:
        with np.load(patch["path"], allow_pickle=False) as data:
            return _patch_arrays(data)
    return tuple(np.asarray(patch[name], dtype=np.float32) for name in ("X", "y", "valido"))


def _prepare_batch(batch: Iterable[Tuple[np.ndarray, np.ndarray, np.ndarray]], device: Any):
    xs, ys, valids = zip(*batch)
    X = torch.stack([torch.from_numpy(np.asarray(x)) for x in xs]).to(device)
    y = torch.stack([torch.from_numpy(np.asarray(v)) for v in ys]).to(device)
    valid = torch.stack([torch.from_numpy(np.asarray(v)) for v in valids]).to(device)

    if X.dim() == 3:
        X = X.unsqueeze(1)
    if y.dim() == 2:
        y = y.unsqueeze(1)
    if valid.dim() == 2:
        valid = valid.unsqueeze(1)

    return X, y, valid


def build_loader(
    patches: Sequence[Dict[str, Any]],
    batch_size: int,
    shuffle: bool = False,
    oversample_positive_patches: bool = False,
    positive_oversample_ratio: float = 0.7,
    min_positive_fraction: float = 1e-6,
    seed: int = 20261022,
    min_positive_pixels: Optional[int] = None,
):
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to train the model.")

    if oversample_positive_patches:
        dataset = PositiveOversamplingDataset(
            patches,
            positive_ratio=positive_oversample_ratio,
            min_positive_fraction=min_positive_fraction,
            min_positive_pixels=min_positive_pixels,
        )
        weights = dataset.sampler_weights()
        positive_mask = np.array(dataset.positive_index, dtype=bool)

        print("\n--- DIAGNÓSTICO OVERSAMPLING ---")
        print(f"Total de patches: {len(dataset)}")
        print(f"Patches positivos: {positive_mask.sum()}")
        print(f"Patches negativos: {(~positive_mask).sum()}")
        print(f"Proporção positiva original: {positive_mask.mean():.4f}")

        positive_weight_sum = weights[positive_mask].sum()
        negative_weight_sum = weights[~positive_mask].sum()

        print(f"Peso total positivo: {positive_weight_sum:.4f}")
        print(f"Peso total negativo: {negative_weight_sum:.4f}")
        print(f"Proporção positiva esperada: {positive_weight_sum / weights.sum():.4f}")
        print("--------------------------------\n")
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights,
            num_samples=max(1, len(weights)),
            replacement=True,
            generator=torch.Generator().manual_seed(seed),
        )
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    dataset = PatchDataset(patches)
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _aligned_valid_mask(target, valid_mask):
    """Return a float mask with invalid pixels forced to zero and valid pixels to one.

    This preserves the intended semantics of `valido`: pixels outside the city are
    ignored, not treated as negative background.
    """
    if valid_mask is None:
        return torch.ones_like(target, dtype=torch.float32)

    valid_mask = valid_mask.to(device=target.device, dtype=torch.float32)
    if valid_mask.dim() == 2:
        valid_mask = valid_mask.unsqueeze(0).unsqueeze(0)
    elif valid_mask.dim() == 3:
        valid_mask = valid_mask.unsqueeze(1)

    if target.shape != valid_mask.shape:
        # Match the channel convention used by the model: [B, 1, H, W]
        if target.shape[0] == valid_mask.shape[0] and target.shape[-2:] == valid_mask.shape[-2:]:
            valid_mask = valid_mask[:, :1, :, :]
        else:
            valid_mask = valid_mask[:, : target.shape[1], :, :]

    return (valid_mask > 0.5).to(torch.float32)


def _align_target_shape(logits, target):
    if target.dim() == 2:
        target = target.unsqueeze(0).unsqueeze(0)
    elif target.dim() == 3:
        target = target.unsqueeze(1)

    if target.shape != logits.shape:
        if target.shape[0] == logits.shape[0] and target.shape[-2:] == logits.shape[-2:]:
            target = target[:, :1, :, :]
        else:
            target = target[:, : logits.shape[1], :, :]
    return target


def _safe_masked_mean(loss, valid_mask):
    denom = valid_mask.sum().clamp_min(1.0)
    return (loss * valid_mask).sum() / denom


def compute_masked_bce_loss(logits, target, valid_mask=None):
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to train the model.")

    if logits.dim() == 2:
        logits = logits.unsqueeze(0).unsqueeze(0)

    target = _align_target_shape(logits, target).to(logits.device)
    valid_mask = _aligned_valid_mask(target, valid_mask).to(logits.device)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return _safe_masked_mean(loss, valid_mask)


def compute_masked_weighted_bce_loss(logits, target, valid_mask=None, pos_weight: Optional[float] = None):
    """Weighted BCE with logits, masking invalid pixels and optionally rebiasing for class imbalance."""
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to train the model.")

    if logits.dim() == 2:
        logits = logits.unsqueeze(0).unsqueeze(0)

    target = _align_target_shape(logits, target).to(logits.device)
    valid_mask = _aligned_valid_mask(target, valid_mask).to(logits.device)

    if pos_weight is None:
        total_valid = valid_mask.sum().clamp_min(1.0)
        positive_mass = (target * valid_mask).sum() / total_valid
        negative_mass = ((1.0 - target) * valid_mask).sum() / total_valid
        pos_weight = (negative_mass / positive_mass.clamp_min(1e-6)).clamp_min(1.0)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits,
        target,
        reduction="none",
        pos_weight=torch.full_like(logits, float(pos_weight), device=logits.device),
    )
    return _safe_masked_mean(loss, valid_mask)


def compute_masked_tversky_loss(logits, target, valid_mask=None, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6, reduction: str = "per_patch"):
    """Tversky loss on valid pixels only: a common sparse-segmentation objective.

    Tversky index = TP / (TP + alpha * FP + beta * FN), so beta > alpha makes false negatives more costly.
    This is helpful for flood segmentation, where false negatives can be more expensive than false positives.
    """
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to train the model.")

    if logits.dim() == 2:
        logits = logits.unsqueeze(0).unsqueeze(0)

    target = _align_target_shape(logits, target).to(logits.device)
    valid_mask = _aligned_valid_mask(target, valid_mask).to(logits.device)

    probs = torch.sigmoid(logits)
    probs = probs * valid_mask
    target_masked = target * valid_mask

    tp = (probs * target_masked).sum(dim=(1, 2, 3))
    fp = (probs * (1.0 - target_masked)).sum(dim=(1, 2, 3))
    fn = ((1.0 - probs) * target_masked).sum(dim=(1, 2, 3))

    if reduction == "batch":
        tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
    elif reduction != "per_patch":
        raise ValueError(f"Unknown Tversky reduction: {reduction}")
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return (1.0 - tversky).mean()


def build_loss(loss_name: str = "weighted_bce", loss_config: Optional[Dict[str, Any]] = None):
    """Return a loss callable that can be configured via config JSON or CLI overrides."""
    normalized = (loss_name or "weighted_bce").strip().lower().replace("-", "_")
    cfg = dict(loss_config or {})

    if normalized in {"bce", "binary_cross_entropy", "binary_cross_entropy_with_logits"}:
        return compute_masked_bce_loss

    if normalized in {"weighted_bce", "weighted_binary_cross_entropy", "wbce"}:
        def _loss(logits, target, valid_mask=None):
            pos_weight = cfg.get("pos_weight")
            if pos_weight is None:
                pos_weight = cfg.get("positive_weight")
            return compute_masked_weighted_bce_loss(logits, target, valid_mask, pos_weight=float(pos_weight) if pos_weight is not None else None)
        return _loss

    if normalized in {"tversky", "tversky_loss", "tv"}:
        alpha = float(cfg.get("alpha", 0.3))
        beta = float(cfg.get("beta", 0.7))
        eps = float(cfg.get("eps", 1e-6))

        def _loss(logits, target, valid_mask=None):
            return compute_masked_tversky_loss(logits, target, valid_mask, alpha=alpha, beta=beta, eps=eps, reduction=cfg.get("tversky_reduction", "per_patch"))

        return _loss

    if normalized in {"hybrid", "bce_tversky", "hybrid_bce_tversky"}:
        alpha = float(cfg.get("alpha", 0.3))
        beta = float(cfg.get("beta", 0.7))
        bce_weight = float(cfg.get("bce_weight", 0.5))
        tversky_weight = float(cfg.get("tversky_weight", 0.5))
        if bce_weight < 0 or tversky_weight < 0 or bce_weight + tversky_weight <= 0:
            raise ValueError("Hybrid weights must be nonnegative and not both zero")

        def _loss(logits, target, valid_mask=None):
            bce = compute_masked_bce_loss(logits, target, valid_mask)
            tversky = compute_masked_tversky_loss(logits, target, valid_mask, alpha=alpha, beta=beta, reduction=cfg.get("tversky_reduction", "per_patch"))
            return (bce_weight * bce + tversky_weight * tversky) / (bce_weight + tversky_weight)

        return _loss

    if normalized == "bce_dice":
        def _loss(logits, target, valid_mask=None):
            return compute_masked_bce_loss(logits, target, valid_mask) + compute_masked_tversky_loss(
                logits, target, valid_mask, alpha=0.5, beta=0.5
            )
        return _loss

    if normalized in {"focal", "focal_loss"}:
        gamma = float(cfg.get("gamma", 2.0))
        alpha = float(cfg.get("alpha", 0.25))

        def _loss(logits, target, valid_mask=None):
            if valid_mask is None:
                valid_mask = torch.ones_like(target)
            if logits.dim() == 2:
                logits = logits.unsqueeze(0).unsqueeze(0)
            if target.dim() == 2:
                target = target.unsqueeze(1)
            elif target.dim() == 3:
                target = target.unsqueeze(1)
            if valid_mask.dim() == 2:
                valid_mask = valid_mask.unsqueeze(1)
            elif valid_mask.dim() == 3:
                valid_mask = valid_mask.unsqueeze(1)

            logits = logits.to(target.device)
            target = target.to(logits.device)
            valid_mask = valid_mask.to(logits.device)

            probs = torch.sigmoid(logits)
            ce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
            p_t = probs * target + (1.0 - probs) * (1.0 - target)
            alpha_factor = target * alpha + (1.0 - target) * (1.0 - alpha)
            modulating = (1.0 - p_t).clamp_min(1e-6) ** gamma
            loss = alpha_factor * modulating * ce
            loss = loss * valid_mask
            return loss.sum() / (valid_mask.sum().clamp_min(1.0))

        return _loss

    raise ValueError(f"Unsupported loss '{loss_name}'. Supported: bce, weighted_bce, tversky, focal, hybrid, bce_dice")


def _print_progress(label: str, current: int, total: int) -> None:
    if total <= 0:
        return
    percent = current / total
    filled = int(30 * percent)
    bar = "#" * filled + "-" * (30 - filled)
    print(f"{label}: [{bar}] {current}/{total} ({percent * 100:5.1f}%)", end="\r")
    if current >= total:
        print()


def _ensure_segmentation_shape(tensor, batch_size: Optional[int] = None):
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(1)
    if tensor.dim() == 3 and batch_size is not None:
        tensor = tensor.unsqueeze(1)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    if batch_size is not None and tensor.shape[0] != batch_size:
        tensor = tensor[:batch_size]
    return tensor


from pipeline.metrics.segmentation_metrics import compute_histogram_metrics, compute_metrics



def _threshold_metrics(tp, fp, fn, tn, eps: float = 1e-9) -> Dict[str, float]:
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)

    f1 = 2 * precision * recall / (precision + recall + eps)
    f2 = 5 * precision * recall / (4 * precision + recall + eps)

    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "f2": float(f2),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }



def _compute_confusion_from_logits(logits, target, valid_mask=None, threshold: float = 0.5):
    """Return TP, FP, FN, TN over valid pixels after thresholding probabilities."""
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for metric computation.")

    logits = logits.detach().cpu()
    target = target.detach().cpu()
    if valid_mask is not None:
        valid_mask = valid_mask.detach().cpu()

    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).to(torch.uint8)
    target = target.to(torch.uint8)

    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)

    if valid_mask is not None:
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        valid_mask = valid_mask.to(torch.bool)
        pred = pred[valid_mask]
        target = target[valid_mask]

    pred = pred.reshape(-1)
    target = target.reshape(-1)

    tp = int(torch.sum((pred == 1) & (target == 1)).item())
    fp = int(torch.sum((pred == 1) & (target == 0)).item())
    fn = int(torch.sum((pred == 0) & (target == 1)).item())
    tn = int(torch.sum((pred == 0) & (target == 0)).item())
    return tp, fp, fn, tn


def _evaluate_threshold_sweep(model, loader, device, thresholds: Sequence[float], input_channels: int = 10, selection_metric: str = "f2"):
    model.eval()
    thresholds = [float(t) for t in thresholds]
    if not thresholds:
        raise ValueError("At least one threshold is required")
    counts = {t: [0, 0, 0, 0] for t in thresholds}
    with torch.no_grad():
        for X, y, valid in loader:
            X = X.to(device)[:, :input_channels]
            y = _ensure_segmentation_shape(y.to(device), batch_size=X.shape[0])
            valid = _ensure_segmentation_shape(valid.to(device), batch_size=X.shape[0])
            logits = model(X)
            for threshold in thresholds:
                confusion = _compute_confusion_from_logits(logits, y, valid, threshold=threshold)
                counts[threshold] = [a + b for a, b in zip(counts[threshold], confusion)]
    threshold_metrics = [
        {"threshold": t, "metrics": _threshold_metrics(*counts[t])}
        for t in thresholds
    ]
    best = max(threshold_metrics, key=lambda item: (item["metrics"][selection_metric], item["metrics"]["precision"]))
    return threshold_metrics, best


def _add_score_histogram(hist, logits, target, valid):
    select = valid.detach().cpu().numpy() > 0.5
    scores = torch.sigmoid(logits.float()).detach().cpu().numpy()[select]
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite model scores on valid pixels")
    labels = target.detach().cpu().numpy()[select] > 0.5
    bins = np.minimum((scores.astype(np.float64) * (hist.shape[1] - 1)).astype(int), hist.shape[1] - 1)
    hist[0] += np.bincount(bins[~labels], minlength=hist.shape[1])
    hist[1] += np.bincount(bins[labels], minlength=hist.shape[1])


class GenericTrainer:
    def __init__(self, model, train_loader, val_loader, config: TrainingConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.resolved_device)
        self.model.to(self.device)
        self.best_state = None
        self.best_epoch = None
        self.best_val_f2 = -1.0
        self.best_score = -1.0
        self.best_threshold = None
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.criterion = build_loss(config.loss_name, config.loss_config)
        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=config.epochs)
            if config.cosine_schedule else None
        )
        self.amp = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=True) if self.amp else None

    def _forward(self, inputs):
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.amp):
            return self.model(inputs).float()

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        data_len = len(self.train_loader)
        print(f"[Stage 2/2] Modelling: training epoch {self._current_epoch}/{self.config.epochs}")

        for batch_idx, (X, y, valid) in enumerate(self.train_loader, start=1):
            X = X.to(self.device)[:, :self.config.input_channels]
            y = _ensure_segmentation_shape(y.to(self.device), batch_size=X.shape[0])
            valid = _ensure_segmentation_shape(valid.to(self.device), batch_size=X.shape[0])

            logits = self._forward(X)
            loss = self.criterion(logits, y, valid)

            self.optimizer.zero_grad(set_to_none=True)
            if self.scaler is None:
                loss.backward()
            else:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
            if self.config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            if self.scaler is None:
                self.optimizer.step()
            else:
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss += float(loss.item()) * X.size(0)
            total_samples += X.size(0)
            _print_progress("Training batches", batch_idx, data_len)

        return total_loss / max(1, total_samples)

    def _aggregate_dataset_metrics(self, loader):
        self.model.eval()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for X, y, valid in loader:
                X = X.to(self.device)[:, :self.config.input_channels]
                y = _ensure_segmentation_shape(y.to(self.device), batch_size=X.shape[0])
                valid = _ensure_segmentation_shape(valid.to(self.device), batch_size=X.shape[0])
                logits = self.model(X)
                batch_tp, batch_fp, batch_fn, batch_tn = _compute_confusion_from_logits(logits, y, valid, threshold=0.5)
                tp += batch_tp
                fp += batch_fp
                fn += batch_fn
                tn += batch_tn
        return compute_metrics(tp, fp, fn, tn)

    def _eval_epoch(self):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        data_len = len(self.val_loader)
        tp = fp = fn = tn = 0
        hist = None if self.config.threshold_bins is None else np.zeros((2, self.config.threshold_bins + 1), dtype=np.int64)

        for batch_idx, (X, y, valid) in enumerate(self.val_loader, start=1):
            X = X.to(self.device)[:, :self.config.input_channels]
            y = _ensure_segmentation_shape(y.to(self.device), batch_size=X.shape[0])
            valid = _ensure_segmentation_shape(valid.to(self.device), batch_size=X.shape[0])

            with torch.no_grad():
                logits = self._forward(X)
                loss = self.criterion(logits, y, valid)

            total_loss += float(loss.item()) * X.size(0)
            total_samples += X.size(0)

            if hist is None:
                batch_tp, batch_fp, batch_fn, batch_tn = _compute_confusion_from_logits(logits, y, valid, threshold=0.5)
                tp += batch_tp
                fp += batch_fp
                fn += batch_fn
                tn += batch_tn
            else:
                _add_score_histogram(hist, logits, y, valid)
            _print_progress("Validation batches", batch_idx, data_len)

        if hist is None:
            base_metrics = compute_metrics(tp, fp, fn, tn)
            threshold_metrics, best_threshold_metrics = _evaluate_threshold_sweep(
                self.model, self.val_loader, self.device, self.config.thresholds,
                self.config.input_channels, self.config.selection_metric
            )
        else:
            base_metrics = compute_histogram_metrics(hist, 0.5)["metrics"]
            threshold_metrics = [compute_histogram_metrics(hist, t) for t in self.config.thresholds]
            best_threshold_metrics = compute_histogram_metrics(hist, selection_metric=self.config.selection_metric)
        return total_loss / max(1, total_samples), base_metrics, threshold_metrics, best_threshold_metrics

    def evaluate_holdout(self, loader, threshold):
        hist = np.zeros((2, (self.config.threshold_bins or 1000) + 1), dtype=np.int64)
        self.model.eval()
        with torch.no_grad():
            for X, y, valid in loader:
                logits = self._forward(X[:, :self.config.input_channels].to(self.device))
                _add_score_histogram(hist, logits, y.unsqueeze(1), valid.unsqueeze(1))
        return compute_histogram_metrics(hist, threshold)

    def train(self, on_improvement=None):
        history = []
        for epoch in range(self.config.epochs):
            self._current_epoch = epoch + 1
            train_loss = self._train_epoch()
            val_loss, val_metrics, val_threshold_metrics, best_threshold_metrics = self._eval_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
            train_metrics = self._aggregate_dataset_metrics(self.train_loader) if self.config.evaluate_train_set else None

            score = float(best_threshold_metrics["metrics"][self.config.selection_metric])
            improved = score > self.best_score
            if improved:
                self.best_score = score
                self.best_val_f2 = float(best_threshold_metrics["metrics"]["f2"])
                self.best_epoch = epoch + 1
                self.best_state = deepcopy(self.model.state_dict())
                self.best_threshold = best_threshold_metrics["threshold"]

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "threshold_sweep": val_threshold_metrics,
                "best_threshold": best_threshold_metrics,
            })
            if improved and on_improvement is not None:
                on_improvement(history)
            
            print(
                f"epoch {epoch + 1}/{self.config.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_iou={val_metrics['iou']:.4f} | "
                f"val_precision={val_metrics['precision']:.4f} | "
                f"val_recall={val_metrics['recall']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "
                f"val_f2={_threshold_metrics(val_metrics['tp'], val_metrics['fp'], val_metrics['fn'], val_metrics['tn'])['f2']:.4f} | "
                f"best_threshold={best_threshold_metrics['threshold']:.3f} "
                f"(iou={best_threshold_metrics['metrics']['iou']:.4f}, "
                f"f2={best_threshold_metrics['metrics']['f2']:.4f}, "
                f"recall={best_threshold_metrics['metrics']['recall']:.4f}) | "
                f"predicted={val_metrics['frac_predicted']:.4f} | "
                f"prevalence={val_metrics['prevalence']:.4f} | "
                f"device={self.device}"
            )
            patience = self.config.early_stopping_patience
            if patience is not None and epoch + 1 - self.best_epoch >= patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        return history


def _save_run_metadata(run_dir: Path, model_name: str, config: TrainingConfig, history: List[Dict[str, Any]], device: str):
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "device": device,
        "config": {k: v for k, v in config.__dict__.items() if v is not None},
        "history": history,
    }
    (run_dir / "config.json").write_text(json.dumps(metadata["config"], indent=2), encoding="utf-8")
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _patch_name(patch):
    name = patch.get("meta", {}).get("name")
    return Path(name or patch["path"]).name


def _save_predictions(model, patches: Sequence[Dict[str, Any]], run_dir: Path, device: Any, input_channels: int = 10):
    model.eval()
    preds_dir = run_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    for patch in patches:
        features, target, valid = _patch_arrays(patch)
        x = torch.from_numpy(features[:input_channels]).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]

        file_name = Path(_patch_name(patch))
        out_path = preds_dir / f"{file_name.stem}_prediction.npz"
        np.savez_compressed(
            out_path,
            prediction=prob,
            target=target,
            valid=valid,
            name=str(file_name),
        )

    summary = {"n_predictions": len(patches), "predictions_dir": str(preds_dir)}
    (run_dir / "predictions_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_training_model(model_name: str, dataset: List[Dict[str, Any]], config: Optional[TrainingConfig] = None, manifest=None):
    """Train a generic model from a list of prepared patches."""
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to run model training.")

    config = config or TrainingConfig(model_name=model_name)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    if config.selection_metric not in {"iou", "f2"}:
        raise ValueError("selection_metric must be iou or f2")
    if config.threshold_bins is not None and config.threshold_bins < 2:
        raise ValueError("threshold_bins must be at least 2")
    if config.epochs < 1:
        raise ValueError("epochs must be positive")
    if config.early_stopping_patience is not None and config.early_stopping_patience < 1:
        raise ValueError("early_stopping_patience must be positive")
    if not dataset:
        raise ValueError("Training dataset is empty")
    available_channels = _patch_arrays(dataset[0])[0].shape[0]
    if config.input_channels is None:
        config = replace(config, input_channels=available_channels)
    if not 1 <= config.input_channels <= available_channels:
        raise ValueError(f"input_channels={config.input_channels} invalid for dataset")
    rng = np.random.RandomState(config.seed)
    holdouts = {}
    if config.split_strategy == "date_region":
        from pipeline.data_preparation.dataset_splitter import conferir_vazamento, montar_split

        if manifest is None:
            raise ValueError("date_region requires a dataset manifest")
        if config.limit is not None:
            raise ValueError("Use train_patch_limit with date_region to preserve holdouts")
        rainfall = {item["data"]: item["chuva_total_mm_media"] for item in manifest["datas"]}
        by_path = {patch["path"]: patch for patch in dataset}
        items = []
        for path in sorted(by_path):
            date, row, col = Path(path).stem.split("_")
            items.append((path, date, int(row[1:]), int(col[1:]), rainfall[date]))
        groups, date_roles, region_roles = montar_split(items, semente=config.split_seed)
        issues = conferir_vazamento(groups, items, date_roles, region_roles)
        if issues:
            raise ValueError(f"Invalid grouped split: {issues}")
        train_patches = [by_path[path] for path in groups.get("treino", [])]
        val_patches = [by_path[path] for path in groups.get("validacao", [])]
        holdouts = {key: [by_path[path] for path in paths]
                    for key, paths in groups.items() if key.startswith("teste_")}
        if config.train_patch_limit is not None:
            np.random.RandomState(config.split_seed).shuffle(train_patches)
    elif config.split_strategy == "random_patch":
        shuffled = list(dataset)
        rng.shuffle(shuffled)
        split_index = max(1, int(len(shuffled) * config.train_eval_ratio))
        train_patches = shuffled[:split_index]
        val_patches = shuffled[split_index:]
    else:
        raise ValueError(f"Unknown split_strategy: {config.split_strategy}")

    if config.eval_patch_limit is not None:
        if config.eval_patch_limit < 1:
            raise ValueError("eval_patch_limit must be positive")
        eval_rng = np.random.RandomState(config.split_seed)
        val_patches = [val_patches[i] for i in eval_rng.permutation(len(val_patches))[:config.eval_patch_limit]]
        holdouts = {
            key: [patches[i] for i in eval_rng.permutation(len(patches))[:config.eval_patch_limit]]
            for key, patches in sorted(holdouts.items())
        }
    if config.train_patch_limit is not None:
        train_patches = train_patches[:config.train_patch_limit]
    if config.val_patch_limit is not None:
        val_patches = val_patches[:config.val_patch_limit]
    if config.limit is not None:
        train_patches = train_patches[:config.limit]
        val_patches = val_patches[:max(1, config.limit // 2)]
    if not train_patches or not val_patches:
        raise ValueError("Training and validation splits must be nonempty")
    print(f"Split: train={len(train_patches)} val={len(val_patches)} input_channels={config.input_channels}")

    from pipeline.models.factory import get_model

    model_config = getattr(config, "model_config", None)
    model = get_model(model_name, in_channels=config.input_channels, model_config=model_config)
    train_loader = build_loader(
        train_patches,
        batch_size=config.batch_size,
        shuffle=True,
        oversample_positive_patches=config.oversample_positive_patches,
        positive_oversample_ratio=config.positive_oversample_ratio,
        min_positive_fraction=config.min_positive_fraction,
        seed=config.seed,
        min_positive_pixels=config.min_positive_pixels,
    )
    val_loader = build_loader(
        val_patches,
        batch_size=config.batch_size,
        shuffle=False,
        oversample_positive_patches=False,
        positive_oversample_ratio=config.positive_oversample_ratio,
        min_positive_fraction=config.min_positive_fraction,
        seed=config.seed,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(config.output_dir) / f"{timestamp}_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    split = {key: [_patch_name(patch) for patch in patches]
             for key, patches in {"treino": train_patches, "validacao": val_patches, **holdouts}.items()}
    (run_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Run directory: {run_dir}")
    trainer = GenericTrainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config)
    checkpoint_path = run_dir / config.checkpoint_name

    def save_checkpoint(history):
        temporary_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
        torch.save({
            "model_name": model_name,
            "history": history,
            "config": config.__dict__,
            "state_dict": trainer.best_state,
            "best_epoch": trainer.best_epoch,
            "best_val_f2": trainer.best_val_f2,
            "best_threshold": trainer.best_threshold,
            "split": split,
        }, temporary_path)
        temporary_path.replace(checkpoint_path)

    history = trainer.train(on_improvement=save_checkpoint)
    save_checkpoint(history)
    metadata = _save_run_metadata(run_dir, model_name, config, history, str(trainer.device))
    metadata.update(best_epoch=trainer.best_epoch, best_val_f2=trainer.best_val_f2,
                    best_threshold=trainer.best_threshold, split_counts={key: len(value) for key, value in split.items()})
    metadata["holdouts"] = {
        key: trainer.evaluate_holdout(build_loader(patches, config.batch_size), trainer.best_threshold)
        for key, patches in holdouts.items()
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    prediction_summary = (
        _save_predictions(model, val_patches, run_dir, trainer.device, config.input_channels)
        if config.save_validation_predictions else None
    )

    return {
        "history": history,
        "checkpoint": str(checkpoint_path),
        "device": str(trainer.device),
        "run_dir": str(run_dir),
        "predictions": prediction_summary,
        "metadata": metadata,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic training entry point for the flood segmentation pipeline.")
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    cfg = TrainingConfig(
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
    print(f"Training config: {cfg}")
    print("This is a smoke-test training entry point; provide prepared patch data to run the full loop.")
