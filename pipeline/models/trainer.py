from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime
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

    @property
    def resolved_device(self) -> str:
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"


class PositiveOversamplingDataset:
    """Sample positive patches more often during training to match the legacy flood-segmentation logic."""

    def __init__(self, patches: Sequence[Dict[str, Any]], positive_ratio: float = 0.7, min_positive_fraction: float = 1e-6):
        self.patches = list(patches)
        self.positive_ratio = max(0.0, min(float(positive_ratio), 1.0))
        self.min_positive_fraction = max(0.0, float(min_positive_fraction))
        self.positive_index = [self._is_positive_patch(p) for p in self.patches]

    @staticmethod
    def _is_positive_patch(patch: Dict[str, Any], min_positive_fraction: float = 1e-6) -> bool:
        y = np.asarray(patch.get("y", np.zeros((1, 1), dtype=np.float32)), dtype=np.float32)
        valid = np.asarray(patch.get("valido", np.ones_like(y)), dtype=np.float32)
        if y.size == 0:
            return False
        positive_fraction = float(((y > 0.5) & (valid > 0.5)).mean())
        return bool(positive_fraction > max(0.0, min_positive_fraction))

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx):
        patch = self.patches[idx]
        X = patch["X"].astype(np.float32)
        y = patch["y"].astype(np.float32)
        valid = patch["valido"].astype(np.float32)
        return X, y, valid

    def sampler_weights(self):
        if not self.patches:
            return np.array([], dtype=np.float32)

        positive_count = sum(self.positive_index)
        negative_count = len(self.patches) - positive_count
        if positive_count == 0:
            return np.ones(len(self.patches), dtype=np.float32)

        pos_weight = 1.0 / max(self.positive_ratio, 1e-6)
        neg_weight = 1.0 / max(1.0 - self.positive_ratio, 1e-6)
        out = []
        for is_positive in self.positive_index:
            out.append(pos_weight if is_positive else neg_weight)
        return np.asarray(out, dtype=np.float32)


class PatchDataset:
    """Simple dataset wrapper built from patch dictionaries."""

    def __init__(self, patches: Sequence[Dict[str, Any]]):
        self.patches = list(patches)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        patch = self.patches[idx]
        X = patch["X"].astype(np.float32)
        y = patch["y"].astype(np.float32)
        valid = patch["valido"].astype(np.float32)
        return X, y, valid


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
):
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to train the model.")

    if oversample_positive_patches:
        dataset = PositiveOversamplingDataset(
            patches,
            positive_ratio=positive_oversample_ratio,
            min_positive_fraction=min_positive_fraction,
        )
        weights = dataset.sampler_weights()
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
        valid_mask = valid_mask.unsqueeze(1)
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
        target = target.unsqueeze(1)
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


def compute_masked_tversky_loss(logits, target, valid_mask=None, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6):
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
            return compute_masked_tversky_loss(logits, target, valid_mask, alpha=alpha, beta=beta, eps=eps)

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

    raise ValueError(f"Unsupported loss '{loss_name}'. Supported: bce, weighted_bce, tversky, focal")


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


from pipeline.metrics.segmentation_metrics import compute_metrics


def _threshold_metrics(tp, fp, fn, tn, eps: float = 1e-9) -> Dict[str, float]:
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return {
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
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


def _evaluate_threshold_sweep(model, loader, device, thresholds: Sequence[float]):
    model.eval()
    threshold_metrics = []
    for threshold in thresholds:
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for X, y, valid in loader:
                X = X.to(device)
                y = _ensure_segmentation_shape(y.to(device), batch_size=X.shape[0])
                valid = _ensure_segmentation_shape(valid.to(device), batch_size=X.shape[0])
                logits = model(X)
                batch_tp, batch_fp, batch_fn, batch_tn = _compute_confusion_from_logits(logits, y, valid, threshold=threshold)
                tp += batch_tp
                fp += batch_fp
                fn += batch_fn
                tn += batch_tn
        threshold_metrics.append({
            "threshold": float(threshold),
            "metrics": _threshold_metrics(tp, fp, fn, tn),
        })

    best = max(threshold_metrics, key=lambda item: item["metrics"]["f1"])
    return threshold_metrics, best


class GenericTrainer:
    def __init__(self, model, train_loader, val_loader, config: TrainingConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(config.resolved_device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.criterion = build_loss(config.loss_name, config.loss_config)

    def _train_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        data_len = len(self.train_loader)
        print(f"[Stage 2/2] Modelling: training epoch {self._current_epoch}/{self.config.epochs}")

        for batch_idx, (X, y, valid) in enumerate(self.train_loader, start=1):
            X = X.to(self.device)
            y = _ensure_segmentation_shape(y.to(self.device), batch_size=X.shape[0])
            valid = _ensure_segmentation_shape(valid.to(self.device), batch_size=X.shape[0])

            logits = self.model(X)
            loss = self.criterion(logits, y, valid)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item()) * X.size(0)
            total_samples += X.size(0)
            _print_progress("Training batches", batch_idx, data_len)

        return total_loss / max(1, total_samples)

    def _aggregate_dataset_metrics(self, loader):
        self.model.eval()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for X, y, valid in loader:
                X = X.to(self.device)
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

        for batch_idx, (X, y, valid) in enumerate(self.val_loader, start=1):
            X = X.to(self.device)
            y = _ensure_segmentation_shape(y.to(self.device), batch_size=X.shape[0])
            valid = _ensure_segmentation_shape(valid.to(self.device), batch_size=X.shape[0])

            with torch.no_grad():
                logits = self.model(X)
                loss = self.criterion(logits, y, valid)

            total_loss += float(loss.item()) * X.size(0)
            total_samples += X.size(0)

            batch_tp, batch_fp, batch_fn, batch_tn = _compute_confusion_from_logits(logits, y, valid, threshold=0.5)
            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            tn += batch_tn
            _print_progress("Validation batches", batch_idx, data_len)

        base_metrics = compute_metrics(tp, fp, fn, tn)
        threshold_metrics, best_threshold_metrics = _evaluate_threshold_sweep(self.model, self.val_loader, self.device, self.config.thresholds)
        return total_loss / max(1, total_samples), base_metrics, threshold_metrics, best_threshold_metrics

    def train(self):
        history = []
        for epoch in range(self.config.epochs):
            self._current_epoch = epoch + 1
            train_loss = self._train_epoch()
            val_loss, val_metrics, val_threshold_metrics, best_threshold_metrics = self._eval_epoch()
            train_metrics = self._aggregate_dataset_metrics(self.train_loader)

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "threshold_sweep": val_threshold_metrics,
                "best_threshold": best_threshold_metrics,
            })
            print(
                f"epoch {epoch + 1}/{self.config.epochs} | "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_iou={val_metrics['iou']:.4f} | val_precision={val_metrics['precision']:.4f} | "
                f"val_recall={val_metrics['recall']:.4f} | val_f1={val_metrics['f1']:.4f} | "
                f"best_threshold={best_threshold_metrics['threshold']:.2f} (f1={best_threshold_metrics['metrics']['f1']:.4f}) | "
                f"device={self.device}"
            )
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


def _save_predictions(model, patches: Sequence[Dict[str, Any]], run_dir: Path, device: Any):
    model.eval()
    preds_dir = run_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    for patch in patches:
        x = torch.from_numpy(np.asarray(patch["X"], dtype=np.float32)).unsqueeze(0).to(device)
        target = np.asarray(patch["y"], dtype=np.float32)
        valid = np.asarray(patch["valido"], dtype=np.float32)

        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]

        file_name = Path(patch["meta"]["name"]) if "meta" in patch and "name" in patch["meta"] else Path(str(patch["path"]).split("\\")[-1])
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


def run_training_model(model_name: str, dataset: List[Dict[str, Any]], config: Optional[TrainingConfig] = None):
    """Train a generic model from a list of prepared patches."""
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required to run model training.")

    config = config or TrainingConfig(model_name=model_name)
    rng = np.random.RandomState(config.seed)
    shuffled = list(dataset)
    rng.shuffle(shuffled)

    split_index = max(1, int(len(shuffled) * config.train_eval_ratio))
    train_patches = shuffled[:split_index]
    val_patches = shuffled[split_index:]

    if config.limit is not None:
        train_patches = train_patches[: config.limit]
        val_patches = val_patches[: max(1, config.limit // 2)]

    from pipeline.models.factory import get_model

    model_config = getattr(config, "model_config", None)
    model = get_model(model_name, in_channels=train_patches[0]["X"].shape[0], model_config=model_config)
    train_loader = build_loader(
        train_patches,
        batch_size=config.batch_size,
        shuffle=True,
        oversample_positive_patches=config.oversample_positive_patches,
        positive_oversample_ratio=config.positive_oversample_ratio,
        min_positive_fraction=config.min_positive_fraction,
        seed=config.seed,
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

    trainer = GenericTrainer(model=model, train_loader=train_loader, val_loader=val_loader, config=config)
    history = trainer.train()

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{timestamp}_{model_name}"
    metadata = _save_run_metadata(run_dir, model_name, config, history, str(trainer.device))
    prediction_summary = _save_predictions(model, val_patches, run_dir, trainer.device)

    checkpoint_path = run_dir / config.checkpoint_name
    torch.save({
        "model_name": model_name,
        "history": history,
        "config": config.__dict__,
        "state_dict": model.state_dict(),
    }, checkpoint_path)

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
