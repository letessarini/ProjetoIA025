from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from pipeline.data_preparation.dataset_loader import find_dataset_dir, list_patch_files, load_manifest, validate_patch_file
from pipeline.data_preparation.prepare_dataset import DEFAULT_PREPROCESS, prepare_dataset
from pipeline.data_preparation.convert_to_nnunet import convert_project_patches_to_nnunet
from pipeline.models.factory import get_model
from pipeline.models.trainer import TrainingConfig, run_training_model
from pipeline.models.nnunet_adapter import NnUNetAdapter

ROOT = Path(__file__).resolve().parents[2]


def find_dataset_dir_local() -> Path:
    """Compatibility wrapper for pipeline entry points."""
    return find_dataset_dir()


def summarize_dataset(dataset_dir: Optional[Path | str] = None, limit: int = 3) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir) if dataset_dir is not None else find_dataset_dir_local()
    manifest = load_manifest(dataset_dir)
    patch_paths = list_patch_files(dataset_dir)
    samples = []
    for patch_path in patch_paths[:limit]:
        try:
            info = validate_patch_file(patch_path)
            samples.append({
                "name": info["name"],
                "shape_X": info["shape_X"],
                "shape_y": info["shape_y"],
                "n_channels": info["n_channels"],
            })
        except Exception as exc:  # pragma: no cover - keep exploratory flow resilient
            samples.append({"name": patch_path.name, "error": str(exc)})

    return {
        "dataset_dir": str(dataset_dir),
        "manifest": manifest,
        "patch_count": len(patch_paths),
        "samples": samples,
    }


def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    config_path = Path(path) if path else ROOT / "pipeline" / "config" / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must contain a JSON object.")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the modular flood-segmentation pipeline.")
    parser.add_argument("--config", type=str, default=str(ROOT / "pipeline" / "config" / "config.json"), help="Path to the JSON config file")
    parser.add_argument("--model", type=str, default=None, help="Model name: unet, u-net, nnunet, nnUNetTrainer")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs for smoke-test mode")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--limit", type=int, default=None, help="Number of patches to prepare/train on; omit to use the full dataset")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--loss", type=str, default=None, help="Loss name: bce, weighted_bce, focal")
    parser.add_argument("--loss-config", type=str, default=None, help="JSON string for loss config, e.g. '{\"pos_weight\": 3.0}'")
    parser.add_argument("--nnunet-dataset-id", type=int, default=None, help="nnU-Net dataset ID (for example 101, 102, 103)")
    parser.add_argument("--nnunet-trainer", type=str, default=None, help="nnU-Net trainer name, e.g. nnUNetTrainer or nnUNetTrainerTversky")
    parser.add_argument("--nnunet-plans", type=str, default=None, help="nnU-Net plans name, e.g. nnUNetPlansZScore")
    parser.add_argument("--nnunet-config", type=str, default=None, help="nnU-Net config, typically 2d")
    parser.add_argument("--nnunet-fold", type=int, default=None, help="nnU-Net fold index")
    return parser.parse_args()


def resolve_limit(config_data: Dict[str, Any], cli_limit: Optional[int]) -> Optional[int]:
    """CLI arguments override config, and a missing/negative value means use the full dataset."""
    if cli_limit is not None:
        return cli_limit if cli_limit > 0 else None

    raw = config_data.get("limit")
    if raw is None or raw == "":
        return None
    value = int(raw)
    return value if value > 0 else None


def parse_loss_config(raw: Optional[str]) -> Dict[str, Any]:
    if raw is None or raw.strip() == "":
        return {}

    text = raw.strip()

    def _coerce(value: str):
        value = value.strip()
        if value.lower() in {"true", "false"}:
            return value.lower() == "true"
        if value.lower() in {"none", "null"}:
            return None
        try:
            if "." in value or "e" in value.lower():
                return float(value)
            return int(value)
        except ValueError:
            return value.strip("\"'")

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
        raise ValueError("--loss-config must be a JSON object")
    except json.JSONDecodeError:
        pass

    if text.startswith("{") and text.endswith("}"):
        inner = text[1:-1].strip()
        if inner:
            items: Dict[str, Any] = {}
            for chunk in [part.strip() for part in inner.split(",") if part.strip()]:
                if "=" in chunk:
                    key, val = chunk.split("=", 1)
                elif ":" in chunk:
                    key, val = chunk.split(":", 1)
                else:
                    raise ValueError(f"Invalid loss config pair: {chunk!r}")
                key = key.strip().strip("\"'{}[]() ")
                val = val.strip().strip("\"'{}[]() ")
                items[key] = _coerce(val)
            return items

    if "=" in text:
        items: Dict[str, Any] = {}
        for chunk in [part.strip() for part in text.split(",") if part.strip()]:
            key, val = chunk.split("=", 1)
            items[key.strip().strip("\"'{}[]() ")] = _coerce(val.strip())
        return items

    raise ValueError(f"Invalid JSON for --loss-config: {raw!r}")


def run_nnunet_pipeline(config_data: Dict[str, Any], model_name: str, epochs: Optional[int], limit: Optional[int], device: str) -> Dict[str, Any]:
    dataset_id = int(config_data.get("nnunet_dataset_id", 102))
    dataset_name = str(config_data.get("nnunet_dataset_name", "FloodSegmentation"))
    trainer = str(config_data.get("nnunet_trainer", "nnUNetTrainer"))
    plans = str(config_data.get("nnunet_plans", "nnUNetPlansZScore"))
    config = str(config_data.get("nnunet_config", "2d"))
    fold = int(config_data.get("nnunet_fold", 0))
    raw_root = Path(config_data.get("nnunet_raw_root", str(ROOT / "nnUNet_raw")))

    dataset_dir = find_dataset_dir_local()
    prepared_dataset = convert_project_patches_to_nnunet(
        raw_root=raw_root,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
    )
    print(f"nnU-Net raw dataset created at: {prepared_dataset}")

    env = {
        "nnUNet_raw": str(raw_root),
        "nnUNet_preprocessed": str(ROOT / "nnUNet_preprocessed"),
        "nnUNet_results": str(ROOT / "nnUNet_results"),
    }
    adapter = NnUNetAdapter(dataset_id=dataset_id, trainer=trainer, plans=plans, config=config, fold=fold, device=device, work_dir=str(ROOT))
    result = adapter.run(action="train", env=env, capture=True)
    print(f"nnU-Net run: {result['command']}")
    print(f"nnU-Net return code: {result['returncode']}")
    if result["stdout"]:
        print(result["stdout"])
    if result["stderr"]:
        print(result["stderr"])
    if not result["ok"]:
        raise RuntimeError(f"nnU-Net command failed: {result['command']}")
    return result


def main() -> None:
    args = parse_args()
    config_data = load_config(args.config)

    model_name = args.model if args.model is not None else config_data.get("model", "unet")
    epochs = args.epochs if args.epochs is not None else int(config_data.get("epochs", 3))
    batch_size = args.batch_size if args.batch_size is not None else int(config_data.get("batch_size", 4))
    limit = resolve_limit(config_data, args.limit)
    seed = args.seed if args.seed is not None else int(config_data.get("seed", 20261022))
    loss_name = args.loss if args.loss is not None else config_data.get("loss_name", "weighted_bce")
    loss_config = parse_loss_config(args.loss_config) if args.loss_config is not None else config_data.get("loss_config", {})
    oversample_positive_patches = bool(config_data.get("oversample_positive_patches", True))
    positive_oversample_ratio = float(config_data.get("positive_oversample_ratio", 0.7))
    min_positive_fraction = float(config_data.get("min_positive_fraction", 1e-6))
    nnunet_dataset_id = args.nnunet_dataset_id if args.nnunet_dataset_id is not None else int(config_data.get("nnunet_dataset_id", 102))
    nnunet_trainer = args.nnunet_trainer if args.nnunet_trainer is not None else str(config_data.get("nnunet_trainer", "nnUNetTrainer"))
    nnunet_plans = args.nnunet_plans if args.nnunet_plans is not None else str(config_data.get("nnunet_plans", "nnUNetPlansZScore"))
    nnunet_config = args.nnunet_config if args.nnunet_config is not None else str(config_data.get("nnunet_config", "2d"))
    nnunet_fold = args.nnunet_fold if args.nnunet_fold is not None else int(config_data.get("nnunet_fold", 0))

    dataset_dir = find_dataset_dir_local()
    print(f"Repo root: {ROOT}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Using config: {args.config}")

    summary = summarize_dataset(dataset_dir, limit=3)
    print(f"Patches found: {summary['patch_count']}")
    for sample in summary["samples"]:
        if "error" in sample:
            print(f"Sample issue: {sample['name']} -> {sample['error']}")
        else:
            print(f"Sample: {sample['name']}, X={sample['shape_X']}, y={sample['shape_y']}")

    print("[Stage 1/2] Loading dataset and preparing patches...")
    prepared = prepare_dataset(dataset_dir=dataset_dir, limit=limit, config=DEFAULT_PREPROCESS, augment=False)
    print(f"Prepared patches for smoke test: {prepared['n_patches']}")
    if not prepared["patches"]:
        raise RuntimeError("No prepared patches were produced from the dataset.")

    print(f"Selected model: {model_name}")
    print("Using CUDA automatically if available, otherwise CPU.")
    print("[Stage 2/2] Modelling: training the selected model...")

    if model_name.lower().replace("-", "_") in {"nnunet", "nn_unet", "nnu_net", "nn_u_net"}:
        print("[Stage 2/2] Running external nnU-Net v2 pipeline...")
        config_data = {
            **config_data,
            "nnunet_dataset_id": nnunet_dataset_id,
            "nnunet_trainer": nnunet_trainer,
            "nnunet_plans": nnunet_plans,
            "nnunet_config": nnunet_config,
            "nnunet_fold": nnunet_fold,
        }
        result = run_nnunet_pipeline(config_data, model_name, epochs, limit, "cuda" if __import__('torch').cuda.is_available() else "cpu")
        print(f"nnU-Net CLI finished successfully: {result['ok']}")
        return

    config = TrainingConfig(
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        limit=limit,
        output_dir=str(ROOT / config_data.get("output_dir", "results")),
        checkpoint_name=config_data.get("checkpoint_name", "last_model.pt"),
        model_config=config_data.get("model_config", {}),
        loss_name=loss_name,
        loss_config=loss_config,
        oversample_positive_patches=oversample_positive_patches,
        positive_oversample_ratio=positive_oversample_ratio,
        min_positive_fraction=min_positive_fraction,
    )

    result = run_training_model(model_name, prepared["patches"], config=config)
    print(f"Training finished on device: {result['device']}")
    print(f"Run directory: {result['run_dir']}")
    print(f"Checkpoint saved to: {result['checkpoint']}")
    print(f"Predictions saved to: {result['predictions']['predictions_dir']}")
    print(f"Config metadata saved to: {result['run_dir']}/run_metadata.json")
    print("Pipeline entry point is working. Next step: add a real evaluation and reporting stage.")


if __name__ == "__main__":
    main()
