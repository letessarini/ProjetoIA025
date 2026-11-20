from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from importlib.metadata import version
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _resolve_exec(name: str) -> Optional[str]:
    return shutil.which(name) or shutil.which(name.replace("v2_", "v2_"))


def is_nnunet_available() -> bool:
    return shutil.which("nnUNetv2_train") is not None


class NnUNetAdapter:
    """Thin wrapper around the external nnU-Net v2 CLI used by this project."""

    def __init__(self, dataset_id: int = 102, trainer: str = "nnUNetTrainer", plans: str = "nnUNetPlansZScore", config: str = "2d", fold: int = 0, device: str = "cuda", work_dir: Optional[str] = None):
        self.dataset_id = int(dataset_id)
        self.trainer = trainer
        self.plans = plans
        self.config = config
        self.fold = int(fold)
        self.device = device
        self.work_dir = Path(work_dir) if work_dir else Path(__file__).resolve().parents[2]

    @staticmethod
    def build_env(extra_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        env = dict(os.environ)
        defaults = {
            "nnUNet_raw": str(Path(__file__).resolve().parents[2] / "nnUNet_raw"),
            "nnUNet_preprocessed": str(Path(__file__).resolve().parents[2] / "nnUNet_preprocessed"),
            "nnUNet_results": str(Path(__file__).resolve().parents[2] / "nnUNet_results"),
        }
        for key, value in defaults.items():
            env.setdefault(key, value)
        if extra_env:
            env.update({k: str(v) for k, v in extra_env.items() if v is not None})
        return env

    def command(self, action: str, *, val: bool = False, val_best: bool = False) -> List[str]:
        base = ["nnUNetv2_train", str(self.dataset_id), self.config, str(self.fold), "-tr", self.trainer, "-p", self.plans, "-device", self.device]
        if action == "train":
            return base
        if action == "val":
            return ["nnUNetv2_train", str(self.dataset_id), self.config, str(self.fold), "-tr", self.trainer, "-p", self.plans, "--val", "--val_best", "-device", self.device]
        if action == "predict":
            return [
                "nnUNetv2_predict",
                "-i", str(self.work_dir / "nnUNet_raw" / f"Dataset{self.dataset_id}_*" / "imagesTs"),
                "-o", str(self.work_dir / "resultados_teste" / f"Dataset{self.dataset_id}"),
                "-d", str(self.dataset_id),
                "-c", self.config,
                "-f", str(self.fold),
                "-tr", self.trainer,
                "-p", self.plans,
                "-chk", "checkpoint_best.pth",
                "-device", self.device,
            ]
        raise ValueError(f"Unsupported nnU-Net action: {action}")

    def run(self, action: str = "train", *, env: Optional[Dict[str, str]] = None, capture: bool = False) -> Dict[str, Any]:
        executable = shutil.which("nnUNetv2_train") if action in {"train", "val"} else shutil.which("nnUNetv2_predict")
        if executable is None:
            raise RuntimeError("nnU-Net CLI not found in PATH. Activate the project environment and ensure nnU-Net is installed.")

        cmd = self.command(action)
        merged_env = self.build_env(env)
        result = subprocess.run(cmd, cwd=str(self.work_dir), env=merged_env, text=True, capture_output=capture, check=False)
        return {
            "action": action,
            "command": " ".join(cmd),
            "returncode": int(result.returncode),
            "stdout": result.stdout if capture else "",
            "stderr": result.stderr if capture else "",
            "ok": result.returncode == 0,
        }


def build_nnunet_model(**kwargs):
    return NnUNetAdapter(**kwargs)


NNUNET_BENCHMARK_VERSION = "2.6.4"


def check_nnunet_benchmark_version():
    installed = version("nnunetv2")
    if installed != NNUNET_BENCHMARK_VERSION:
        raise RuntimeError(f"nnunet_2d requires nnunetv2=={NNUNET_BENCHMARK_VERSION}; found {installed}.")


def fingerprint_benchmark_dataset(dataset_id):
    from nnunetv2.experiment_planning.dataset_fingerprint.fingerprint_extractor import DatasetFingerprintExtractor
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprint_dataset

    class FloodFingerprint(DatasetFingerprintExtractor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # None of the benchmark bands uses global foreground intensity statistics.
            self.num_foreground_voxels_for_intensitystats = 1_000_000

    extract_fingerprint_dataset(int(dataset_id), FloodFingerprint, num_processes=1, check_dataset_integrity=True)


def train_benchmark_fold(dataset_id, epochs, seed, device, smoke, summary_path):
    """Run the upstream trainer with the configured epoch budget on the shared fold."""
    import random

    import numpy as np
    import torch
    from nnunetv2.run.run_training import get_trainer_from_args

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = get_trainer_from_args(str(dataset_id), "2d", 0, device=torch.device(device))
    trainer.num_epochs = epochs
    if smoke:
        os.environ["nnUNet_n_proc_DA"] = "0"
        trainer.configuration_manager.configuration["batch_size"] = 2
        trainer.num_iterations_per_epoch = 2
        trainer.num_val_iterations_per_epoch = 1
    trainer.run_training()
    checkpoint = Path(trainer.output_folder) / "checkpoint_best.pth"
    # This checkpoint was written locally by the trainer in this subprocess.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    Path(summary_path).write_text(json.dumps({
        "best_epoch": saved["current_epoch"],
        "checkpoint": str(checkpoint),
        "batch_size": trainer.batch_size,
        "iterations_per_epoch": trainer.num_iterations_per_epoch,
    }), encoding="utf-8")


def run_nnunet_benchmark(dataset, config, manifest, smoke=False):
    """Use the pipeline split with nnU-Net planning, training, and probability export."""
    import numpy as np

    from pipeline.data_preparation.convert_to_nnunet import export_nnunet_cases
    from pipeline.metrics.segmentation_metrics import compute_histogram_metrics, update_score_histogram
    from pipeline.models.trainer import select_training_patches

    check_nnunet_benchmark_version()
    if config.epochs < 1 or (config.threshold_bins is not None and config.threshold_bins < 2):
        raise ValueError("nnU-Net requires a positive epoch count and at least two threshold bins")
    groups = select_training_patches(dataset, config, manifest)
    now = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = (Path(config.output_dir) / f"{now}_nnunet_2d").resolve()
    run_dir.mkdir(parents=True)
    split = {role: [Path(patch["path"]).name for patch in patches] for role, patches in groups.items()}
    (run_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Run directory: {run_dir}", flush=True)

    dataset_id = 901
    dataset_name = f"Dataset{dataset_id}_FloodBenchmark"
    env = NnUNetAdapter.build_env({key: run_dir / key for key in (
        "nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"
    )})
    # Small subprocess counts keep the smoke test practical on Windows and CPU.
    env["nnUNet_n_proc_DA"] = "1" if smoke else "2"
    env.setdefault("nnUNet_compile", "false")
    env.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "2")
    project_root = Path(__file__).resolve().parents[2]

    def run_python(script, *args):
        subprocess.run([sys.executable, "-u", "-c", script, *map(str, args)],
                       cwd=project_root, env=env, check=True)

    def run_cli(name, *args):
        print(f"[nnU-Net] {name}", flush=True)
        run_python("import sys; from importlib.metadata import entry_points; "
                   "next(iter(entry_points(group='console_scripts', name=sys.argv.pop(1)))).load()()",
                   name, *args)

    raw = Path(env["nnUNet_raw"]) / dataset_name
    processed = Path(env["nnUNet_preprocessed"]) / dataset_name
    names = manifest["canais"]
    channels = config.input_channels if config.input_channels is not None else len(names)
    if not 1 <= channels <= len(names):
        raise ValueError(f"input_channels={channels} invalid for the nnU-Net manifest")
    spacing = float(manifest["grade"]["res_m"])
    images, labels = raw / "imagesTr", raw / "labelsTr"
    export_nnunet_cases(groups["treino"], images, channels, spacing, labels)
    description = {
        "channel_names": {str(i): "noNorm" if name == "mascara_valida" else name
                          for i, name in enumerate(names[:channels])},
        "labels": {"background": 0, "flood": 1, "ignore": 2},
        "numTraining": len(groups["treino"]),
        "file_ending": ".nii.gz",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    dataset_json = raw / "dataset.json"
    dataset_json.write_text(json.dumps(description, indent=2), encoding="utf-8")
    print("[nnU-Net] Extracting the training fingerprint", flush=True)
    run_python("import sys; from pipeline.models.nnunet_adapter import fingerprint_benchmark_dataset; "
               "fingerprint_benchmark_dataset(sys.argv[1])", dataset_id)
    run_cli("nnUNetv2_plan_experiment", "-d", dataset_id)

    # Apply training-derived plans to validation and keep all test cases out of preprocessing.
    export_nnunet_cases(groups["validacao"], images, channels, spacing, labels)
    description["numTraining"] += len(groups["validacao"])
    dataset_json.write_text(json.dumps(description, indent=2), encoding="utf-8")
    run_cli("nnUNetv2_preprocess", "-d", dataset_id, "-c", "2d", "-np", 1 if smoke else 2)
    fold = [{"train": [Path(name).stem for name in split["treino"]],
             "val": [Path(name).stem for name in split["validacao"]]}]
    (processed / "splits_final.json").write_text(json.dumps(fold, indent=2), encoding="utf-8")
    summary_path = run_dir / "training_summary.json"
    train_options = dict(dataset_id=dataset_id, epochs=config.epochs, seed=config.seed,
                         device=config.resolved_device, smoke=smoke, summary_path=str(summary_path))
    print("[nnU-Net] Training 2D fold 0", flush=True)
    run_python("import json, sys; from pipeline.models.nnunet_adapter import train_benchmark_fold; "
               "train_benchmark_fold(**json.loads(sys.argv[1]))", json.dumps(train_options))
    fitted = json.loads(summary_path.read_text(encoding="utf-8"))

    evaluation = {name: patches for name, patches in groups.items() if name != "treino"}
    inputs, predictions = raw / "imagesTs", run_dir / "predictions"
    export_nnunet_cases((patch for patches in evaluation.values() for patch in patches),
                        inputs, channels, spacing)
    run_cli("nnUNetv2_predict", "-i", inputs, "-o", predictions, "-d", dataset_id, "-c", "2d",
            "-f", 0, "-chk", "checkpoint_best.pth", "-device", config.resolved_device,
            "--save_probabilities", "--disable_tta", "-npp", 1, "-nps", 1)
    histograms = {}
    for role, patches in evaluation.items():
        hist = np.zeros((2, (config.threshold_bins or 1000) + 1), dtype=np.int64)
        for patch in patches:
            path = Path(patch["path"])
            with np.load(predictions / f"{path.stem}.npz", allow_pickle=False) as pred, np.load(path, allow_pickle=False) as data:
                probability = pred["probabilities"][1, 0]
                target = data["y"]
                if probability.shape != target.shape:
                    raise ValueError(f"nnU-Net probability shape {probability.shape} differs from target {target.shape} in {path.name}")
                update_score_histogram(hist, probability, target, data["valido"])
        histograms[role] = hist

    validation = compute_histogram_metrics(histograms.pop("validacao"), selection_metric=config.selection_metric)
    metadata = {
        "model_name": "nnunet_2d",
        "device": config.resolved_device,
        "config": {
            "seed": config.seed, "split_seed": config.split_seed, "epochs": config.epochs,
            "input_channels": channels, "loss_name": "nnUNetTrainer Dice + cross-entropy",
            "batch_size": fitted["batch_size"], "iterations_per_epoch": fitted["iterations_per_epoch"],
            "trainer": "nnUNetTrainer", "configuration": "2d", "fold": 0,
            "threshold_bins": config.threshold_bins or 1000, "selection_metric": config.selection_metric,
        },
        "best_epoch": fitted["best_epoch"],
        "best_threshold": validation["threshold"],
        "validation": validation,
        "holdouts": {role: compute_histogram_metrics(hist, validation["threshold"])
                     for role, hist in histograms.items()},
        "split_counts": {key: len(value) for key, value in split.items()},
        "plans": str(processed / "nnUNetPlans.json"),
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"run_dir": str(run_dir), "metadata": metadata, "device": metadata["device"],
            "checkpoint": fitted["checkpoint"], "predictions": None}
