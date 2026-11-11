from __future__ import annotations

import os
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
