#!/usr/bin/env python3
"""
Varredura de modelos e losses para o projeto de segmentacao de inundacoes.

Execute na raiz do repositorio:
    python benchmark_models_losses.py --config pipeline/config/config.json

O script chama pipeline.py separadamente para cada combinacao, salva um log
por experimento e consolida as metricas encontradas em resultados_benchmark.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_LOSS_CONFIGS = {
    "bce": {},
    "tversky": {"alpha": 0.3, "beta": 0.7},
    "focal": {"alpha": 0.25, "gamma": 2.0},
}

METRIC_PATTERNS = {
    "val_iou": r"val_iou=([0-9]*\.?[0-9]+)",
    "val_precision": r"val_precision=([0-9]*\.?[0-9]+)",
    "val_recall": r"val_recall=([0-9]*\.?[0-9]+)",
    "val_f1": r"val_f1=([0-9]*\.?[0-9]+)",
    "val_f2": r"val_f2=([0-9]*\.?[0-9]+)",
    "best_threshold": r"best_threshold=([0-9]*\.?[0-9]+)",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Executa benchmark de modelos e losses."
    )
    parser.add_argument(
        "--config",
        default="pipeline/config/config.json",
        help="Caminho do arquivo de configuracao do pipeline.",
    )
    parser.add_argument(
        "--pipeline",
        default="pipeline.py",
        help="Caminho do script principal do treinamento.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["unet", "unet++", "nnunet"],
        help="Modelos a testar. Ex.: unet unet++ nnunet",
    )
    parser.add_argument(
        "--losses",
        nargs="+",
        default=["bce", "tversky", "focal"],
        help="Losses a testar. Ex.: bce tversky focal",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita o numero de amostras/patches para smoke test.",
    )
    parser.add_argument(
        "--gpu",
        default=None,
        help="Valor de CUDA_VISIBLE_DEVICES. Ex.: 0",
    )
    parser.add_argument(
        "--output-dir",
        default="resultados_benchmark",
        help="Pasta para logs e CSV.",
    )
    parser.add_argument(
        "--loss-configs",
        default=None,
        help=(
            "JSON opcional sobrescrevendo configs por loss. "
            'Ex.: \'{"tversky":{"alpha":0.3,"beta":0.7},'
            '"focal":{"alpha":0.25,"gamma":2.0}}\''
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continua a varredura se um experimento falhar.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Pula experimentos que ja tenham arquivo .success.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Executavel Python usado para iniciar pipeline.py.",
    )
    return parser.parse_args()


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value).strip("_").lower()


def extract_metrics(log_text: str) -> dict:
    result = {}
    for metric, pattern in METRIC_PATTERNS.items():
        matches = re.findall(pattern, log_text)
        result[metric] = float(matches[-1]) if matches else None
    return result


def main():
    args = parse_args()

    project_root = Path.cwd()
    pipeline_path = (project_root / args.pipeline).resolve()
    config_path = (project_root / args.config).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    logs_dir = output_dir / "logs"

    if not pipeline_path.is_file():
        raise FileNotFoundError(f"Pipeline nao encontrado: {pipeline_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuracao nao encontrada: {config_path}")
    if args.epochs < 1:
        raise ValueError("--epochs precisa ser maior que zero.")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit precisa ser maior que zero.")

    loss_configs = dict(DEFAULT_LOSS_CONFIGS)
    if args.loss_configs:
        custom_configs = json.loads(args.loss_configs)
        if not isinstance(custom_configs, dict):
            raise ValueError("--loss-configs precisa ser um objeto JSON.")
        loss_configs.update(custom_configs)

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    experiments = [
        (model, loss)
        for model in args.models
        for loss in args.losses
    ]

    summary_path = output_dir / "resultados_benchmark.csv"
    fields = [
        "model",
        "loss",
        "epochs",
        "status",
        "return_code",
        "val_iou",
        "val_precision",
        "val_recall",
        "val_f1",
        "val_f2",
        "best_threshold",
        "log_file",
        "started_at",
        "finished_at",
    ]

    rows = []
    if args.resume and summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))

    completed = {
        (row["model"], row["loss"])
        for row in rows
        if row.get("status") == "success"
    }

    print(f"Experimentos planejados: {len(experiments)}")
    print(f"Saida: {output_dir}")
    print(f"GPU visivel: {env.get('CUDA_VISIBLE_DEVICES', 'padrao')}")
    print("Execucao sequencial: um treinamento por vez.\n")

    for index, (model, loss) in enumerate(experiments, start=1):
        experiment_name = f"{index:02d}_{slug(model)}_{slug(loss)}"
        log_path = logs_dir / f"{experiment_name}.log"
        success_path = logs_dir / f"{experiment_name}.success"

        if args.resume and (model, loss) in completed and success_path.exists():
            print(f"[{index}/{len(experiments)}] Pulando {model} + {loss}: ja concluido.")
            continue

        if loss not in loss_configs:
            raise ValueError(
                f"Nao existe configuracao para a loss '{loss}'. "
                f"Adicione-a em DEFAULT_LOSS_CONFIGS ou use --loss-configs."
            )

        loss_config_json = json.dumps(loss_configs[loss])
        command = [
            args.python,
            str(pipeline_path),
            "--config",
            str(config_path),
            "--model",
            model,
            "--epochs",
            str(args.epochs),
            "--loss",
            loss,
            "--loss-config",
            loss_config_json,
        ]

        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])

        started_at = datetime.now().isoformat(timespec="seconds")
        print("=" * 78)
        print(f"[{index}/{len(experiments)}] Modelo: {model} | Loss: {loss}")
        print("Comando:", subprocess.list2cmdline(command))
        print("Log:", log_path)

        with log_path.open("w", encoding="utf-8") as log_file:
            log_file.write("COMMAND: " + subprocess.list2cmdline(command) + "\n\n")
            log_file.flush()

            process = subprocess.Popen(
                command,
                cwd=project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )

            captured_lines = []
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
                captured_lines.append(line)

            return_code = process.wait()

        finished_at = datetime.now().isoformat(timespec="seconds")
        log_text = "".join(captured_lines)
        metrics = extract_metrics(log_text)
        status = "success" if return_code == 0 else "failed"

        if status == "success":
            success_path.write_text(finished_at, encoding="utf-8")
        elif success_path.exists():
            success_path.unlink()

        row = {
            "model": model,
            "loss": loss,
            "epochs": args.epochs,
            "status": status,
            "return_code": return_code,
            **metrics,
            "log_file": str(log_path.relative_to(project_root)),
            "started_at": started_at,
            "finished_at": finished_at,
        }
        rows = [
            old for old in rows
            if not (old.get("model") == model and old.get("loss") == loss)
        ]
        rows.append(row)

        with summary_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nResultado: {status} | codigo de saida: {return_code}")
        print(f"CSV atualizado: {summary_path}")

        if return_code != 0 and not args.continue_on_error:
            print(
                "\nA varredura foi interrompida porque um experimento falhou. "
                "Consulte o log acima. Use --continue-on-error para continuar."
            )
            break

    print("\n" + "=" * 78)
    print("VARREDURA FINALIZADA")
    print(f"CSV: {summary_path}")
    print(f"Logs: {logs_dir}")
    print(
        "\nConfira no CSV principalmente val_f2, val_recall, val_precision "
        "e val_iou. Compare os modelos usando a mesma divisao de treino/validacao."
    )


if __name__ == "__main__":
    main()
