


    
  
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