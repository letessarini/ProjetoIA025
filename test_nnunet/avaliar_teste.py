
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=int, default=101)
parser.add_argument("--max-figuras", type=int, default=20)
args = parser.parse_args()

root = Path.home() / "ProjetoIA025"
raw = next((root / "nnUNet_raw").glob(f"Dataset{args.dataset}_*"))
gt_dir = raw / "labelsTs"
pred_dir = root / "resultados_teste" / f"Dataset{args.dataset}"
out = root / "resultados_teste" / f"avaliacao_Dataset{args.dataset}"
fig_dir = out / "figuras"
fig_dir.mkdir(parents=True, exist_ok=True)

registros = []
total_tp = total_fp = total_fn = total_tn = 0

arquivos = sorted(gt_dir.glob("*.tif"))
assert len(arquivos) == 216, f"Esperados 216 patches, encontrados {len(arquivos)}"

for i, gt_path in enumerate(arquivos):
    pred_path = pred_dir / gt_path.name
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)

    gt = np.squeeze(tifffile.imread(gt_path))
    pred = np.squeeze(tifffile.imread(pred_path))

    if gt.shape != pred.shape:
        raise ValueError(f"Dimensões diferentes: {gt_path.name}")

    valid = gt != 2
    real = (gt == 1) & valid
    previsto = (pred == 1) & valid

    tp = int(np.sum(real & previsto))
    fp = int(np.sum(~real & previsto & valid))
    fn = int(np.sum(real & ~previsto & valid))
    tn = int(np.sum(~real & ~previsto & valid))

    total_tp += tp
    total_fp += fp
    total_fn += fn
    total_tn += tn

    # Sem união: ambas as máscaras estão vazias.
    # Mantemos NaN para Dice/IoU e avaliamos FP separadamente.
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan
    iou = tp / (tp + fp + fn) if tp + fp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan

    registros.append({
        "patch": gt_path.stem,
        "tem_inundacao_real": bool(real.any()),
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Dice": dice, "IoU": iou,
        "Precision": precision, "Recall": recall,
    })

    if i < args.max_figuras:
        # 0 = fundo, 1 = TP, 2 = FP, 3 = FN, 4 = ignorado
        erros = np.zeros(gt.shape, dtype=np.uint8)
        erros[~valid] = 4
        erros[real & previsto] = 1
        erros[~real & previsto & valid] = 2
        erros[real & ~previsto & valid] = 3

        from matplotlib.colors import ListedColormap, BoundaryNorm
        cmap = ListedColormap([
            "#eeeeee", "#26966c", "#d84e4e",
            "#edae39", "#777777"
        ])
        norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)

        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        ax[0].imshow(np.ma.masked_where(~valid, gt), vmin=0, vmax=1, cmap="Blues")
        ax[0].set_title("Máscara real")
        ax[1].imshow(np.ma.masked_where(~valid, pred), vmin=0, vmax=1, cmap="Blues")
        ax[1].set_title("Previsão")
        ax[2].imshow(erros, cmap=cmap, norm=norm)
        ax[2].set_title("Verde: TP | Vermelho: FP | Amarelo: FN")

        for a in ax:
            a.axis("off")
        fig.suptitle(gt_path.stem)
        fig.tight_layout()
        fig.savefig(fig_dir / f"{gt_path.stem}.png", dpi=150)
        plt.close(fig)

df = pd.DataFrame(registros)
df.to_csv(out / "metricas_por_patch.csv", index=False)

def dividir(a, b):
    return a / b if b else np.nan

resumo = {
    "Dataset": args.dataset,
    "Patches": len(df),
    "Patches_positivos": int(df.tem_inundacao_real.sum()),
    "Patches_negativos": int((~df.tem_inundacao_real).sum()),
    "TP": total_tp, "FP": total_fp,
    "FN": total_fn, "TN": total_tn,
    "Dice_global": dividir(2 * total_tp, 2 * total_tp + total_fp + total_fn),
    "IoU_global": dividir(total_tp, total_tp + total_fp + total_fn),
    "Precision_global": dividir(total_tp, total_tp + total_fp),
    "Recall_global": dividir(total_tp, total_tp + total_fn),
    "Dice_medio_patches_definidos": df.Dice.mean(),
    "IoU_medio_patches_definidos": df.IoU.mean(),
    "FP_pixels_patches_negativos": int(
        df.loc[~df.tem_inundacao_real, "FP"].sum()
    ),
}
pd.DataFrame([resumo]).to_csv(out / "resumo.csv", index=False)

print("\n===== RESULTADOS DO TESTE =====")
for chave, valor in resumo.items():
    print(f"{chave}: {valor}")
print(f"\nArquivos salvos em: {out}")
