
from pathlib import Path
import json
import numpy as np
import tifffile

# ============================================
# CONFIGURAÇÕES
# ============================================

ORIGEM = Path("/home/leticia/ProjetoIA025/patches")

DESTINO = Path(
    "/home/leticia/ProjetoIA025/nnUNet_raw/"
    "Dataset101_Inundacao"
)

images = DESTINO / "imagesTr"
labels = DESTINO / "labelsTr"

images.mkdir(parents=True, exist_ok=True)
labels.mkdir(parents=True, exist_ok=True)

# Todos os arquivos .npz
arquivos = sorted(ORIGEM.rglob("*.npz"))

com_inundacao = 0
sem_inundacao = 0
convertidos = 0
erros = []

print("=" * 60)
print("CONVERSÃO DO DATASET PARA nnU-Net v2")
print("=" * 60)
print(f"Total de arquivos encontrados: {len(arquivos)}")

# ============================================
# CONVERSÃO
# ============================================

for idx, arquivo in enumerate(arquivos):

    try:
        with np.load(arquivo, allow_pickle=False) as dados:
            X = dados["X"].astype(np.float32)
            y = dados["y"].copy()
            valido = dados["valido"]

        if X.shape != (10, 256, 256):
            raise ValueError(f"Formato X inválido: {X.shape}")

        if y.shape != (256, 256):
            raise ValueError(f"Formato y inválido: {y.shape}")

        if valido.shape != (256, 256):
            raise ValueError(f"Formato valido inválido: {valido.shape}")

        if not np.isfinite(X).all():
            raise ValueError("X contém NaN ou infinito")

        if not np.isin(y, [0, 1]).all():
            raise ValueError("y não é binário")

        if not np.isin(valido, [0, 1]).all():
            raise ValueError("valido não é binário")

        if not np.array_equal(X[9], valido):
            raise ValueError("Canal 9 diferente da máscara valido")

        mascara_valida = valido == 1
        inundacao_valida = (y == 1) & mascara_valida

        if not mascara_valida.any():
            raise ValueError("Patch sem pixels válidos")

        if inundacao_valida.any():
            com_inundacao += 1
        else:
            sem_inundacao += 1

        # Pixels inválidos recebem o valor ignore=2
        y[valido == 0] = 2

        case_id = f"patch_{idx:04d}"

        # Exportar os 9 canais ambientais
        for canal in range(9):
            tifffile.imwrite(
                images / f"{case_id}_{canal:04d}.tif",
                X[canal]
            )

        # Exportar máscara de inundação
        tifffile.imwrite(
            labels / f"{case_id}.tif",
            y.astype(np.uint8)
        )

        convertidos += 1

        print(
            f"[{idx + 1}/{len(arquivos)}] "
            f"{arquivo.name} -> {case_id} "
            f"| Inundados: {int(inundacao_valida.sum())}"
        )

    except Exception as e:
        erros.append((arquivo.name, str(e)))
        print(f"[ERRO] {arquivo.name}: {e}")

# ============================================
# DATASET.JSON
# ============================================

dataset_json = {
    "channel_names": {
        str(i): f"canal_{i}" for i in range(9)
    },
    "labels": {
        "background": 0,
        "inundacao": 1,
        "ignore": 2
    },
    "numTraining": convertidos,
    "file_ending": ".tif"
}

with open(DESTINO / "dataset.json", "w", encoding="utf-8") as f:
    json.dump(dataset_json, f, indent=4, ensure_ascii=False)

# ============================================
# RESUMO FINAL
# ============================================

resumo = f"""
============================================================
RESUMO DA CONVERSÃO PARA nnU-Net
============================================================

Total de patches encontrados: {len(arquivos)}
Patches convertidos: {convertidos}

Patches COM inundação: {com_inundacao}
Patches SEM inundação: {sem_inundacao}

Erros encontrados: {len(erros)}

Total de canais ambientais: 9
Formato das imagens: TIFF
Formato das máscaras: TIFF

Diretório de saída:
{DESTINO}

============================================================
"""

print(resumo)

with open(DESTINO / "resumo_conversao.txt", "w", encoding="utf-8") as f:
    f.write(resumo)

if erros:
    print("Arquivos com problemas:")
    for nome, erro in erros:
        print(f"  {nome}: {erro}")

print("Conversão finalizada!")
