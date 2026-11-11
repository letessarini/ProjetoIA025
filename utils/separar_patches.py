from pathlib import Path
import numpy as np
import shutil
import csv

# ============================================
# CONFIGURAÇÕES
# ============================================

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "DATASET_UNET" / "DATASET_UNET"
if not DATASET_DIR.exists():
    DATASET_DIR = ROOT / "DATASET_UNET"

PATCHES_DIR = DATASET_DIR / "patches"
if not PATCHES_DIR.exists():
    PATCHES_DIR = ROOT / "patches"

OUTPUT_DIR = ROOT / "patches_separados"

DIR_COM = OUTPUT_DIR / "com_inundacao"
DIR_SEM = OUTPUT_DIR / "sem_inundacao"

DIR_COM.mkdir(parents=True, exist_ok=True)
DIR_SEM.mkdir(parents=True, exist_ok=True)

arquivos = sorted(PATCHES_DIR.rglob("*.npz"))

com_inundacao = 0
sem_inundacao = 0
ignorados = 0

resultados = []

# ============================================
# PROCESSAMENTO
# ============================================

for i, arquivo in enumerate(arquivos, 1):

    try:
        with np.load(arquivo, allow_pickle=False) as dados:
            y = dados["y"]
            valido = dados["valido"]

            if y.shape != valido.shape:
                raise ValueError("Dimensões incompatíveis")

            mascara_valida = valido == 1
            inundacao_valida = (y == 1) & mascara_valida

            pixels_validos = int(mascara_valida.sum())
            pixels_inundados = int(inundacao_valida.sum())

            if pixels_validos == 0:
                ignorados += 1
                print(f"[IGNORADO] {arquivo.name}: sem pixels válidos")
                continue

            if pixels_inundados > 0:
                categoria = "com_inundacao"
                destino_dir = DIR_COM
                com_inundacao += 1
            else:
                categoria = "sem_inundacao"
                destino_dir = DIR_SEM
                sem_inundacao += 1

            # Preserva caminhos relativos, evitando sobrescrever
            # arquivos de subpastas diferentes com nomes iguais
            destino = destino_dir / arquivo.relative_to(PATCHES_DIR)
            destino.parent.mkdir(parents=True, exist_ok=True)

            # Copia sem alterar o arquivo original
            shutil.copy2(arquivo, destino)

            resultados.append({
                "arquivo": str(arquivo.relative_to(PATCHES_DIR)),
                "categoria": categoria,
                "pixels_validos": pixels_validos,
                "pixels_inundados": pixels_inundados,
                "percentual_inundacao": round(
                    100 * pixels_inundados / pixels_validos, 4
                )
            })

            print(
                f"[{i}/{len(arquivos)}] "
                f"{arquivo.name} -> {categoria} "
                f"({pixels_inundados} pixels inundados)"
            )

    except Exception as e:
        ignorados += 1
        print(f"[ERRO] {arquivo.name}: {e}")

# ============================================
# SALVAR CSV
# ============================================

csv_path = OUTPUT_DIR / "classificacao_patches.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "arquivo",
            "categoria",
            "pixels_validos",
            "pixels_inundados",
            "percentual_inundacao"
        ]
    )

    writer.writeheader()
    writer.writerows(resultados)

# ============================================
# RESUMO
# ============================================

print("\n" + "=" * 60)
print("RESUMO DA SEPARAÇÃO")
print("=" * 60)

print(f"Total de arquivos encontrados: {len(arquivos)}")
print(f"Patches com inundação: {com_inundacao}")
print(f"Patches sem inundação: {sem_inundacao}")
print(f"Patches ignorados/erro: {ignorados}")

print(f"\nArquivos separados em: {OUTPUT_DIR}")
print(f"CSV gerado em: {csv_path}")