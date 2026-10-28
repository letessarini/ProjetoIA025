
from pathlib import Path
from collections import Counter
import numpy as np

PATCHES_DIR = Path("/home/leticia/ProjetoIA025/patches")

arquivos = sorted(PATCHES_DIR.rglob("*.npz"))

print("=" * 60)
print("VALIDAÇÃO DO DATASET U-NET")
print("=" * 60)
print(f"Diretório: {PATCHES_DIR}")
print(f"Total de patches: {len(arquivos)}")

if not arquivos:
    raise SystemExit("Nenhum arquivo .npz encontrado!")

campos_esperados = {
    "X", "y", "valido", "date", "precip_val", "frac_valida"
}

formatos_X = Counter()
formatos_y = Counter()
datas = Counter()

com_inundacao = 0
sem_inundacao = 0
pixels_inundados = 0
pixels_validos = 0
pixels_inundados_validos = 0

erros = []
exemplos_erros = []

for i, arquivo in enumerate(arquivos, 1):
    try:
        with np.load(arquivo, allow_pickle=False) as dados:
            campos = set(dados.files)
            faltantes = campos_esperados - campos

            if faltantes:
                raise ValueError(f"Campos ausentes: {faltantes}")

            X = dados["X"]
            y = dados["y"]
            valido = dados["valido"]
            data = str(dados["date"].item())

            formatos_X[X.shape] += 1
            formatos_y[y.shape] += 1
            datas[data] += 1

            if X.shape != (10, 256, 256):
                erros.append((arquivo.name, f"X: {X.shape}"))

            if y.shape != (256, 256):
                erros.append((arquivo.name, f"y: {y.shape}"))

            if valido.shape != (256, 256):
                erros.append((arquivo.name, f"valido: {valido.shape}"))

            if not np.isfinite(X).all():
                erros.append((arquivo.name, "X contém NaN ou infinito"))

            if not np.isin(y, [0, 1]).all():
                erros.append((arquivo.name, "y não é binário"))

            if not np.isin(valido, [0, 1]).all():
                erros.append((arquivo.name, "valido não é binário"))

            mascara = valido == 1
            inundacao = y == 1

            pixels_inundados += int(inundacao.sum())
            pixels_validos += int(mascara.sum())
            pixels_inundados_validos += int(
                (inundacao & mascara).sum()
            )

            if (inundacao & mascara).any():
                com_inundacao += 1
            else:
                sem_inundacao += 1

    except Exception as e:
        erros.append((arquivo.name, str(e)))

    if i % 100 == 0:
        print(f"Processados: {i}/{len(arquivos)}")

print("\n" + "=" * 60)
print("RESULTADOS")
print("=" * 60)

print("\nFormatos de X:")
for formato, quantidade in formatos_X.items():
    print(f"  {formato}: {quantidade}")

print("\nFormatos de y:")
for formato, quantidade in formatos_y.items():
    print(f"  {formato}: {quantidade}")

print(f"\nDatas diferentes: {len(datas)}")
print(f"Patches com inundação: {com_inundacao}")
print(f"Patches sem inundação: {sem_inundacao}")

print(f"\nPixels válidos: {pixels_validos:,}")
print(f"Pixels inundados (total): {pixels_inundados:,}")
print(f"Pixels inundados válidos: {pixels_inundados_validos:,}")

if pixels_validos:
    percentual = 100 * pixels_inundados_validos / pixels_validos
    print(f"Percentual de inundação em pixels válidos: {percentual:.4f}%")

print("\nPatches por data:")
for data, quantidade in sorted(datas.items()):
    print(f"  {data}: {quantidade}")

print(f"\nTotal de problemas encontrados: {len(erros)}")
for nome, problema in erros[:20]:
    print(f"  {nome}: {problema}")

if len(erros) > 20:
    print(f"  ... e mais {len(erros) - 20} problemas")

print("\nValidação finalizada.")
