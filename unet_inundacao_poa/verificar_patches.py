"""
================================================================================
VERIFICACAO DOS PATCHES (.npz corrompidos, erro "Bad CRC-32")
================================================================================

Le todos os patches por inteiro, PASSADAS vezes, e separa:
  - arquivos que falham em todas as passadas: estao corrompidos;
  - arquivos que falham so em algumas: o arquivo esta bom e a falha e de leitura
    (memoria RAM ou disco).

Se COPIA_BOA apontar para outra copia da pasta patches, os arquivos corrompidos
sao trocados pela copia (desde que ela abra sem erro); o original e renomeado
para .corrompido.

USO
  python verificar_patches.py
================================================================================
"""

import glob
import os
import shutil
import time
import zipfile

import numpy as np

import config

COPIA_BOA = None          # ex.: "/outro/disco/DATASET_UNET/patches"
PASSADAS = 2


def arquivo_ok(caminho):
    """None se todas as matrizes do .npz abrem (o zipfile confere o CRC), senao a mensagem de erro."""
    try:
        with np.load(caminho, allow_pickle=False) as d:
            for chave in d.files:
                np.asarray(d[chave])
        return None
    except (zipfile.BadZipFile, ValueError, OSError, EOFError) as e:
        return "%s: %s" % (type(e).__name__, e)


def main():
    pasta = os.path.join(config.DADOS, "patches")
    arquivos = sorted(glob.glob(os.path.join(pasta, "*.npz")))
    if not arquivos:
        raise SystemExit("Nenhum .npz em %s (confira config.DADOS)." % pasta)
    print("%d patches em %s\n" % (len(arquivos), pasta))

    ruins_por_passada = []
    for passada in range(1, PASSADAS + 1):
        t0 = time.time()
        ruins = {}
        for i, cam in enumerate(arquivos, 1):
            erro = arquivo_ok(cam)
            if erro:
                ruins[cam] = erro
                print("  passada %d: ERRO em %s -> %s" % (passada, os.path.basename(cam), erro))
            if i % 200 == 0:
                print("  passada %d: %d/%d lidos" % (passada, i, len(arquivos)))
        print("Passada %d: %d arquivo(s) com erro (%.0f s)\n" % (passada, len(ruins), time.time() - t0))
        ruins_por_passada.append(ruins)

    sempre = set.intersection(*(set(r) for r in ruins_por_passada))
    as_vezes = set.union(*(set(r) for r in ruins_por_passada)) - sempre

    print("=" * 70)
    if not sempre and not as_vezes:
        print("Todos os patches abriram nas %d passadas." % PASSADAS)
    if sempre:
        print("Arquivos CORROMPIDOS (falharam em todas as passadas):")
        for cam in sorted(sempre):
            print("   ", cam)
    if as_vezes:
        print("Arquivos que falharam so em algumas passadas (falha de leitura, nao do arquivo):")
        for cam in sorted(as_vezes):
            print("   ", cam)
        print("\nVale testar a memoria RAM e verificar o disco.")
    print("=" * 70)

    if sempre and COPIA_BOA:
        print("\nTrocando os corrompidos pela copia em %s" % COPIA_BOA)
        for cam in sorted(sempre):
            boa = os.path.join(COPIA_BOA, os.path.basename(cam))
            if not os.path.exists(boa):
                print("  sem copia para %s" % os.path.basename(cam))
                continue
            erro = arquivo_ok(boa)
            if erro:
                print("  a copia de %s tambem esta corrompida (%s)" % (os.path.basename(cam), erro))
                continue
            os.replace(cam, cam + ".corrompido")
            shutil.copy2(boa, cam)
            print("  %s trocado (%s)" % (os.path.basename(cam), "ok" if arquivo_ok(cam) is None else "AINDA COM ERRO"))
    elif sempre:
        print("\nPara consertar: aponte COPIA_BOA para uma copia boa da pasta patches,")
        print("ou gere esses patches novamente.")


if __name__ == "__main__":
    main()
