from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def regiao_do_patch(y0: int, x0: int, lado: int = 512) -> Tuple[int, int]:
    return (int(y0) // int(lado), int(x0) // int(lado))


def dividir_datas(datas: List[str], chuva: List[float], frac: Tuple[float, float, float] = (0.70, 0.15, 0.15), semente: int = 20261022) -> Dict[str, str]:
    """Split dates into train/val/test while preserving rainfall stratification."""
    ordem = sorted(range(len(datas)), key=lambda i: (-float(chuva[i]), str(datas[i])))
    n = len(ordem)
    alvo = [int(round(f * n)) for f in frac]
    alvo[0] = n - alvo[1] - alvo[2]

    rng = np.random.RandomState(semente)
    sequencia = (["treino"] * alvo[0] + ["val"] * alvo[1] + ["teste"] * alvo[2])
    rng.shuffle(sequencia)

    out: Dict[str, str] = {}
    for pos, i in enumerate(ordem):
        out[str(datas[i])] = sequencia[pos]
    return out


def dividir_regioes(regioes: List[Tuple[int, int]], frac_retida: float = 0.25, semente: int = 20261022) -> Dict[Tuple[int, int], str]:
    """Split regions into 'treino' and 'retida'."""
    regs = sorted(set(regioes))
    rng = np.random.RandomState(semente + 1)
    n_ret = max(1, int(round(frac_retida * len(regs))))
    idx = rng.permutation(len(regs))[:n_ret]
    retidas = {regs[i] for i in idx}
    return {r: ("retida" if r in retidas else "treino") for r in regs}


def conjunto_do_patch(papel_data: str, papel_regiao: str) -> str | None:
    """2x2 split table for train/val/test and spatial/temporal leakage control."""
    if papel_regiao == "treino":
        if papel_data == "treino":
            return "treino"
        if papel_data == "val":
            return "validacao"
        return "teste_tempo"
    if papel_data == "treino":
        return "teste_espaco"
    if papel_data == "teste":
        return "teste_ambos"
    return None


def montar_split(itens, lado: int = 512, frac: Tuple[float, float, float] = (0.70, 0.15, 0.15), frac_reg: float = 0.25, semente: int = 20261022):
    """itens = [(caminho, data, y0, x0, chuva_media)] -> dict conjunto -> lista."""
    datas, chuva = [], []
    vistos: Dict[str, float] = {}

    for _, d, _, _, mm in itens:
        if d not in vistos:
            vistos[d] = float(mm)

    datas = sorted(vistos)
    chuva = [vistos[d] for d in datas]
    papel_d = dividir_datas(datas, chuva, frac=frac, semente=semente)

    regs = [regiao_do_patch(y0, x0, lado) for _, _, y0, x0, _ in itens]
    papel_r = dividir_regioes(regs, frac_retida=frac_reg, semente=semente)

    saida: Dict[str, List[str]] = {}
    for (cam, d, y0, x0, _), r in zip(itens, regs):
        c = conjunto_do_patch(papel_d[d], papel_r[r])
        if c is None:
            continue
        saida.setdefault(c, []).append(cam)

    return saida, papel_d, papel_r


def conferir_vazamento(saida, itens, papel_d, papel_r, lado: int = 512):
    """Return problems in the split, or an empty list when there is no leakage."""
    meta = {cam: (d, regiao_do_patch(y0, x0, lado)) for cam, d, y0, x0, _ in itens}
    problemas = []

    def datas_de(c):
        return {meta[k][0] for k in saida.get(c, [])}

    def regs_de(c):
        return {meta[k][1] for k in saida.get(c, [])}

    if datas_de("treino") & datas_de("teste_tempo"):
        problemas.append("data de treino em teste_tempo")
    if datas_de("treino") & datas_de("teste_ambos"):
        problemas.append("data de treino em teste_ambos")
    if datas_de("validacao") & datas_de("teste_tempo"):
        problemas.append("data de validacao em teste_tempo")
    if regs_de("treino") & regs_de("teste_espaco"):
        problemas.append("regiao de treino em teste_espaco")
    if regs_de("treino") & regs_de("teste_ambos"):
        problemas.append("regiao de treino em teste_ambos")

    todos = [k for c in saida.values() for k in c]
    if len(todos) != len(set(todos)):
        problemas.append("o mesmo patch em mais de um conjunto")

    return problemas


if __name__ == "__main__":
    itens = []
    for di in range(12):
        for ry in range(3):
            for rx in range(4):
                itens.append((f"p_{di:02d}_{ry}_{rx}.npz", f"2024{di + 1:02d}01", ry * 512, rx * 512, 10.0 * di))

    saida, pd_, pr_ = montar_split(itens, lado=512)
    print("Conjuntos:", sorted(saida.keys()))
    print("Vazamento:", conferir_vazamento(saida, itens, pd_, pr_, lado=512))
