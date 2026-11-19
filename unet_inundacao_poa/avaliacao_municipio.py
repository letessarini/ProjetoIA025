"""
================================================================================
AVALIACAO NA ESCALA DO MUNICIPIO (mapa reconstruido por data)
================================================================================

A rede sempre preve patches de 256 x 256 px. Este script compara jeitos de
CONTAR os acertos dessas previsoes nas datas de teste:

  media_por_patch           IoU de cada patch, depois a media simples (cada patch
                            pesa igual, tenha muita ou pouca agua)
  soma_dos_patches          soma VP, FP e FN de todos os patches e calcula uma IoU
                            (e o numero reportado pelo treinar_unet.py)
  mapa                      as previsoes dos patches sao encaixadas lado a lado no
                            mapa de cada data; cada pixel e contado uma vez
  mapa_janelas_sobrepostas  o modelo roda em janelas que andam meio patch por vez
                            e as previsoes sao combinadas com peso maior no centro
  climatologia_mapa         a climatologia no mesmo mapa, como referencia

Zonas: regioes_treino (datas novas), regioes_retidas (datas e lugares novos),
dominio (tudo) e, se existir a mascara gerada pelo mascara_municipio.py:
  municipio                 tudo dentro do limite municipal
  municipio_regioes_treino  dentro do limite e fora das regioes retidas

Os limiares do modelo (por patch e com janelas) e da climatologia sao escolhidos
nas datas de VALIDACAO. A area inundada (km2) real e prevista sai por data.

SAIDA (em config.RESULTADOS/municipio_<experimento>)
  avaliacao_municipio.csv           uma linha por (data, zona, medida)
  resumo_municipio.csv              as medidas somando todas as datas de teste
  comparacao_escalas.png            IoU por data em cada medida
  mapa_erro_<data>.png              mapa de erro com o limite municipal

USO
  python avaliacao_municipio.py
  python avaliacao_municipio.py --experimento R10_r34_sem_urbano
================================================================================
"""

import argparse
import csv
import os

import numpy as np

import comum
import config

PASSO = 128                    # passo das janelas sobrepostas (128 = 50% de sobreposicao)
GERAR_FIGURAS = True

MEDIDAS = {"media_por_patch": "media por patch", "soma_dos_patches": "soma dos patches", "mapa": "mapa",
           "mapa_janelas_sobrepostas": "mapa, janelas sobrepostas", "climatologia_mapa": "climatologia (mapa)"}
ZONAS = ("regioes_treino", "regioes_retidas", "dominio", "municipio", "municipio_regioes_treino")


# =============================================================================
# 1. MAPA DE UMA DATA
# =============================================================================

def montar_mosaico(cams, idx, forma, lado):
    """Junta os patches de uma data no mapa da grade.

    Devolve X, y, valido, cobertura e um dicionario com os pixels validos que a
    soma dos patches conta e o mapa nao: os que caem fora da grade e os contados
    mais de uma vez (patches sobrepostos)."""
    X = np.zeros((len(idx),) + forma, dtype="float32")
    y = np.zeros(forma, dtype="float32")
    v = np.zeros(forma, dtype="float32")
    vezes = np.zeros(forma, dtype="int16")
    fora = 0
    for cam in cams:
        y0, x0 = comum.posicao_do_nome(cam)
        Xp, yp, vp = comum.ler_patch(cam, idx)
        h, w = min(lado, forma[0] - y0), min(lado, forma[1] - x0)
        X[:, y0:y0 + h, x0:x0 + w] = Xp[:, :h, :w]
        y[y0:y0 + h, x0:x0 + w] = yp[0, :h, :w]
        v[y0:y0 + h, x0:x0 + w] = vp[0, :h, :w]
        vezes[y0:y0 + h, x0:x0 + w] += 1
        fora += int((vp[0] > 0.5).sum() - (vp[0, :h, :w] > 0.5).sum())
    repetidos = int((np.maximum(vezes - 1, 0) * (v > 0.5)).sum())
    return X, y, v, vezes > 0, {"px_fora_da_grade": fora, "px_repetidos": repetidos}


def peso_janela(lado):
    """Peso maior no centro da janela e menor nas bordas (onde a rede ve menos contexto)."""
    w = np.hanning(lado) * 0.9 + 0.1
    return np.outer(w, w).astype("float32")


def prever_com_sobreposicao(prever_janelas, X, cobertura, lado, passo, p_reserva):
    """Media ponderada das previsoes em janelas sobrepostas. Pixels sem nenhuma
    janela completa ficam com a previsao por patch (p_reserva)."""
    H, W = cobertura.shape
    janelas = [(a, b) for a in range(0, H - lado + 1, passo) for b in range(0, W - lado + 1, passo)
               if cobertura[a:a + lado, b:b + lado].all()]
    soma = np.zeros((H, W), dtype="float32")
    pesos = np.zeros((H, W), dtype="float32")
    peso = peso_janela(lado)
    for k in range(0, len(janelas), 64):
        bloco = janelas[k:k + 64]
        P = prever_janelas(np.stack([X[:, a:a + lado, b:b + lado] for a, b in bloco]))
        for (a, b), p in zip(bloco, P):
            soma[a:a + lado, b:b + lado] += p * peso
            pesos[a:a + lado, b:b + lado] += peso
    return np.where(pesos > 0, soma / np.maximum(pesos, 1e-9), p_reserva), len(janelas)


def checar_costuras(X, cobertura, lado):
    """Razao entre o salto dos canais de entrada na emenda de dois patches e o salto
    entre colunas vizinhas no meio do patch (~1 = canais continuos entre patches)."""
    H, W = cobertura.shape
    emenda, meio = [], []
    for x in range(lado, W, lado):
        m = cobertura[:, x] & cobertura[:, x - 1]
        if m.sum() < 10 or x + lado // 2 >= W:
            continue
        m2 = cobertura[:, x + lado // 2] & cobertura[:, x + lado // 2 - 1] & m
        emenda.append(np.abs(X[:, m, x] - X[:, m, x - 1]).mean(axis=1))
        meio.append(np.abs(X[:, m2, x + lado // 2] - X[:, m2, x + lado // 2 - 1]).mean(axis=1))
    if not emenda:
        return None
    return np.mean(emenda, axis=0) / np.maximum(np.mean(meio, axis=0), 1e-6)


# =============================================================================
# 2. METRICAS
# =============================================================================

def contagens(pred, real, valido):
    tp, fp, fn, _ = comum.contar(pred, real, valido)
    return tp, fp, fn


def linha_metricas(tp, fp, fn, km2_por_px):
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"iou": tp / (tp + fp + fn) if tp + fp + fn else float("nan"),
            "f1": 2 * prec * rec / (prec + rec) if (tp and prec + rec) else 0.0,
            "precisao": prec, "recall": rec,
            "area_real_km2": (tp + fn) * km2_por_px, "area_prevista_km2": (tp + fp) * km2_por_px,
            "tp": tp, "fp": fp, "fn": fn}


# =============================================================================
# 3. FIGURAS
# =============================================================================

def figura_escalas(caminho, por_data, nome_modelo):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datas = sorted(por_data)
    medidas = [("media_por_patch", "Média por patch", "#7FD1E0"), ("soma_dos_patches", "Soma dos patches", "#38A3A5"),
               ("mapa", "Mapa", "#1B6F8F"), ("climatologia_mapa", "Climatologia (mapa)", "#5A6B7B")]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    larg = 0.2
    xs = np.arange(len(datas))
    for i, (k, rot, cor) in enumerate(medidas):
        ax.bar(xs + (i - 1.5) * larg, [por_data[d].get(k, np.nan) for d in datas], larg, label=rot, color=cor)
    ax.set_xticks(xs)
    ax.set_xticklabels([comum.data_br(d) for d in datas])
    ax.set_ylabel("IoU no dominio")
    ax.set_title("%s: IoU por data de teste em cada forma de contagem" % nome_modelo)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figura_mapa_erro(caminho, real, pred, valido, titulo, contorno=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    e = np.zeros(real.shape, dtype="uint8")
    p, r, v = pred.astype(bool), real > 0.5, valido > 0.5
    e[p & r], e[p & ~r], e[~p & r], e[~v] = 1, 2, 3, 4
    cores = ["#F4F8FA", "#1B6F8F", "#D98324", "#E9C46A", "#C9D1D6"]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(e, cmap=ListedColormap(cores), norm=BoundaryNorm(np.arange(-0.5, 5.5, 1), 5), interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(titulo)
    itens = [Patch(color=c, label=l) for c, l in zip(cores[1:], ["acerto de agua", "alarme falso",
                                                                "agua perdida", "fora do dominio"])]
    if contorno is not None and contorno.any():
        ax.contour(contorno.astype("float32"), levels=[0.5], colors="#C0392B", linewidths=0.8)
        itens.append(Line2D([], [], color="#C0392B", lw=1.2, label="limite municipal"))
    fig.legend(handles=itens, loc="lower center", ncol=3 if len(itens) > 4 else 4, frameon=False)
    fig.tight_layout(rect=[0, 0.08 if len(itens) > 4 else 0.05, 1, 1])
    fig.savefig(caminho, dpi=250)
    plt.close(fig)


# =============================================================================
# 4. PROGRAMA PRINCIPAL
# =============================================================================

def carregar_mascara(caminho, forma):
    if not caminho or not os.path.isfile(caminho):
        print("Sem mascara do municipio (rode o mascara_municipio.py para ter a zona 'municipio').")
        return None
    municipio = np.load(caminho).astype(bool)
    if municipio.shape != forma:
        raise SystemExit("A mascara do municipio tem forma %s, mas a grade e %s." % (municipio.shape, forma))
    print("Mascara do municipio: %s" % caminho)
    return municipio


def main(experimento=None, caminho_mascara=None, carregar=comum.carregar_modelo):
    experimento = experimento or config.EXPERIMENTO_FINAL
    destino = os.path.join(config.RESULTADOS, "municipio_%s" % experimento)
    os.makedirs(destino, exist_ok=True)
    man, itens = comum.carregar_itens()
    canais = list(man["canais"])
    saida, _, papel_r = comum.montar_split(itens)
    lado = int(man["patch"]["tamanho"])
    forma = tuple(man["grade"]["forma"])
    km2_por_px = (config.TAMANHO_PIXEL_M ** 2) / 1e6
    nome_modelo = experimento.split("_")[0]
    print("Grade %d x %d px; patches de %d px; janelas a cada %d px" % (forma[0], forma[1], lado, PASSO))

    ry = (np.arange(forma[0]) // config.LADO_REGIAO_PX)[:, None]
    rx = (np.arange(forma[1]) // config.LADO_REGIAO_PX)[None, :]
    retida = np.zeros(forma, dtype=bool)
    for (a, b), papel in papel_r.items():
        if papel == "retida":
            retida |= (ry == a) & (rx == b)
    municipio = carregar_mascara(caminho_mascara or os.path.join(config.RESULTADOS, "mascara_municipio.npy"), forma)
    if municipio is not None:
        print("  %.1f km2 do municipio na grade, %.1f km2 fora das regioes retidas"
              % (municipio.sum() * km2_por_px, (municipio & ~retida).sum() * km2_por_px))

    prever_janelas, idx, _ = carregar(experimento, canais, config.RESULTADOS)
    freq = comum.climatologia_na_grade(saida["treino"], forma)

    def por_data(conjuntos):
        grupos = {}
        for c in conjuntos:
            for cam in saida.get(c, []):
                grupos.setdefault(comum.data_do_nome(cam), []).append(cam)
        return grupos

    def prever_data(cams):
        X, y, v, cob, info = montar_mosaico(cams, idx, forma, lado)
        probs = comum.prever_arquivos(prever_janelas, cams, idx)
        P_patch = np.zeros(forma, dtype="float32")
        for cam, p in zip(cams, probs):
            a, b = comum.posicao_do_nome(cam)
            h, w = min(lado, forma[0] - a), min(lado, forma[1] - b)
            P_patch[a:a + h, b:b + w] = p[:h, :w]
        P_sob, n_jan = prever_com_sobreposicao(prever_janelas, X, cob, lado, PASSO, P_patch)
        return X, y, v, cob, info, P_patch, P_sob, probs, n_jan

    # ---- 1. limiares nas datas de validacao ----
    print("\nEscolhendo os limiares nas datas de validacao...")
    pp, ps, ys_, vs_ = [], [], [], []
    for _, cams in sorted(por_data(["validacao"]).items()):
        _, y, v, cob, _, P_patch, P_sob, _, _ = prever_data(cams)
        pp.append(P_patch[cob]); ps.append(P_sob[cob]); ys_.append(y[cob]); vs_.append(v[cob])
    lim_patch, lim_sob = comum.melhor_limiar(pp, ys_, vs_), comum.melhor_limiar(ps, ys_, vs_)
    lim_clim = comum.melhor_limiar([comum.recortar(freq, *comum.posicao_do_nome(c), lado) for c in saida["validacao"]],
                                   *comum.alvos_e_validos(saida["validacao"]))
    print("  limiar por patch %.3f | janelas sobrepostas %.3f | climatologia %.3f" % (lim_patch, lim_sob, lim_clim))
    del pp, ps, ys_, vs_

    # ---- 2. datas de teste ----
    registros, iou_por_data, totais, ious_patch, costuras = [], {}, {}, {}, []
    extra = {"px_fora_da_grade": 0, "px_repetidos": 0}
    for d, cams in sorted(por_data(["teste_tempo", "teste_ambos"]).items()):
        X, y, v, cob, info, P_patch, P_sob, probs, n_jan = prever_data(cams)
        for k in extra:
            extra[k] += info[k]
        r = checar_costuras(X, cob, lado)
        if r is not None:
            costuras.append(r)
        zonas = {"regioes_treino": cob & ~retida, "regioes_retidas": cob & retida, "dominio": cob}
        if municipio is not None:
            zonas["municipio"] = cob & municipio
            zonas["municipio_regioes_treino"] = cob & municipio & ~retida
        clim = np.zeros(forma, dtype="float32")
        for cam in cams:
            a, b = comum.posicao_do_nome(cam)
            h, w = min(lado, forma[0] - a), min(lado, forma[1] - b)
            clim[a:a + h, b:b + w] = comum.recortar(freq, a, b, lado)[:h, :w]
        prev = {"mapa": P_patch >= lim_patch, "mapa_janelas_sobrepostas": P_sob >= lim_sob,
                "climatologia_mapa": clim >= lim_clim}

        # contagem por patch
        por_patch = {}
        for cam, p in zip(cams, probs):
            _, yp, vp = comum.ler_patch(cam, [0])
            cs = contagens(p >= lim_patch, yp[0], vp[0])
            a, b = comum.posicao_do_nome(cam)
            for z in ("regioes_retidas" if retida[a, b] else "regioes_treino", "dominio"):
                por_patch.setdefault(z, []).append(cs)
        for z, lst in por_patch.items():
            ious = [tp / (tp + fp + fn) for tp, fp, fn in lst if tp + fp + fn]
            ious_patch.setdefault(z, []).extend(ious)
            tp, fp, fn = (int(s) for s in np.sum(lst, axis=0))
            registros.append(dict(data=d, zona=z, medida="media_por_patch", n_patches=len(lst),
                                  **dict(linha_metricas(tp, fp, fn, km2_por_px),
                                         iou=float(np.mean(ious)) if ious else float("nan"))))
            registros.append(dict(data=d, zona=z, medida="soma_dos_patches", n_patches=len(lst),
                                  **linha_metricas(tp, fp, fn, km2_por_px)))
            t = totais.setdefault((z, "soma_dos_patches"), [0, 0, 0])
            t[0] += tp; t[1] += fp; t[2] += fn

        # contagem no mapa
        for z, m in zonas.items():
            for medida, pred in prev.items():
                tp, fp, fn = contagens(pred[m], y[m], v[m])
                registros.append(dict(data=d, zona=z, medida=medida, n_patches=len(cams),
                                      **linha_metricas(tp, fp, fn, km2_por_px)))
                t = totais.setdefault((z, medida), [0, 0, 0])
                t[0] += tp; t[1] += fp; t[2] += fn

        iou_por_data[d] = res = {r_["medida"]: r_["iou"] for r_ in registros
                                 if r_["data"] == d and r_["zona"] == "dominio"}
        print("  %s: %2d patches, %3d janelas | IoU no dominio: media/patch %.3f  soma %.3f  mapa %.3f  janelas %.3f"
              "  climatologia %.3f" % (d, len(cams), n_jan, res["media_por_patch"], res["soma_dos_patches"],
                                       res["mapa"], res["mapa_janelas_sobrepostas"], res["climatologia_mapa"]))
        if GERAR_FIGURAS:
            figura_mapa_erro(os.path.join(destino, "mapa_erro_%s.png" % d), y, prev["mapa"], v * cob,
                             "%s - IoU %.2f" % (comum.data_br(d), res["mapa"]), contorno=municipio)

    # ---- 3. gravar e resumir ----
    campos = ["data", "zona", "medida", "n_patches", "iou", "f1", "precisao", "recall",
              "area_real_km2", "area_prevista_km2", "tp", "fp", "fn"]
    with open(os.path.join(destino, "avaliacao_municipio.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader(); w.writerows(registros)

    resumo = []
    for (z, medida), (tp, fp, fn) in sorted(totais.items()):
        por_d = [r_["iou"] for r_ in registros if r_["zona"] == z and r_["medida"] == medida]
        resumo.append(dict(zona=z, medida=medida, iou_media_das_datas=float(np.nanmean(por_d)),
                           iou_dp_das_datas=float(np.nanstd(por_d, ddof=1)) if len(por_d) > 1 else float("nan"),
                           **linha_metricas(tp, fp, fn, km2_por_px)))
    for z, ious in ious_patch.items():
        resumo.append(dict(zona=z, medida="media_por_patch", iou=float(np.mean(ious)), n_patches=len(ious)))
    with open(os.path.join(destino, "resumo_municipio.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["zona", "medida", "n_patches", "iou", "iou_media_das_datas",
                                          "iou_dp_das_datas", "f1", "precisao", "recall", "area_real_km2",
                                          "area_prevista_km2", "tp", "fp", "fn"], extrasaction="ignore")
        w.writeheader(); w.writerows(resumo)

    print("\n" + "=" * 92)
    print("RESUMO (somando as %d datas de teste)" % len(iou_por_data))
    print("=" * 92)
    for z in ZONAS:
        linhas_z = [r_ for r_ in resumo if r_["zona"] == z]
        if not linhas_z:
            continue
        print("\n%s" % z)
        print("  %-28s %-7s %-16s %-7s %-7s %-7s %s" % ("medida", "IoU", "media+-dp datas", "F1", "prec", "recall",
                                                          "area real -> prevista (km2)"))
        for medida, nome in MEDIDAS.items():
            r_ = next((x for x in linhas_z if x["medida"] == medida), None)
            if not r_:
                continue
            if medida == "media_por_patch":
                print("  %-28s %.3f   (%d patches)" % (nome, r_["iou"], r_["n_patches"]))
                continue
            print("  %-28s %.3f   %.3f +- %.3f   %.3f   %.3f   %.3f   %.1f -> %.1f"
                  % (nome, r_["iou"], r_["iou_media_das_datas"], r_["iou_dp_das_datas"], r_["f1"],
                     r_["precisao"], r_["recall"], r_["area_real_km2"], r_["area_prevista_km2"]))

    print("\nPixels validos contados na soma dos patches e nao no mapa: %d fora da grade, %d repetidos"
          " (patches sobrepostos)." % (extra["px_fora_da_grade"], extra["px_repetidos"]))
    if not any(extra.values()):
        print("  -> soma_dos_patches e mapa usam exatamente os mesmos pixels.")
    if costuras:
        razao = np.max(np.mean(costuras, axis=0))
        print("Continuidade dos canais nas emendas entre patches: %.2f %s"
              % (razao, "(ok)" if razao < 3 else "<- alta: canais normalizados por patch?"))
    if GERAR_FIGURAS and iou_por_data:
        figura_escalas(os.path.join(destino, "comparacao_escalas.png"), iou_por_data, nome_modelo)
    print("\nResultados em %s" % destino)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Avaliacao do modelo no mapa reconstruido do municipio.")
    ap.add_argument("--experimento", help="padrao: config.EXPERIMENTO_FINAL")
    ap.add_argument("--mascara", help="padrao: <RESULTADOS>/mascara_municipio.npy")
    args = ap.parse_args()
    try:
        main(args.experimento, args.mascara)
    except comum.PatchCorrompido as e:
        raise SystemExit("\nPatch corrompido: %s\nRode o verificar_patches.py." % e)
