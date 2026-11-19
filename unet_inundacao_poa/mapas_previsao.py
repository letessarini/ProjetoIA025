"""
================================================================================
MAPAS DE MANCHA: REAL x PREVISTA x ERRO
================================================================================

Usa um modelo treinado (padrao: config.EXPERIMENTO_FINAL) e gera, em
config.RESULTADOS/mapas_<experimento>:

  mosaico_<data>.png        o dominio inteiro numa data de teste: mancha real,
                            mancha prevista e mapa de erro, com as regioes
                            retidas (nunca vistas no treino) tracejadas
  painel_<conjunto>.png     os patches com mais agua de cada conjunto de teste:
                            real | previsto | erro do modelo | erro da climatologia
  patches/<patch>.png       uma figura por patch do painel
  metricas_por_patch.csv    IoU do modelo e da climatologia em cada figura

Cores do mapa de erro: azul = acerto de agua (VP), laranja = alarme falso (FP),
amarelo = agua perdida (FN), cinza = fora do dominio simulado.

USO
  python mapas_previsao.py
  python mapas_previsao.py --experimento R10_r34_sem_urbano --datas 20240501 20240118
================================================================================
"""

import argparse
import csv
import os

import numpy as np

import comum
import config

N_DATAS_AUTO = 2                                   # sem --datas: as 2 datas de teste com mais agua
PAINEIS = {"teste_tempo": 4, "teste_espaco": 2}    # patches mostrados de cada conjunto
DPI = 200
DPI_MOSAICO = 300

COR = {"seco": "#F4F8FA", "agua": "#1B6F8F", "tp": "#1B6F8F", "fp": "#D98324",
       "fn": "#E9C46A", "invalido": "#C9D1D6", "texto": "#0B2540"}


# =============================================================================
# 1. MAPAS DE ERRO E METRICAS
# =============================================================================

def mapa_de_erro(pred, real, valido):
    """0 = seco certo, 1 = acerto de agua, 2 = alarme falso, 3 = agua perdida, 4 = invalido."""
    e = np.zeros(real.shape, dtype="uint8")
    p, r, v = pred.astype(bool), real > 0.5, valido > 0.5
    e[p & r] = 1
    e[p & ~r] = 2
    e[~p & r] = 3
    e[~v] = 4
    return e


def iou(pred, real, valido):
    tp, fp, fn, _ = comum.contar(pred, real, valido)
    return tp / (tp + fp + fn) if (tp + fp + fn) else float("nan"), tp, fp, fn


# =============================================================================
# 2. FIGURAS
# =============================================================================

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch, Rectangle
    return plt, ListedColormap, BoundaryNorm, Patch, Rectangle


def _mostrar_mascara(ax, mascara, valido, ListedColormap, BoundaryNorm):
    m = np.where(valido > 0.5, (mascara > 0.5).astype(int), 2)
    cmap = ListedColormap([COR["seco"], COR["agua"], COR["invalido"]])
    ax.imshow(m, cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3), interpolation="nearest")


def _mostrar_erro(ax, erro, ListedColormap, BoundaryNorm):
    cmap = ListedColormap([COR["seco"], COR["tp"], COR["fp"], COR["fn"], COR["invalido"]])
    ax.imshow(erro, cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 5.5, 1), 5), interpolation="nearest")


def _legenda(fig, Patch):
    itens = [Patch(color=COR["tp"], label="acerto de agua"), Patch(color=COR["fp"], label="alarme falso"),
             Patch(color=COR["fn"], label="agua perdida"), Patch(color=COR["invalido"], label="fora do dominio")]
    fig.legend(handles=itens, loc="lower center", ncol=4, frameon=False, fontsize=10)


def figura_mosaico(caminho, real, pred, valido, retidas, lado_regiao, titulo, iou_m, iou_c, nome_modelo):
    plt, ListedColormap, BoundaryNorm, Patch, Rectangle = _plt()
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.8))
    _mostrar_mascara(ax[0], real, valido, ListedColormap, BoundaryNorm)
    _mostrar_mascara(ax[1], pred, valido, ListedColormap, BoundaryNorm)
    _mostrar_erro(ax[2], mapa_de_erro(pred, real, valido), ListedColormap, BoundaryNorm)
    ax[0].set_title("Mancha real (SFINCS)", fontsize=12, color=COR["texto"])
    ax[1].set_title("Mancha prevista (%s)" % nome_modelo, fontsize=12, color=COR["texto"])
    ax[2].set_title("Erro: IoU %.2f (climatologia: %.2f)" % (iou_m, iou_c), fontsize=12, color=COR["texto"])
    for a in ax:
        for (ry, rx) in retidas:
            a.add_patch(Rectangle((rx * lado_regiao - 0.5, ry * lado_regiao - 0.5), lado_regiao, lado_regiao,
                                  fill=False, ls="--", lw=1.4, ec=COR["texto"]))
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(titulo + "   (tracejado = regioes nunca vistas no treino)", fontsize=13, color=COR["texto"])
    _legenda(fig, Patch)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(caminho, dpi=DPI_MOSAICO); plt.close(fig)


def figura_patches(caminho, linhas, titulo, nome_modelo):
    """linhas = [(rotulo, real, pred_modelo, pred_clim, valido, iou_m, iou_c)]."""
    plt, ListedColormap, BoundaryNorm, Patch, _ = _plt()
    n = len(linhas)
    fig, ax = plt.subplots(n, 4, figsize=(12.5, 3.1 * n + 0.8), squeeze=False)
    for i, (rot, real, pm, pc, v, im, ic) in enumerate(linhas):
        _mostrar_mascara(ax[i, 0], real, v, ListedColormap, BoundaryNorm)
        _mostrar_mascara(ax[i, 1], pm, v, ListedColormap, BoundaryNorm)
        _mostrar_erro(ax[i, 2], mapa_de_erro(pm, real, v), ListedColormap, BoundaryNorm)
        _mostrar_erro(ax[i, 3], mapa_de_erro(pc, real, v), ListedColormap, BoundaryNorm)
        ax[i, 0].set_ylabel(rot, fontsize=9.5, color=COR["texto"])
        ax[i, 2].set_xlabel("IoU %.2f" % im if not np.isnan(im) else "IoU -", fontsize=10, color=COR["texto"])
        ax[i, 3].set_xlabel("IoU %.2f" % ic if not np.isnan(ic) else "IoU -", fontsize=10, color=COR["texto"])
        for a in ax[i]:
            a.set_xticks([]); a.set_yticks([])
    for j, t in enumerate(["Mancha real", "Prevista (%s)" % nome_modelo, "Erro do modelo", "Erro da climatologia"]):
        ax[0, j].set_title(t, fontsize=11.5, color=COR["texto"])
    fig.suptitle(titulo, fontsize=13, color=COR["texto"])
    _legenda(fig, Patch)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(caminho, dpi=DPI); plt.close(fig)


# =============================================================================
# 3. PROGRAMA PRINCIPAL
# =============================================================================

def escolher_patches(arquivos, n):
    """Os n patches com mais agua real, de preferencia de datas diferentes."""
    area = []
    for cam in arquivos:
        _, y, v = comum.ler_patch(cam, [0])
        area.append(int(((y[0] > 0.5) & (v[0] > 0.5)).sum()))
    ordem = [arquivos[i] for i in np.argsort(area)[::-1] if area[i] > 0]
    escolhidos, datas = [], set()
    for cam in ordem:
        if comum.data_do_nome(cam) not in datas:
            escolhidos.append(cam); datas.add(comum.data_do_nome(cam))
        if len(escolhidos) == n:
            return escolhidos
    for cam in ordem:
        if cam not in escolhidos:
            escolhidos.append(cam)
        if len(escolhidos) == n:
            break
    return escolhidos


def main(experimento=None, datas_mosaico=None, carregar=comum.carregar_modelo):
    experimento = experimento or config.EXPERIMENTO_FINAL
    destino = os.path.join(config.RESULTADOS, "mapas_%s" % experimento)
    os.makedirs(os.path.join(destino, "patches"), exist_ok=True)
    man, itens = comum.carregar_itens()
    canais = list(man["canais"])
    saida, papel_d, papel_r = comum.montar_split(itens)
    lado = int(man["patch"]["tamanho"])
    forma = tuple(man["grade"]["forma"])
    chuva = {d: mm for _, d, _, _, mm in itens}
    nome_modelo = experimento.split("_")[0]

    print("Carregando %s ..." % experimento)
    prever_janelas, idx, _ = carregar(experimento, canais, config.RESULTADOS)
    prever = lambda cams: comum.prever_arquivos(prever_janelas, cams, idx)

    # limiares escolhidos na validacao (lidos do resultados.csv ou recalculados)
    limiar = comum.ler_limiar(experimento)
    if limiar is None:
        limiar = comum.melhor_limiar(prever(saida["validacao"]), *comum.alvos_e_validos(saida["validacao"]))
    freq = comum.climatologia_na_grade(saida["treino"], forma)
    lim_clim = comum.ler_limiar("0_climatologia")
    if lim_clim is None:
        val = saida["validacao"]
        lim_clim = comum.melhor_limiar([comum.recortar(freq, *comum.posicao_do_nome(c), lado) for c in val],
                                       *comum.alvos_e_validos(val))
    print("Limiar do modelo: %.3f | limiar da climatologia: %.3f" % (limiar, lim_clim))

    registros = []

    # ---- 1. mosaicos de datas de teste inteiras ----
    datas_teste = sorted(d for d, p in papel_d.items() if p == "teste")
    por_data = {d: [cam for cam, dd, _, _, _ in itens if dd == d] for d in datas_teste}
    if not datas_mosaico:
        agua = {d: sum(int((comum.ler_patch(c, [0])[1][0] > 0.5).sum()) for c in cams)
                for d, cams in por_data.items()}
        datas_mosaico = sorted(agua, key=agua.get, reverse=True)[:N_DATAS_AUTO]
    retidas = [r for r, p in papel_r.items() if p == "retida"]
    for d in datas_mosaico:
        if d not in por_data:
            print("  AVISO: %s nao e data de teste; ignorada." % d)
            continue
        cams = por_data[d]
        real = np.zeros(forma, dtype="float32")
        pred = np.zeros(forma, dtype=bool)
        pclim = np.zeros(forma, dtype=bool)
        valido = np.zeros(forma, dtype="float32")
        for cam, p in zip(cams, prever(cams)):
            y0, x0 = comum.posicao_do_nome(cam)
            _, y, v = comum.ler_patch(cam, [0])
            h, w = min(lado, forma[0] - y0), min(lado, forma[1] - x0)
            real[y0:y0 + h, x0:x0 + w] = y[0][:h, :w]
            valido[y0:y0 + h, x0:x0 + w] = v[0][:h, :w]
            pred[y0:y0 + h, x0:x0 + w] = (p >= limiar)[:h, :w]
            pclim[y0:y0 + h, x0:x0 + w] = (comum.recortar(freq, y0, x0, lado) >= lim_clim)[:h, :w]
        # recorta as bordas sem nenhum patch
        lin_ok, col_ok = np.where(valido.any(axis=1))[0], np.where(valido.any(axis=0))[0]
        r0, r1, c0, c1 = lin_ok.min(), lin_ok.max() + 1, col_ok.min(), col_ok.max() + 1
        cort = lambda a: a[r0:r1, c0:c1]
        lr = config.LADO_REGIAO_PX
        ret_local = [((ry * lr - r0) / lr, (rx * lr - c0) / lr) for ry, rx in retidas]
        im, tp, fp, fn = iou(cort(pred), cort(real), cort(valido))
        ic = iou(cort(pclim), cort(real), cort(valido))[0]
        titulo = "%s  -  chuva media %.1f mm" % (comum.data_br(d), chuva.get(d, float("nan")))
        cam_fig = os.path.join(destino, "mosaico_%s.png" % d)
        figura_mosaico(cam_fig, cort(real), cort(pred), cort(valido), ret_local, lr, titulo, im, ic, nome_modelo)
        print("  mosaico %s: IoU %.3f (climatologia %.3f) -> %s" % (comum.data_br(d), im, ic, cam_fig))
        registros.append({"tipo": "mosaico", "patch": d, "conjunto": "data inteira", "data": d,
                          "agua_real_px": int(((cort(real) > 0.5) & (cort(valido) > 0.5)).sum()),
                          "iou_modelo": im, "iou_climatologia": ic, "tp": tp, "fp": fp, "fn": fn})

    # ---- 2. paineis de patches ----
    for conjunto, n in PAINEIS.items():
        cams = escolher_patches(saida.get(conjunto, []), n)
        if not cams:
            continue
        linhas = []
        for cam, p in zip(cams, prever(cams)):
            _, y, v = comum.ler_patch(cam, [0])
            real, v = y[0], v[0]
            pm = p >= limiar
            pc = comum.recortar(freq, *comum.posicao_do_nome(cam), lado) >= lim_clim
            im, tp, fp, fn = iou(pm, real, v)
            ic = iou(pc, real, v)[0]
            nome = os.path.basename(cam)[:-4]
            d = comum.data_do_nome(cam)
            rot = "%s\n%s" % (comum.data_br(d), nome.split("_", 1)[1])
            linhas.append((rot, real, pm, pc, v, im, ic))
            figura_patches(os.path.join(destino, "patches", "%s.png" % nome), [(rot, real, pm, pc, v, im, ic)],
                           "%s - %s" % (conjunto.replace("_", " "), nome), nome_modelo)
            registros.append({"tipo": "patch", "patch": nome, "conjunto": conjunto, "data": d,
                              "agua_real_px": int(((real > 0.5) & (v > 0.5)).sum()),
                              "iou_modelo": im, "iou_climatologia": ic, "tp": tp, "fp": fp, "fn": fn})
        titulo = {"teste_tempo": "Datas novas, lugares ja vistos (teste tempo)",
                  "teste_espaco": "Lugares nunca vistos (teste espaco)"}.get(conjunto, conjunto)
        cam_fig = os.path.join(destino, "painel_%s.png" % conjunto)
        figura_patches(cam_fig, linhas, titulo, nome_modelo)
        print("  painel %s com %d patches -> %s" % (conjunto, len(linhas), cam_fig))

    with open(os.path.join(destino, "metricas_por_patch.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tipo", "patch", "conjunto", "data", "agua_real_px",
                                          "iou_modelo", "iou_climatologia", "tp", "fp", "fn"])
        w.writeheader()
        w.writerows(registros)
    print("\nFiguras em %s" % destino)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mapas de mancha real, prevista e erro.")
    ap.add_argument("--experimento", help="padrao: config.EXPERIMENTO_FINAL")
    ap.add_argument("--datas", nargs="+", help="datas de teste (AAAAMMDD) para os mosaicos")
    args = ap.parse_args()
    try:
        main(args.experimento, args.datas)
    except comum.PatchCorrompido as e:
        raise SystemExit("\nPatch corrompido: %s\nRode o verificar_patches.py." % e)
