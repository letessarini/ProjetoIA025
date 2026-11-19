"""
================================================================================
FIGURAS-RESUMO DA AVALIACAO NO MUNICIPIO
================================================================================

Le as tabelas do avaliacao_municipio.py e gera, na mesma pasta:

  resumo_escalas.png    IoU do modelo (barras) e da climatologia (losangos)
                        contando por patch e no mapa do municipio
  resumo_por_data.png   IoU no municipio inteiro e area inundada real x prevista
                        em cada data de teste

USO
  python figuras_resumo.py
  python figuras_resumo.py --experimento R10_r34_sem_urbano
================================================================================
"""

import argparse
import csv
import os

import numpy as np

import config

MODELO = "#2a78d6"      # modelo
CLIM = "#7d8790"        # climatologia (referencia)
REAL = "#3d3d3a"        # mancha real (contorno)
TXT, TXT2, GRADE = "#0b0b0b", "#52514e", "#e6e6e3"


def ler_csv(caminho):
    if not os.path.exists(caminho):
        raise SystemExit("Nao achei %s. Rode o avaliacao_municipio.py antes." % caminho)
    with open(caminho, encoding="utf-8-sig") as f:
        linhas = list(csv.DictReader(f))
    for r in linhas:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                pass
    return linhas


def br(x, nd=2):
    return ("%.*f" % (nd, x)).replace(".", ",")


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.edgecolor": "#b9b8b3", "axes.labelcolor": TXT2,
                         "xtick.color": TXT2, "ytick.color": TXT2})
    return matplotlib, plt


def figura_escalas(caminho, resumo, nome_modelo, n_datas, periodo):
    matplotlib, plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    def pega(zona, medida, col="iou"):
        r = next((r for r in resumo if r["zona"] == zona and r["medida"] == medida), None)
        return None if r is None or r.get(col) in ("", None) else float(r[col])

    tem_municipio = pega("municipio", "mapa") is not None
    z_vistas = "municipio_regioes_treino" if tem_municipio else "regioes_treino"
    z_todo = "municipio" if tem_municipio else "dominio"
    n_patches = pega("regioes_treino", "media_por_patch", "n_patches")
    linhas = [
        ("Média simples por patch", "lugares já vistos · %d patches" % (n_patches or 0),
         pega("regioes_treino", "media_por_patch"), None),
        ("Soma dos patches", "lugares já vistos (como no treino)", pega("regioes_treino", "soma_dos_patches"), None),
        ("Partes já vistas", "regiões usadas no treino", pega(z_vistas, "mapa"), pega(z_vistas, "climatologia_mapa")),
        ("Município inteiro" if tem_municipio else "Domínio inteiro", "tudo dentro do limite",
         pega(z_todo, "mapa"), pega(z_todo, "climatologia_mapa")),
        ("Regiões nunca vistas", "lugar e data novos", pega("regioes_retidas", "mapa"),
         pega("regioes_retidas", "climatologia_mapa")),
    ]
    ys = np.array([5.6, 4.6, 2.6, 1.6, 0.6])
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.barh(ys, [l[2] or 0 for l in linhas], height=0.56, color=MODELO, zorder=2)
    for y, (t1, t2, m, c) in zip(ys, linhas):
        m = m or 0.0
        if c is not None:
            ax.plot([c], [y], marker="D", ms=10, color=CLIM, mec="white", mew=1.8, zorder=3, ls="none")
            if c > 0.02:
                ax.text(c, y + 0.42, "clim. " + br(c), va="center", ha="center", fontsize=10, color=TXT2)
        ax.text(max(m, c or 0) + 0.012, y, br(m, 3 if m < 0.05 else 2), va="center", ha="left",
                fontsize=13, color=TXT, fontweight="bold")
        ax.text(-0.015, y + 0.12, t1, ha="right", va="center", fontsize=12.5, color=TXT,
                transform=ax.get_yaxis_transform())
        ax.text(-0.015, y - 0.2, t2, ha="right", va="center", fontsize=10, color=TXT2,
                transform=ax.get_yaxis_transform())
    titulo_mapa = "MAPA DE %s (reconstruído)" % config.MUNICIPIO["nome"].upper() if tem_municipio \
        else "MAPA DO DOMÍNIO (reconstruído)"
    for y, rot in ((6.35, "PATCHES 256 × 256"), (3.35, titulo_mapa)):
        ax.text(-0.015, y, rot, ha="right", va="center", fontsize=10.5, color=TXT2, fontweight="bold",
                transform=ax.get_yaxis_transform())
    ax.axhline(3.8, color="#cfcfca", lw=1, ls=(0, (2, 3)), xmin=-0.45, clip_on=False)
    topo = max([l[2] or 0 for l in linhas] + [l[3] or 0 for l in linhas])
    ax.set_ylim(0.0, 6.7); ax.set_xlim(0, max(0.2, topo * 1.2)); ax.set_yticks([])
    ax.set_xlabel("IoU (soma das %d datas de teste, %s)" % (n_datas, periodo))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: br(x, 1)))
    ax.grid(axis="x", color=GRADE, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(handles=[Patch(color=MODELO, label=nome_modelo),
                       Line2D([], [], marker="D", ms=9, color=CLIM, mec="white", ls="none", label="Climatologia")],
              loc="lower right", frameon=False, fontsize=11)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.96, bottom=0.11)
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def figura_por_data(caminho, avaliacao, nome_modelo, zona):
    matplotlib, plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    def serie(medida, col):
        return {str(int(r["data"])): float(r[col]) for r in avaliacao if r["zona"] == zona and r["medida"] == medida}

    iou_m, iou_c = serie("mapa", "iou"), serie("climatologia_mapa", "iou")
    real, prev_m = serie("mapa", "area_real_km2"), serie("mapa", "area_prevista_km2")
    prev_c = serie("climatologia_mapa", "area_prevista_km2")
    datas = sorted(real)
    rot = ["%s/%s\n%s" % (d[6:8], d[4:6], d[:4]) for d in datas]
    seco = np.array([real[d] == 0 for d in datas])
    x = np.arange(len(datas))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14.5, 5.4), gridspec_kw={"wspace": 0.18})
    im = np.where(seco, np.nan, [iou_m[d] for d in datas])
    ic = np.where(seco, np.nan, [iou_c[d] for d in datas])
    a1.bar(x, im, width=0.6, color=MODELO, zorder=2)
    a1.plot(x, ic, marker="D", ms=9, color=CLIM, mec="white", mew=1.6, ls="none", zorder=3)
    for i in range(len(x)):
        if seco[i]:
            a1.text(x[i], 0.04, "sem cheia\n(data seca)", ha="center", va="bottom", fontsize=10, color=TXT2)
        else:
            a1.text(x[i], 0.025, br(im[i]), ha="center", va="bottom", fontsize=11, color="white", fontweight="bold")
    a1.set_xticks(x); a1.set_xticklabels(rot, fontsize=10.5)
    a1.set_ylim(0, max(0.5, np.nanmax(np.r_[im, ic]) * 1.15)); a1.set_xlim(-0.6, len(x) - 0.3)
    a1.set_ylabel({"municipio": "IoU no município inteiro"}.get(zona, "IoU no domínio"))
    a1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: br(y, 1)))
    a1.set_title("Acerto da mancha em cada data", fontsize=13.5, color=TXT, loc="left")
    a1.legend(handles=[Patch(color=MODELO, label=nome_modelo),
                       Line2D([], [], marker="D", ms=8, color=CLIM, mec="white", ls="none", label="Climatologia")],
              loc="upper right", frameon=False, fontsize=10.5, ncol=2)

    w = 0.36
    vr, vm = np.array([real[d] for d in datas]), np.array([prev_m[d] for d in datas])
    a2.bar(x - w / 2, vr, width=w, facecolor="white", edgecolor=REAL, lw=1.4, zorder=2)
    a2.bar(x + w / 2, vm, width=w, color=MODELO, zorder=2)
    clim_km2 = float(np.median([prev_c[d] for d in datas]))
    a2.axhline(clim_km2, color=CLIM, lw=1.8, ls=(0, (5, 3)), zorder=1)
    topo = max(vr.max(), vm.max(), clim_km2)
    for i in range(len(x)):
        a2.text(x[i] - w / 2, vr[i] + topo * 0.015, br(vr[i], 1), ha="center", va="bottom", fontsize=9.5, color=TXT2)
        a2.text(x[i] + w / 2, vm[i] + topo * 0.015, br(vm[i], 1), ha="center", va="bottom", fontsize=9.5, color=TXT)
    a2.set_xticks(x); a2.set_xticklabels(rot, fontsize=10.5)
    a2.set_ylim(0, topo * 1.2); a2.set_ylabel("Área inundada (km²)")
    a2.set_title("Área inundada: real × prevista", fontsize=13.5, color=TXT, loc="left")
    a2.legend(handles=[Patch(facecolor="white", edgecolor=REAL, label="Real (SFINCS)"),
                       Patch(color=MODELO, label=nome_modelo),
                       Line2D([], [], color=CLIM, lw=1.8, ls=(0, (5, 3)),
                              label="Climatologia (%s km² em toda data)" % br(clim_km2, 1))],
              loc="upper left", frameon=False, fontsize=10.5, ncol=3, columnspacing=1.2, handlelength=1.6)
    for ax in (a1, a2):
        ax.grid(axis="y", color=GRADE, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, bottom=0.14)
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def main(experimento=None):
    experimento = experimento or config.EXPERIMENTO_FINAL
    pasta = os.path.join(config.RESULTADOS, "municipio_%s" % experimento)
    resumo = ler_csv(os.path.join(pasta, "resumo_municipio.csv"))
    avaliacao = ler_csv(os.path.join(pasta, "avaliacao_municipio.csv"))
    nome_modelo = experimento.split("_")[0]
    datas = sorted({str(int(r["data"])) for r in avaliacao})
    periodo = "%s–%s" % (datas[0][:4], datas[-1][:4])
    zona = "municipio" if any(r["zona"] == "municipio" for r in avaliacao) else "dominio"
    figura_escalas(os.path.join(pasta, "resumo_escalas.png"), resumo, nome_modelo, len(datas), periodo)
    figura_por_data(os.path.join(pasta, "resumo_por_data.png"), avaliacao, nome_modelo, zona)
    print("Figuras em %s" % pasta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Figuras-resumo da avaliacao no municipio.")
    ap.add_argument("--experimento", help="padrao: config.EXPERIMENTO_FINAL")
    main(ap.parse_args().experimento)
