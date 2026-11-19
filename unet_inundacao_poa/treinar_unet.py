"""
================================================================================
TREINO E AVALIACAO DAS REDES DE SEGMENTACAO DA MANCHA DE INUNDACAO
================================================================================

Treina os experimentos definidos em config.EXPERIMENTOS e avalia cada um nos
quatro conjuntos de avaliacao, junto com dois baselines sem rede ("tudo seco"
e climatologia).

  Split sem vazamento: as DATAS sao divididas em treino/validacao/teste
  (estratificadas pela chuva) e a grade e dividida em REGIOES de 512 x 512 px,
  das quais 25% nunca sao vistas no treino. O cruzamento da:
    treino        datas de treino   x regioes de treino
    validacao     datas de val.     x regioes de treino
    teste_tempo   datas de teste    x regioes de treino  (datas novas)
    teste_espaco  datas de treino   x regioes retidas    (lugares novos)
    teste_ambos   datas de teste    x regioes retidas    (datas e lugares novos)

  Treino: AdamW + cosseno, precisao mista na GPU, parada antecipada pela IoU de
  validacao. O limiar de decisao e escolhido na validacao (passo 0,001) e usado
  sem mudanca nos testes. IC 95% por bootstrap de patches e, nas comparacoes
  pareadas, tambem por bootstrap de datas.

SAIDA (em config.RESULTADOS)
  resultados.csv             uma linha por (experimento, conjunto), com os baselines
  comparacoes_pareadas.csv   diferencas de IoU com IC 95% pareado
  replicas.csv               media e desvio-padrao entre sementes
  <experimento>.pth          pesos do melhor modelo (so existe se o treino terminou)
  <experimento>_curvas.png   perda e IoU de validacao por epoca
  comparacao.png             IoU de todos os experimentos
  split.json                 papel de cada data e de cada regiao

USO
  python treinar_unet.py --modo teste_rapido     # 2 epocas curtas, confere o ambiente
  python treinar_unet.py                         # todos os experimentos de config.RODAR
  python treinar_unet.py --experimentos R6_unet_resnet34 R6_s2
  python treinar_unet.py --modo baselines        # so a climatologia e o "tudo seco"

  Experimentos ja treinados (com .pth) sao apenas recarregados e reavaliados.
  Um treino interrompido recomeca do zero.
================================================================================
"""

import argparse
import csv
import json
import os
import time

import numpy as np

import comum
import config


# =============================================================================
# 1. TREINO
# =============================================================================

def criar_scaler(ativo):
    """GradScaler para precisao mista."""
    import torch
    try:
        return torch.amp.GradScaler("cuda", enabled=ativo)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=ativo)


def treinar(nome, cfg, saida, idx, destino, rapido):
    """Treina (ou carrega, se ja terminou) e devolve (rede, dev, checkpoint)."""
    import torch

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["semente"])
    rede = comum.construir_rede(cfg["arquitetura"], len(idx)).to(dev)
    cam_final = os.path.join(destino, "%s.pth" % nome)
    cam_parcial = os.path.join(destino, "%s_parcial.pth" % nome)

    if os.path.exists(cam_final):
        ck = comum.carregar_pesos(cam_final, dev)
        rede.load_state_dict(ck["modelo"])
        rede.eval()
        print("  ja treinado (melhor epoca %d de %d). Carregando %s"
              % (ck["epoca"], ck["epocas_rodadas"], os.path.basename(cam_final)))
        return rede, dev, ck
    if os.path.exists(cam_parcial):
        print("  AVISO: treino interrompido encontrado. Recomecando do zero.")

    n_par = sum(p.numel() for p in rede.parameters())
    print("  %s, %.2f M parametros, %d canais, perda %s, aumento %s, lote %d, semente %d, %s"
          % (cfg["arquitetura"], n_par / 1e6, len(idx), cfg["perda"], cfg["aumento"], cfg["lote"],
             cfg["semente"], dev))

    epocas = 2 if rapido else cfg["epocas"]
    lotes_por_epoca = 3 if rapido else None
    tr = comum.LoteDePatches(saida["treino"], idx, treino=True, aumento=cfg["aumento"],
                             frac_molhado=cfg["frac_molhado"], semente=cfg["semente"])
    va = comum.LoteDePatches(saida["validacao"], idx, treino=False)
    opt = torch.optim.AdamW(rede.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epocas)
    amp = dev.type == "cuda"
    scaler = criar_scaler(amp)

    hist = {"perda_treino": [], "perda_val": [], "iou_val": [], "limiar_val": []}
    melhor, melhor_ep, espera = -1.0, 0, 0
    t_inicio = time.time()
    for ep in range(epocas):
        t0 = time.time()
        rede.train()
        soma, n = 0.0, 0
        for X, y, v in tr.lotes(cfg["lote"], por_epoca=lotes_por_epoca):
            X, y, v = (torch.from_numpy(a).to(dev) for a in (X, y, v))
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                logits = rede(X)
            perda = comum.calcular_perda(logits.float(), y, v, cfg)    # perda em float32
            if not torch.isfinite(perda):
                continue
            scaler.scale(perda).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            soma += float(perda.detach()) * X.shape[0]
            n += X.shape[0]
        sched.step()

        # validacao: perda e IoU no melhor limiar
        rede.eval()
        hp, hn = comum.novo_histograma()
        soma_v, n_v = 0.0, 0
        with torch.no_grad():
            for X, y, v in va.lotes(cfg["lote"]):
                X, y, v = (torch.from_numpy(a).to(dev) for a in (X, y, v))
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                    logits = rede(X)
                logits = logits.float()
                soma_v += float(comum.calcular_perda(logits, y, v, cfg)) * X.shape[0]
                n_v += X.shape[0]
                comum.somar_histograma(hp, hn, torch.sigmoid(logits).cpu().numpy(),
                                       y.cpu().numpy(), v.cpu().numpy())
        limiar, iou_val = comum.melhor_limiar_histograma(hp, hn)
        for chave, valor in (("perda_treino", soma / max(1, n)), ("perda_val", soma_v / max(1, n_v)),
                             ("iou_val", iou_val), ("limiar_val", limiar)):
            hist[chave].append(float(valor))

        marca = ""
        if iou_val > melhor + 1e-4:
            melhor, melhor_ep, espera = iou_val, ep + 1, 0
            torch.save({"modelo": rede.state_dict(), "epoca": ep + 1, "iou_val": float(iou_val)}, cam_parcial)
            marca = "  <- melhor"
        else:
            espera += 1
        print("    ep %3d/%d  perda %.4f / val %.4f  IoU val %.3f (limiar %.3f)  %4.0fs%s"
              % (ep + 1, epocas, hist["perda_treino"][-1], hist["perda_val"][-1], iou_val, limiar,
                 time.time() - t0, marca))
        if espera >= config.PACIENCIA:
            print("    parada antecipada na epoca %d (melhor: %d)" % (ep + 1, melhor_ep))
            break

    # o arquivo final so passa a existir quando o treino termina
    ck = comum.carregar_pesos(cam_parcial, dev)
    ck.update({"historico": hist, "epocas_rodadas": len(hist["iou_val"]),
               "minutos": (time.time() - t_inicio) / 60.0, "config": dict(cfg, tirar=list(cfg["tirar"]))})
    torch.save(ck, cam_final)
    os.remove(cam_parcial)
    rede.load_state_dict(ck["modelo"])
    rede.eval()
    plotar_curvas(hist, ck["epoca"], os.path.join(destino, "%s_curvas.png" % nome), nome)
    return rede, dev, ck


def avaliar(probs, alvos, validos, limiar, linha_base):
    """Metricas de um conjunto no limiar escolhido na validacao."""
    cs = [comum.contar(p >= limiar, a, v) for p, a, v in zip(probs, alvos, validos)]
    tp, fp, fn, tn = (int(x) for x in np.sum(cs, axis=0))
    m = comum.metricas(tp, fp, fn, tn)
    lo, hi = comum.ic_iou(cs)
    m.update(linha_base)
    m.update({"limiar": limiar, "iou_ic95_lo": lo, "iou_ic95_hi": hi, "n_patches": len(probs)})
    m["_contagens"] = cs        # (tp, fp, fn, tn) por patch, para as comparacoes pareadas
    print("      %-13s IoU %.3f [%.3f-%.3f]  F1 %.3f  prec %.3f  rec %.3f  agua prevista %.1f%% (real %.1f%%)"
          % (m["conjunto"], m["iou"], lo, hi, m["f1"], m["precisao"], m["recall"],
             100 * m["frac_prevista"], 100 * m["prevalencia"]))
    return m


# =============================================================================
# 2. GRAFICOS E TABELAS
# =============================================================================

def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def plotar_curvas(hist, melhor_ep, caminho, nome):
    plt = _plt()
    if plt is None:
        return
    ep = range(1, len(hist["iou_val"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(ep, hist["perda_treino"], label="treino")
    ax[0].plot(ep, hist["perda_val"], label="validacao")
    ax[0].set_xlabel("epoca"); ax[0].set_ylabel("perda"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(ep, hist["iou_val"], color="#1B6F8F")
    ax[1].set_xlabel("epoca"); ax[1].set_ylabel("IoU na validacao (melhor limiar)"); ax[1].grid(alpha=0.3)
    for a in ax:
        a.axvline(melhor_ep, color="#D98324", ls="--", alpha=0.7)
    fig.suptitle("%s (melhor: epoca %d)" % (nome, melhor_ep))
    fig.tight_layout(); fig.savefig(caminho, dpi=150); plt.close(fig)


def plotar_comparacao(linhas, destino):
    plt = _plt()
    if plt is None:
        return
    exps = [e for e in dict.fromkeys(r["experimento"] for r in linhas) if not e.startswith("0_")]
    if not exps:
        return
    iou = {(r["experimento"], r["conjunto"]): r["iou"] for r in linhas}
    clim = iou.get(("0_climatologia", "teste_tempo"), 0.0)
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(exps) + 1.8))
    ys = np.arange(len(exps))
    ax.barh(ys - 0.2, [iou.get((e, "teste_tempo"), 0) for e in exps], 0.4, color="#1B6F8F",
            label="teste_tempo (datas novas)")
    ax.barh(ys + 0.2, [iou.get((e, "teste_espaco"), 0) for e in exps], 0.4, color="#D98324",
            label="teste_espaco (regioes novas)")
    ax.axvline(clim, color="black", ls="--", lw=1, label="climatologia no teste_tempo (%.3f)" % clim)
    ax.set_yticks(ys); ax.set_yticklabels(exps); ax.invert_yaxis()
    ax.set_xlabel("IoU (pixels validos)"); ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.3 / len(exps)), ncol=3, frameon=False)
    fig.savefig(os.path.join(destino, "comparacao.png"), dpi=150, bbox_inches="tight"); plt.close(fig)


def gravar(linhas, destino):
    campos = ["experimento", "conjunto", "limiar", "iou", "iou_ic95_lo", "iou_ic95_hi", "f1",
              "precisao", "recall", "frac_prevista", "prevalencia", "n_patches", "tp", "fp", "fn", "tn",
              "arquitetura", "perda", "aumento", "canais", "semente", "epoca_melhor", "epocas_rodadas", "minutos",
              "limiar_na_borda"]
    with open(os.path.join(destino, "resultados.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        w.writerows(linhas)


def _pega(linhas, e, c):
    return next((r for r in linhas if r["experimento"] == e and r["conjunto"] == c), None)


def _comp(comps, e, ref, c):
    return next((r for r in comps if r["experimento"] == e and r["referencia"] == ref and r["conjunto"] == c), None)


def comparacoes_pareadas(linhas, destino):
    """Diferencas de IoU (IC 95% pareado) contra a climatologia e contra config.REFERENCIA."""
    exps = [e for e in dict.fromkeys(r["experimento"] for r in linhas) if not e.startswith("0_")]
    saida = []
    for e in exps:
        for ref, conjs in (("0_climatologia", ("teste_tempo",)), (config.REFERENCIA, config.CONJUNTOS_AVALIACAO)):
            if e == ref:
                continue
            for c in conjs:
                a, b = _pega(linhas, e, c), _pega(linhas, ref, c)
                if not (a and b):
                    continue
                d, lo, hi = comum.ic_diferenca(a["_contagens"], b["_contagens"])
                _, lo_d, hi_d = comum.ic_diferenca_por_data(a["_contagens"], b["_contagens"], a.get("_datas", []))
                saida.append({"experimento": e, "referencia": ref, "conjunto": c, "iou": a["iou"],
                              "iou_referencia": b["iou"], "diferenca": d, "dif_ic95_lo": lo, "dif_ic95_hi": hi,
                              "veredito": "melhor" if lo > 0 else ("pior" if hi < 0 else "empate"),
                              "dif_ic95_lo_data": lo_d, "dif_ic95_hi_data": hi_d,
                              "veredito_data": "melhor" if lo_d > 0 else ("pior" if hi_d < 0 else "empate")})
    with open(os.path.join(destino, "comparacoes_pareadas.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["experimento", "referencia", "conjunto", "iou", "iou_referencia",
                                          "diferenca", "dif_ic95_lo", "dif_ic95_hi", "veredito",
                                          "dif_ic95_lo_data", "dif_ic95_hi_data", "veredito_data"])
        w.writeheader()
        w.writerows(saida)
    return saida


def resumo_replicas(linhas, comps, destino):
    """Media e desvio-padrao de cada grupo de config.REPLICAS."""
    saida = []
    for grupo, membros in config.REPLICAS.items():
        presentes = [m for m in membros if _pega(linhas, m, "teste_tempo") and _pega(linhas, m, "teste_espaco")]
        if len(presentes) < 2:
            continue
        tempo = np.array([_pega(linhas, m, "teste_tempo")["iou"] for m in presentes])
        espaco = np.array([_pega(linhas, m, "teste_espaco")["iou"] for m in presentes])
        cs = [_comp(comps, m, "0_climatologia", "teste_tempo") for m in presentes]
        saida.append({"grupo": grupo, "n_sementes": len(presentes),
                      "tempo_media": float(tempo.mean()), "tempo_dp": float(tempo.std(ddof=1)),
                      "tempo_min": float(tempo.min()), "tempo_max": float(tempo.max()),
                      "espaco_media": float(espaco.mean()), "espaco_dp": float(espaco.std(ddof=1)),
                      "superam_por_patch": sum(1 for r in cs if r and r["veredito"] == "melhor"),
                      "superam_por_data": sum(1 for r in cs if r and r["veredito_data"] == "melhor")})
    if not saida:
        return
    print("\n3) Repeticoes com sementes diferentes (media +- desvio-padrao)")
    print("%-28s %-4s %-26s %-18s %s" % ("grupo", "n", "tempo: media +- dp (min-max)", "espaco: media",
                                         "superam a climatologia"))
    for r in saida:
        print("%-28s %-4d %.3f +- %.3f (%.3f-%.3f)   %.3f +- %.3f    %d/%d por patch, %d/%d por data"
              % (r["grupo"], r["n_sementes"], r["tempo_media"], r["tempo_dp"], r["tempo_min"], r["tempo_max"],
                 r["espaco_media"], r["espaco_dp"], r["superam_por_patch"], r["n_sementes"],
                 r["superam_por_data"], r["n_sementes"]))
    with open(os.path.join(destino, "replicas.csv"), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(saida[0].keys()))
        w.writeheader()
        w.writerows(saida)


def relatorio(linhas, comps, destino):
    """Resumo impresso no fim (tudo tambem fica nos .csv)."""
    exps = [e for e in dict.fromkeys(r["experimento"] for r in linhas) if not e.startswith("0_")]
    clim = _pega(linhas, "0_climatologia", "teste_tempo")
    print("\n" + "=" * 86 + "\nRESUMO\n" + "=" * 86)
    if clim:
        print("Climatologia no teste_tempo: IoU %.3f [%.3f-%.3f]\n"
              % (clim["iou"], clim["iou_ic95_lo"], clim["iou_ic95_hi"]))
    nomes_v = {"melhor": "SUPERA", "pior": "PERDE", "empate": "empata"}
    print("1) IoU no teste_tempo e diferenca para a climatologia (IC 95% pareado)")
    print("   'por patch' sorteia patches; 'por data' sorteia datas inteiras")
    print("%-22s %-6s %-34s %-26s %s" % ("experimento", "tempo", "dif. (IC por patch)", "IC por data", "espaco"))
    for e in exps:
        t, s = _pega(linhas, e, "teste_tempo"), _pega(linhas, e, "teste_espaco")
        if not t or not s:
            continue
        r = _comp(comps, e, "0_climatologia", "teste_tempo")
        txt = ("%+.3f [%+.3f; %+.3f] %s" % (r["diferenca"], r["dif_ic95_lo"], r["dif_ic95_hi"], nomes_v[r["veredito"]])
               if r else "-")
        txt_d = ("[%+.3f; %+.3f] %s" % (r["dif_ic95_lo_data"], r["dif_ic95_hi_data"], nomes_v[r["veredito_data"]])
                 if r else "-")
        print("%-22s %.3f  %-34s %-26s %.3f" % (e, t["iou"], txt, txt_d, s["iou"]))

    if any(r["referencia"] == config.REFERENCIA for r in comps):
        print("\n2) Diferenca de IoU para %s (IC 95%% pareado; '*' = IC nao cruza o zero)" % config.REFERENCIA)
        print("%-22s %-30s %s" % ("experimento", "teste_tempo", "teste_espaco"))
        for e in exps:
            if e == config.REFERENCIA:
                continue
            partes = []
            for c in ("teste_tempo", "teste_espaco"):
                r = _comp(comps, e, config.REFERENCIA, c)
                partes.append("%+.3f [%+.3f; %+.3f]%s" % (r["diferenca"], r["dif_ic95_lo"], r["dif_ic95_hi"],
                                                         "*" if r["veredito"] != "empate" else " ") if r else "-")
            print("%-22s %-30s %s" % (e, partes[0], partes[1]))
    resumo_replicas(linhas, comps, destino)


# =============================================================================
# 3. PROGRAMA PRINCIPAL
# =============================================================================

def main(modo="completo", experimentos=None):
    rapido = modo == "teste_rapido"
    experimentos = list(experimentos or config.RODAR)
    for nome in experimentos:
        comum.receita(nome)                 # para logo se algum nome nao existir
    if modo != "baselines":
        comum.conferir_torch()
    destino = os.path.join(config.RESULTADOS, "teste_rapido") if rapido else config.RESULTADOS
    os.makedirs(destino, exist_ok=True)

    man, itens = comum.carregar_itens()
    canais = list(man["canais"])
    print("%d patches, %d datas, canais: %s" % (len(itens), len({i[1] for i in itens}), ", ".join(canais)))

    saida, papel_d, papel_r = comum.montar_split(itens)
    problemas = comum.conferir_vazamento(saida, itens)
    if problemas:
        raise SystemExit("VAZAMENTO no split: " + "; ".join(problemas))
    print("\nSplit (sem vazamento):")
    for c in ("treino", "validacao", "teste_tempo", "teste_espaco", "teste_ambos"):
        print("  %-13s %5d patches" % (c, len(saida.get(c, []))))
    retidas = sorted(str(r) for r, p in papel_r.items() if p == "retida")
    print("  regioes retidas: %s de %d" % (", ".join(retidas), len(papel_r)))
    split = comum.split_em_dict(saida, papel_d, papel_r)
    if config.SPLIT_REFERENCIA:
        if os.path.exists(config.SPLIT_REFERENCIA):
            comum.comparar_split(config.SPLIT_REFERENCIA, split)
        else:
            print("  AVISO: SPLIT_REFERENCIA nao encontrado: %s" % config.SPLIT_REFERENCIA)
    with open(os.path.join(destino, "split.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, ensure_ascii=False, indent=2)

    if rapido:
        saida = {c: (v[:12] if c != "treino" else v) for c, v in saida.items()}
        print("\nTESTE RAPIDO: cada conjunto de avaliacao foi cortado em 12 patches;")
        print("os numeros so mostram que o codigo funciona.")
    conjuntos = [c for c in config.CONJUNTOS_AVALIACAO if saida.get(c)]
    lado = int(man["patch"]["tamanho"])
    linhas = []

    # ---- baselines sem rede ----
    print("\n" + "=" * 78 + "\nBASELINES SEM REDE\n" + "=" * 78)
    freq = comum.climatologia_na_grade(saida["treino"], tuple(man["grade"]["forma"]))
    dados_conj = {}
    datas_conj = {c: [comum.data_do_nome(cam) for cam in saida[c]] for c in conjuntos}
    for c in conjuntos:
        dados_conj[c] = comum.alvos_e_validos(saida[c])
    baselines = {
        "0_tudo_seco": lambda c: [np.zeros((lado, lado), dtype="float32") for _ in saida[c]],
        "0_climatologia": lambda c: [comum.recortar(freq, *comum.posicao_do_nome(cam), lado) for cam in saida[c]],
    }
    for nome, prever_base in baselines.items():
        limiar = comum.melhor_limiar(prever_base("validacao"), *dados_conj["validacao"])
        print("  %s (limiar %.3f)" % (nome, limiar))
        for c in conjuntos:
            linhas.append(avaliar(prever_base(c), *dados_conj[c], limiar,
                                  {"experimento": nome, "conjunto": c, "arquitetura": "-", "canais": "-",
                                   "_datas": datas_conj[c]}))
    gravar(linhas, destino)
    if modo == "baselines":
        print("\nResultados em %s" % destino)
        return

    # ---- experimentos ----
    for nome in experimentos:
        cfg = comum.receita(nome)
        usar, idx = comum.canais_do_experimento(canais, cfg)
        print("\n" + "=" * 78 + "\n%s  (canais: %s)\n" % (nome, ", ".join(usar)) + "=" * 78)
        rede, dev, ck = treinar(nome, cfg, saida, idx, destino, rapido)

        probs, alvos, validos = comum.prever(rede, dev, saida["validacao"], idx, cfg["lote"])
        hp, hn = comum.novo_histograma()
        for p, a, v in zip(probs, alvos, validos):
            comum.somar_histograma(hp, hn, p, a, v)
        limiar, iou_val = comum.melhor_limiar_histograma(hp, hn)
        borda = limiar <= 0.01 or limiar >= 0.99
        print("  limiar escolhido na validacao: %.3f (IoU %.3f)%s"
              % (limiar, iou_val, "  <- AVISO: limiar na borda (modelo mal calibrado)" if borda else ""))
        base = {"experimento": nome, "arquitetura": cfg["arquitetura"], "semente": cfg["semente"],
                "perda": cfg["perda"] if cfg["perda"] == "bce_dice" else "tversky_%.1f_%.1f" % (cfg["alfa"], cfg["beta"]),
                "aumento": cfg["aumento"], "canais": " ".join(usar), "epoca_melhor": ck["epoca"],
                "epocas_rodadas": ck.get("epocas_rodadas", ""), "minutos": round(ck.get("minutos", 0.0), 1),
                "limiar_na_borda": borda}
        for c in conjuntos:
            if c != "validacao":
                probs, alvos, validos = comum.prever(rede, dev, saida[c], idx, cfg["lote"])
            linhas.append(avaliar(probs, alvos, validos, limiar, dict(base, conjunto=c, _datas=datas_conj[c])))
        gravar(linhas, destino)            # grava a cada experimento

        del rede
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    plotar_comparacao(linhas, destino)
    comps = comparacoes_pareadas(linhas, destino)
    relatorio(linhas, comps, destino)
    print("\nResultados em %s" % destino)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Treina e avalia os experimentos de config.py.")
    ap.add_argument("--modo", choices=("completo", "teste_rapido", "baselines"), default="completo")
    ap.add_argument("--experimentos", nargs="+", help="nomes de config.EXPERIMENTOS (padrao: config.RODAR)")
    args = ap.parse_args()
    try:
        main(args.modo, args.experimentos)
    except comum.PatchCorrompido as e:
        raise SystemExit("\nPatch corrompido (falhou em 3 tentativas): %s\n"
                         "Rode o verificar_patches.py. Experimentos ja terminados continuam salvos." % e)
