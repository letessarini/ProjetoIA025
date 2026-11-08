"""
================================================================================
nlp_34 -- TREINAR A U-NET (Com Checkpoint / Resume automático)
================================================================================
"""

import os
import csv
import json
import glob
import time
import numpy as np

MODO = "completo"          # "teste" | "baselines" | "completo"

# Caminhos relativos automáticos 
RAIZ = os.path.dirname(os.path.abspath(__file__))
DADOS = os.path.join(RAIZ, "..", "DATASET_UNET")
DESTINO = os.path.join(RAIZ, "..", "resultados_exp1_unet")
os.makedirs(DESTINO, exist_ok=True)

SEMENTE = 20261022

# --- split ---
FRAC_DATAS = (0.70, 0.15, 0.15)     # treino, validacao, teste
LADO_REGIAO_PX = 512                # 512 px de 20 m = 10,2 km
FRAC_REGIOES_RETIDAS = 0.25

# --- treino ---
ARQUITETURA = "unet_pequena"        # U-Net nativa em PyTorch
CANAIS_BASE = 32
EPOCAS = 80                         # fiz o treinamento com 80 épocas para agilizar pois estava demorando muito, obtendo o melhor resultado antes das 80
PACIENCIA = 15
LOTE = 16
LR = 3e-4
WEIGHT_DECAY = 1e-5
FRAC_MOLHADO_NO_LOTE = 0.7          # sobreamostragem de patch com agua
TVERSKY_ALFA, TVERSKY_BETA = 0.3, 0.7   # beta > alfa pune falso negativo
WORKERS = 2

# --- avaliacao ---
LIMIARES = tuple(np.round(np.arange(0.01, 0.991, 0.02), 3))
N_BOOTSTRAP = 300

# Grupos de canais para a ablação
GRUPOS = {
    "chuva": ("precip_total_mm", "precip_max_mmh"),
    "urbano": ("lulc_idx", "imperv", "susc_via"),
    "relevo": ("decl_pct",),
    "noticias": ("noticias_nlp",),
}

# Proporções de patches secos para o Experimento 1
PROPORCOES_SECOS = [0.0, 0.25, 0.50, 0.75, 1.0]


# =============================================================================
# 1. SPLIT E FILTRAGEM DO EXPERIMENTO 1
# =============================================================================

def regiao_do_patch(y0, x0, lado=LADO_REGIAO_PX):
    return (int(y0) // int(lado), int(x0) // int(lado))


def dividir_datas(datas, chuva, frac=FRAC_DATAS, semente=SEMENTE):
    ordem = sorted(range(len(datas)), key=lambda i: (-float(chuva[i]), str(datas[i])))
    n = len(ordem)
    alvo = [int(round(f * n)) for f in frac]
    alvo[0] = n - alvo[1] - alvo[2]
    rng = np.random.RandomState(semente)
    sequencia = (["treino"] * alvo[0] + ["val"] * alvo[1] + ["teste"] * alvo[2])
    rng.shuffle(sequencia)
    out = {}
    for pos, i in enumerate(ordem):
        out[str(datas[i])] = sequencia[pos]
    return out


def dividir_regioes(regioes, frac_retida=FRAC_REGIOES_RETIDAS, semente=SEMENTE):
    regs = sorted(set(regioes))
    rng = np.random.RandomState(semente + 1)
    n_ret = max(1, int(round(frac_retida * len(regs))))
    idx = rng.permutation(len(regs))[:n_ret]
    retidas = {regs[i] for i in idx}
    return {r: ("retida" if r in retidas else "treino") for r in regs}


def conjunto_do_patch(papel_data, papel_regiao):
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


def montar_split(itens, lado=LADO_REGIAO_PX, frac=FRAC_DATAS,
                 frac_reg=FRAC_REGIOES_RETIDAS, semente=SEMENTE):
    datas, chuva = [], []
    vistos = {}
    for _, d, _, _, mm in itens:
        if d not in vistos:
            vistos[d] = float(mm)
    datas = sorted(vistos)
    chuva = [vistos[d] for d in datas]
    papel_d = dividir_datas(datas, chuva, frac, semente)
    regs = [regiao_do_patch(y0, x0, lado) for _, _, y0, x0, _ in itens]
    papel_r = dividir_regioes(regs, frac_reg, semente)

    saida = {}
    for (cam, d, y0, x0, _), r in zip(itens, regs):
        c = conjunto_do_patch(papel_d[d], papel_r[r])
        if c is None:
            continue
        saida.setdefault(c, []).append(cam)
    return saida, papel_d, papel_r


def filtrar_treino_por_balanceamento(arquivos_treino, frac_secos, semente=SEMENTE):
    molhados = []
    secos = []
    for cam in arquivos_treino:
        with np.load(cam, allow_pickle=False) as d:
            y = np.asarray(d["y"])
            v = np.asarray(d["valido"]) if "valido" in d else np.ones_like(y)
            if int(((y == 1) & (v == 1)).sum()) > 0:
                molhados.append(cam)
            else:
                secos.append(cam)

    if frac_secos >= 1.0:
        return molhados + secos
    if frac_secos <= 0.0:
        return molhados

    rng = np.random.RandomState(semente)
    n_secos_alvo = int(round(len(secos) * frac_secos))
    secos_escolhidos = list(rng.choice(secos, size=n_secos_alvo, replace=False)) if n_secos_alvo > 0 else []
    return molhados + secos_escolhidos


# =============================================================================
# 2. METRICAS MASCARADAS
# =============================================================================

def contar(pred, alvo, valido=None):
    p = np.asarray(pred).astype(bool).ravel()
    a = np.asarray(alvo).astype(bool).ravel()
    m = np.ones_like(p, dtype=bool) if valido is None else np.asarray(valido).astype(bool).ravel()
    p, a = p[m], a[m]
    return (int(np.sum(p & a)), int(np.sum(p & ~a)),
            int(np.sum(~p & a)), int(np.sum(~p & ~a)))


def metricas(tp, fp, fn, tn, eps=1e-9):
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    n = tp + fp + fn + tn
    return {"iou": tp / (tp + fp + fn + eps),
            "f1": 2 * prec * rec / (prec + rec + eps),
            "precisao": prec, "recall": rec,
            "acuracia": (tp + tn) / (n + eps),
            "prevalencia": (tp + fn) / (n + eps),
            "frac_prevista": (tp + fp) / (n + eps),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def ic_iou(probs, alvos, validos, limiar, n=N_BOOTSTRAP, semente=SEMENTE):
    cs = [contar(np.asarray(p) >= limiar, a, v) for p, a, v in zip(probs, alvos, validos)]
    if not cs:
        return (float("nan"), float("nan"))
    arr = np.array(cs, dtype="float64")
    rng = np.random.RandomState(semente)
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(arr), len(arr))
        tp, fp, fn, _tn = arr[i].sum(axis=0)
        vals.append(tp / (tp + fp + fn + 1e-9))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def melhor_limiar(probs, alvos, validos, limiares=LIMIARES, criterio="iou"):
    melhor, valor = limiares[0], -1.0
    for t in limiares:
        tp = fp = fn = tn = 0
        for p, a, v in zip(probs, alvos, validos):
            c = contar(np.asarray(p) >= t, a, v)
            tp += c[0]; fp += c[1]; fn += c[2]; tn += c[3]
        m = metricas(tp, fp, fn, tn)[criterio]
        if m > valor:
            melhor, valor = float(t), float(m)
    return melhor, valor


# =============================================================================
# 3. BASELINES E SELEÇÃO DE CANAIS
# =============================================================================

def posicao_do_nome(caminho):
    b = os.path.basename(caminho)
    if b.endswith(".npz"):
        b = b[:-4]
    partes = b.split("_")
    return int(partes[-2][1:]), int(partes[-1][1:])


def climatologia_na_grade(arquivos, forma, lado):
    soma = np.zeros(forma, dtype="float64")
    cont = np.zeros(forma, dtype="float64")
    for cam in arquivos:
        y0, x0 = posicao_do_nome(cam)
        with np.load(cam, allow_pickle=False) as d:
            y = np.asarray(d["y"], dtype="float64")
            v = np.asarray(d["valido"], dtype="float64") if "valido" in d else np.ones_like(y)
        h, w = y.shape
        y1, x1 = min(y0 + h, forma[0]), min(x0 + w, forma[1])
        soma[y0:y1, x0:x1] += (y * v)[:y1 - y0, :x1 - x0]
        cont[y0:y1, x0:x1] += v[:y1 - y0, :x1 - x0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cont > 0, soma / np.maximum(cont, 1e-9), 0.0)


def recortar(grade_freq, y0, x0, lado):
    out = np.zeros((lado, lado), dtype="float64")
    h = min(lado, grade_freq.shape[0] - y0)
    w = min(lado, grade_freq.shape[1] - x0)
    if h > 0 and w > 0:
        out[:h, :w] = grade_freq[y0:y0 + h, x0:x0 + w]
    return out


def prever_climatologia(freq_grade, arquivos, lado):
    return [recortar(freq_grade, *posicao_do_nome(c), lado=lado) for c in arquivos]


def prever_constante(forma, valor, n):
    return [np.full(forma, float(valor))] * n


def canal_como_probabilidade(Xs, i):
    return [np.asarray(X[i], dtype="float64") for X in Xs]


def indices_dos_canais(canais, usar):
    falta = [u for u in usar if u not in canais]
    if falta:
        raise ValueError("canais inexistentes: %s" % ", ".join(falta))
    return [i for i, c in enumerate(canais) if c in usar]


def experimentos(canais, grupos=GRUPOS):
    todos = tuple(canais)
    chuva = tuple(c for c in grupos["chuva"] if c in canais)
    terreno = tuple(c for c in todos if c not in chuva)
    exps = []
    if chuva:
        exps.append(("A_so_terreno", terreno))
        exps.append(("B_so_chuva", chuva))
    exps.append(("C_completo", todos))
    for nome, membros in sorted(grupos.items()):
        presentes = tuple(c for c in membros if c in canais)
        if not presentes:
            continue
        resto = tuple(c for c in todos if c not in presentes)
        if resto and resto != todos:
            exps.append(("D_sem_%s" % nome, resto))
    vistos, saida = set(), []
    for nome, cs in exps:
        if cs and cs not in vistos:
            vistos.add(cs)
            saida.append((nome, cs))
    return saida


# =============================================================================
# 4. REDE E TREINO (Com verificação de Checkpoint)
# =============================================================================

def construir_rede(n_canais):
    import torch
    import torch.nn as nn

    def bloco(a, b):
        return nn.Sequential(
            nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b),
            nn.ReLU(inplace=True),
            nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b),
            nn.ReLU(inplace=True))

    class UNetPequena(nn.Module):
        def __init__(self, cin, base=CANAIS_BASE):
            super(UNetPequena, self).__init__()
            c = [base, base * 2, base * 4, base * 8]
            self.e1, self.e2 = bloco(cin, c[0]), bloco(c[0], c[1])
            self.e3, self.e4 = bloco(c[1], c[2]), bloco(c[2], c[3])
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.d3 = bloco(c[3] + c[2], c[2])
            self.d2 = bloco(c[2] + c[1], c[1])
            self.d1 = bloco(c[1] + c[0], c[0])
            self.saida = nn.Conv2d(c[0], 1, 1)

        def forward(self, x):
            import torch as _t
            s1 = self.e1(x)
            s2 = self.e2(self.pool(s1))
            s3 = self.e3(self.pool(s2))
            b = self.e4(self.pool(s3))
            x = self.d3(_t.cat([self.up(b), s3], 1))
            x = self.d2(_t.cat([self.up(x), s2], 1))
            x = self.d1(_t.cat([self.up(x), s1], 1))
            return self.saida(x)

    return UNetPequena(n_canais)


def tversky_mascarada(logits, alvo, valido, alfa=TVERSKY_ALFA, beta=TVERSKY_BETA, eps=1e-6):
    import torch
    p = torch.sigmoid(logits) * valido
    a = alvo * valido
    tp = (p * a).sum(dim=(1, 2, 3))
    fp = (p * (1 - a)).sum(dim=(1, 2, 3))
    fn = ((1 - p) * a).sum(dim=(1, 2, 3))
    t = (tp + eps) / (tp + alfa * fp + beta * fn + eps)
    return (1.0 - t).mean()


class LoteDePatches(object):
    def __init__(self, arquivos, idx_canais, treino, semente=SEMENTE):
        self.arquivos = list(arquivos)
        self.idx = list(idx_canais)
        self.treino = treino
        self.rng = np.random.RandomState(semente)
        self.molhados, self.secos = [], []
        for a in self.arquivos:
            with np.load(a, allow_pickle=False) as d:
                (self.molhados if int(d["y"].sum()) >= 28 else self.secos).append(a)

    def __len__(self):
        return len(self.arquivos)

    def ler(self, caminho):
        with np.load(caminho, allow_pickle=False) as d:
            X = np.asarray(d["X"], dtype="float32")[self.idx]
            y = np.asarray(d["y"], dtype="float32")[None, :, :]
            v = np.asarray(d["valido"], dtype="float32")[None, :, :] if "valido" in d else np.ones_like(y)
        return X, y, v

    def lotes(self, tamanho, por_epoca=None):
        if self.treino:
            n = por_epoca or max(1, len(self.arquivos) // tamanho)
            for _ in range(n):
                esc = []
                for _ in range(tamanho):
                    quer = self.rng.rand() < FRAC_MOLHADO_NO_LOTE
                    fonte = (self.molhados if (quer and self.molhados) else (self.secos or self.molhados))
                    esc.append(fonte[self.rng.randint(len(fonte))])
                yield self._montar(esc)
        else:
            for i in range(0, len(self.arquivos), tamanho):
                yield self._montar(self.arquivos[i:i + tamanho])

    def _montar(self, caminhos):
        Xs, ys, vs = [], [], []
        for c in caminhos:
            X, y, v = self.ler(c)
            Xs.append(X); ys.append(y); vs.append(v)
        return (np.stack(Xs), np.stack(ys), np.stack(vs))


def treinar(exp, arquivos_por_conjunto, canais, usar, destino):
    import torch
    torch.manual_seed(SEMENTE)
    np.random.seed(SEMENTE)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    idx = indices_dos_canais(canais, usar)
    rede = construir_rede(len(idx)).to(dev)
    n_par = sum(p.numel() for p in rede.parameters())

    cam = os.path.join(destino, "%s.pth" % exp)

    if os.path.exists(cam):
        print(f"  [CHECKPOINT] Modelo '{exp}' já existe em {cam}. Pulando treinamento e carregando pesos salvos.")
        rede.load_state_dict(torch.load(cam, map_location=dev)["modelo"])
        rede.eval()
        return rede, dev, idx

    print("  %s: %d canais (%s), %.2f M parametros, %s" % (exp, len(idx), ", ".join(usar), n_par / 1e6, dev))

    tr = LoteDePatches(arquivos_por_conjunto["treino"], idx, True)
    va = LoteDePatches(arquivos_por_conjunto["validacao"], idx, False)
    opt = torch.optim.AdamW(rede.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCAS)

    hist_tr, hist_va, melhor, melhor_ep, espera = [], [], 1e9, 0, 0
    for ep in range(EPOCAS):
        t0 = time.time()
        rede.train()
        soma, n = 0.0, 0
        for X, y, v in tr.lotes(LOTE):
            X = torch.from_numpy(X).to(dev)
            y = torch.from_numpy(y).to(dev)
            v = torch.from_numpy(v).to(dev)
            opt.zero_grad(set_to_none=True)
            perda = tversky_mascarada(rede(X), y, v)
            if not torch.isfinite(perda):
                continue
            perda.backward()
            torch.nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
            opt.step()
            soma += float(perda) * X.shape[0]
            n += X.shape[0]
        p_tr = soma / max(1, n)

        rede.eval()
        soma, n = 0.0, 0
        with torch.no_grad():
            for X, y, v in va.lotes(LOTE):
                X = torch.from_numpy(X).to(dev)
                y = torch.from_numpy(y).to(dev)
                v = torch.from_numpy(v).to(dev)
                perda = tversky_mascarada(rede(X), y, v)
                if torch.isfinite(perda):
                    soma += float(perda) * X.shape[0]
                    n += X.shape[0]
        p_va = soma / max(1, n)
        sched.step()
        hist_tr.append(p_tr); hist_va.append(p_va)

        marca = ""
        if p_va < melhor - 1e-4:
            melhor, melhor_ep, espera = p_va, ep + 1, 0
            torch.save({"modelo": rede.state_dict(), "canais": list(usar), "epoca": ep + 1, "perda_val": p_va}, cam)
            marca = "  <- salvo"
        else:
            espera += 1
        print("    ep %3d/%d  treino %.4f  val %.4f  %4.1fs%s" % (ep + 1, EPOCAS, p_tr, p_va, time.time() - t0, marca))
        if espera >= PACIENCIA:
            print("    parada antecipada na epoca %d (melhor foi a %d)" % (ep + 1, melhor_ep))
            break

    rede.load_state_dict(torch.load(cam, map_location=dev)["modelo"])
    rede.eval()
    return rede, dev, idx


def prever(rede, dev, arquivos, idx):
    import torch
    probs, alvos, validos = [], [], []
    ds = LoteDePatches(arquivos, idx, False)
    with torch.no_grad():
        for X, y, v in ds.lotes(LOTE):
            p = torch.sigmoid(rede(torch.from_numpy(X).to(dev)))
            p = p.cpu().numpy()
            for i in range(p.shape[0]):
                probs.append(p[i, 0]); alvos.append(y[i, 0]); validos.append(v[i, 0])
    return probs, alvos, validos


# =============================================================================
# 5. AVALIAÇÃO E RESULTADOS
# =============================================================================

def avaliar(probs, alvos, validos, limiar, exp, conjunto, canais_txt):
    tp = fp = fn = tn = 0
    for p, a, v in zip(probs, alvos, validos):
        c = contar(np.asarray(p) >= limiar, a, v)
        tp += c[0]; fp += c[1]; fn += c[2]; tn += c[3]
    m = metricas(tp, fp, fn, tn)
    lo, hi = ic_iou(probs, alvos, validos, limiar)
    m.update({"experimento": exp, "conjunto": conjunto, "limiar": limiar,
              "canais": canais_txt, "iou_ic95_lo": lo, "iou_ic95_hi": hi,
              "n_patches": len(probs)})
    print("      %-14s IoU %.4f [%.4f-%.4f]  F1 %.4f  prec %.4f  rec %.4f"
          "  molhou %.1f%% (real %.1f%%)"
          % (conjunto, m["iou"], lo, hi, m["f1"], m["precisao"], m["recall"],
             100.0 * m["frac_prevista"], 100.0 * m["prevalencia"]))
    return m


def gravar(linhas, destino):
    campos = ["experimento", "conjunto", "limiar", "iou", "iou_ic95_lo",
              "iou_ic95_hi", "f1", "precisao", "recall", "frac_prevista",
              "prevalencia", "acuracia", "n_patches", "tp", "fp", "fn", "tn",
              "canais"]
    with open(os.path.join(destino, "resultados.csv"), "w",
              encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=campos, extrasaction="ignore")
        w.writeheader()
        for r in linhas:
            w.writerow(r)


def ler_o_resultado(linhas):
    def iou(exp, conj):
        for r in linhas:
            if r["experimento"] == exp and r["conjunto"] == conj:
                return r["iou"]
        return None
    print("\n" + "=" * 70)
    print("COMO LER ISTO")
    print("=" * 70)
    clim = iou("0_climatologia", "teste_ambos")
    comp = iou("C_completo", "teste_ambos")

    if clim is not None and comp is not None:
        print("Climatologia %.4f  x  modelo completo %.4f" % (clim, comp))
        if clim == 0.0:
            print("  -> A climatologia obteve IoU 0.0000 no teste_ambos.")
            if comp > 0.0:
                print("     A U-Net pontuou acima de zero, demonstrando capacidade de generalização.")
        elif comp <= clim * 1.05:
            print("  -> A rede NAO bate o mapa fixo.")
        else:
            print("  -> A rede supera o mapa fixo em %.0f%%." % (100.0 * (comp / clim - 1)))


# =============================================================================
# 6. MAIN
# =============================================================================

def carregar_itens(dados=DADOS):
    man = json.load(open(os.path.join(dados, "manifest.json"), encoding="utf-8"))
    chuva = {r["data"]: r.get("chuva_total_mm_media", 0.0) for r in man["datas"]}
    itens = []
    for cam in sorted(glob.glob(os.path.join(dados, "patches", "*.npz"))):
        b = os.path.basename(cam)[:-4]
        try:
            data, sy, sx = b.split("_")
            y0, x0 = int(sy[1:]), int(sx[1:])
        except ValueError:
            continue
        itens.append((cam, data, y0, x0, chuva.get(data, 0.0)))
    if not itens:
        raise SystemExit("Nenhum patch encontrado.")
    return man, itens


def main():
    os.makedirs(DESTINO, exist_ok=True)
    man, itens = carregar_itens()
    canais = man["canais"]
    print(f"{len(itens)} patches carregados. Canais: {', '.join(canais)}")

    saida, papel_d, papel_r = montar_split(itens)
    print("\nSplit (sem vazamento):")
    for c in ("treino", "validacao", "teste_tempo", "teste_espaco", "teste_ambos"):
        print("  %-14s %5d patches" % (c, len(saida.get(c, []))))

    linhas = []

    # ---- 1. Baselines sem rede
    print("\n" + "=" * 70)
    print("1. BASELINES SEM REDE")
    print("=" * 70)
    todos_idx = list(range(len(canais)))
    forma = tuple(man["grade"]["forma"])
    lado = int(man["patch"]["tamanho"])
    freq = climatologia_na_grade(saida["treino"], forma, lado)

    conjuntos = {c: saida[c] for c in ("validacao", "teste_tempo", "teste_espaco", "teste_ambos") if saida.get(c)}
    cache = {}
    for c, arqs in conjuntos.items():
        d = LoteDePatches(arqs, todos_idx, False)
        Xs, ys, vs = [], [], []
        for X, y, v in d.lotes(LOTE):
            for i in range(X.shape[0]):
                Xs.append(X[i]); ys.append(y[i, 0]); vs.append(v[i, 0])
        cache[c] = (Xs, ys, vs, list(arqs))

    i_susc = canais.index("susc_via") if "susc_via" in canais else None
    base = [
        ("0_tudo_seco", lambda Xs, aq: prever_constante(Xs[0].shape[1:], 0.0, len(Xs))),
        ("0_climatologia", lambda Xs, aq: prever_climatologia(freq, aq, lado))
    ]
    if i_susc is not None:
        base.append(("0_susc_sozinha", lambda Xs, aq: canal_como_probabilidade(Xs, i_susc)))

    for nome, f_prev in base:
        Xv, yv, vv, av = cache["validacao"]
        t, m_val = melhor_limiar(f_prev(Xv, av), yv, vv)
        print("  %-16s limiar %.3f (IoU %.4f na validacao)" % (nome, t, m_val))
        for c in conjuntos:
            Xs, ys, vs, aq = cache[c]
            linhas.append(avaliar(f_prev(Xs, aq), ys, vs, t, nome, c, ""))

    # ---- 2. Modelos de Ablação de Canais
    for nome, usar in experimentos(canais):
        print("\n" + "=" * 70)
        print("MODELO %s" % nome)
        print("=" * 70)
        rede, dev, idx = treinar(nome, saida, canais, usar, DESTINO)
        pv, yv, vv = prever(rede, dev, saida["validacao"], idx)
        t, m_val = melhor_limiar(pv, yv, vv)
        print("  limiar escolhido na validacao: %.2f (IoU %.4f)" % (t, m_val))
        for c in conjuntos:
            ps, ys, vs = prever(rede, dev, saida[c], idx)
            linhas.append(avaliar(ps, ys, vs, t, nome, c, " ".join(usar)))
        gravar(linhas, DESTINO)

    # ---- 3. Experimento 1: Balanceamento de Dados (Patches Secos)
    print("\n" + "=" * 70)
    print("EXECUTANDO EXPERIMENTO 1: BALANCEAMENTO DE DADOS (Patches Secos)")
    print("=" * 70)

    treino_original = saida["treino"]
    for frac_secos in PROPORCOES_SECOS:
        exp_nome = f"bal_secos_{int(frac_secos * 100)}pct"
        print(f"\n--- Processando Experimento com {int(frac_secos * 100)}% de patches secos ---")

        saida_experimento = dict(saida)
        saida_experimento["treino"] = filtrar_treino_por_balanceamento(treino_original, frac_secos)

        usar = tuple(canais)
        rede, dev, idx = treinar(exp_nome, saida_experimento, canais, usar, DESTINO)

        pv, yv, vv = prever(rede, dev, saida_experimento["validacao"], idx)
        t, m_val = melhor_limiar(pv, yv, vv)

        ps, ys, vs = prever(rede, dev, saida_experimento["teste_ambos"], idx)
        tp = fp = fn = tn = 0
        for p, a, v in zip(ps, ys, vs):
            c = contar(np.asarray(p) >= t, a, v)
            tp += c[0]; fp += c[1]; fn += c[2]; tn += c[3]
        m = metricas(tp, fp, fn, tn)
        lo, hi = ic_iou(ps, ys, vs, t)

        # ADIÇÃO: Incluídas todas as chaves básicas de métrica para alinhar 
        # com os campos esperados pelo `gravar()` e evitar erros no DictWriter.
        m.update({
            "experimento": exp_nome,
            "conjunto": "teste_ambos",
            "limiar": t,
            "canais": "completo",
            "iou_ic95_lo": lo,
            "iou_ic95_hi": hi,
            "n_patches": len(ps)
        })
        linhas.append(m)

    gravar(linhas, DESTINO)
    ler_o_resultado(linhas)
    print(f"\n[SUCESSO] Processo finalizado! Resultados salvos em: {os.path.join(DESTINO, 'resultados.csv')}")
    return 0


if __name__ == "__main__":
    main()