"""
================================================================================
nlp_34 -- TREINAR A U-NET
================================================================================

ENTRADA
  DATASET_UNET/manifest.json e DATASET_UNET/patches/*.npz   (nlp_33)

SAIDA
  TREINO_UNET/resultados.csv        uma linha por (experimento, conjunto)
  TREINO_UNET/<exp>.pth             o melhor peso de cada experimento
  TREINO_UNET/<exp>_perda.png       treino x validacao
  TREINO_UNET/comparacao.png        todos contra os baselines
  TREINO_UNET/split.json            quais datas e regioes em cada conjunto
================================================================================
"""

import os
import csv
import json
import glob
import time

import numpy as np

MODO = "completo"          # "teste" | "baselines" | "completo"

PASTA = r"C:\Users\m291018\OneDrive\05-Mateus Domingos\INMET"
DADOS = os.path.join(PASTA, "DATASET_UNET")
DESTINO = os.path.join(PASTA, "TREINO_UNET")

SEMENTE = 20261022

# --- split ---
FRAC_DATAS = (0.70, 0.15, 0.15)     # treino, validacao, teste
LADO_REGIAO_PX = 512                # 512 px de 20 m = 10,2 km
FRAC_REGIOES_RETIDAS = 0.25

# --- treino ---
ARQUITETURA = "unet_pequena"        # "unet_pequena" | "smp"
CANAIS_BASE = 32
EPOCAS = 120
PACIENCIA = 15
LOTE = 16
LR = 3e-4
WEIGHT_DECAY = 1e-5
FRAC_MOLHADO_NO_LOTE = 0.7          # sobreamostragem de patch com agua
TVERSKY_ALFA, TVERSKY_BETA = 0.3, 0.7   # beta > alfa pune falso negativo
WORKERS = 0                         # 0 e o mais seguro no Windows

# --- avaliacao ---
LIMIARES = tuple(np.round(np.arange(0.01, 0.991, 0.02), 3))
N_BOOTSTRAP = 300       # reamostragem de patches para o IC da IoU

# Grupos de canais para a ablacao. Tirar um canal por vez nao mostra nada
# quando tres deles dizem a mesma coisa de tres jeitos.
GRUPOS = {
    "chuva": ("precip_total_mm", "precip_max_mmh"),
    "urbano": ("lulc_idx", "imperv", "susc_via"),
    "relevo": ("decl_pct",),
}


# =============================================================================
# 1. SPLIT (nucleo testavel)
# =============================================================================

def regiao_do_patch(y0, x0, lado=LADO_REGIAO_PX):
    return (int(y0) // int(lado), int(x0) // int(lado))


def dividir_datas(datas, chuva, frac=FRAC_DATAS, semente=SEMENTE):
    """Datas em treino/validacao/teste, estratificado por magnitude de chuva.

    Sorteio puro com 54 datas poe os tres eventos extremos no mesmo conjunto
    com probabilidade alta. Ordeno por chuva e distribuo em ciclo, o que
    garante que cada conjunto recebe eventos de todas as faixas."""
    ordem = sorted(range(len(datas)), key=lambda i: (-float(chuva[i]),
                                                     str(datas[i])))
    n = len(ordem)
    alvo = [int(round(f * n)) for f in frac]
    alvo[0] = n - alvo[1] - alvo[2]
    # A sequencia e embaralhada e depois percorrida NA ORDEM DE CHUVA: assim
    # cada conjunto recebe datas de todas as faixas, sem que a posicao no
    # ranking de chuva determine o conjunto.
    rng = np.random.RandomState(semente)
    sequencia = (["treino"] * alvo[0] + ["val"] * alvo[1] +
                 ["teste"] * alvo[2])
    rng.shuffle(sequencia)
    out = {}
    for pos, i in enumerate(ordem):
        out[str(datas[i])] = sequencia[pos]
    return out


def dividir_regioes(regioes, frac_retida=FRAC_REGIOES_RETIDAS,
                    semente=SEMENTE):
    """Regioes em 'treino' e 'retida'. Sorteio, mas com semente fixa e
    garantindo ao menos uma retida."""
    regs = sorted(set(regioes))
    rng = np.random.RandomState(semente + 1)
    n_ret = max(1, int(round(frac_retida * len(regs))))
    idx = rng.permutation(len(regs))[:n_ret]
    retidas = {regs[i] for i in idx}
    return {r: ("retida" if r in retidas else "treino") for r in regs}


def conjunto_do_patch(papel_data, papel_regiao):
    """A tabela 2x2 do cabecalho. None = patch nao usado."""
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
    return None            # val x regiao retida: fora, para nao contaminar


def montar_split(itens, lado=LADO_REGIAO_PX, frac=FRAC_DATAS,
                 frac_reg=FRAC_REGIOES_RETIDAS, semente=SEMENTE):
    """itens = [(caminho, data, y0, x0, chuva_media)] -> dict conjunto -> lista.

    Devolve tambem os papeis, para o teste conferir que nao ha vazamento."""
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


def conferir_vazamento(saida, itens, papel_d, papel_r, lado=LADO_REGIAO_PX):
    """Devolve lista de problemas. Vazia = limpo.

    Confere o que de fato importa: nenhuma DATA de treino aparece nos testes
    temporais, e nenhuma REGIAO de treino aparece nos testes espaciais."""
    meta = {cam: (d, regiao_do_patch(y0, x0, lado))
            for cam, d, y0, x0, _ in itens}
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


# =============================================================================
# 2. METRICAS MASCARADAS (nucleo testavel)
# =============================================================================

def contar(pred, alvo, valido=None):
    """(tp, fp, fn, tn) contados SO onde valido. pred e alvo binarios."""
    p = np.asarray(pred).astype(bool).ravel()
    a = np.asarray(alvo).astype(bool).ravel()
    if valido is None:
        m = np.ones_like(p, dtype=bool)
    else:
        m = np.asarray(valido).astype(bool).ravel()
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
            # quanto da area valida o modelo molhou. Com prevalencia de 1 a 2%,
            # a IoU e maximizada molhando muito: sem este numero ao lado, um
            # limiar que inunda metade da cidade parece um bom resultado.
            "frac_prevista": (tp + fp) / (n + eps),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def ic_iou(probs, alvos, validos, limiar, n=N_BOOTSTRAP, semente=SEMENTE):
    """IC 95% da IoU reamostrando PATCHES (nao pixels).

    Com 32 patches em teste_ambos, diferenca de 0,01 na IoU nao significa
    nada. Reamostrar pixel subestimaria o intervalo, porque pixels do mesmo
    patch nao sao independentes."""
    cs = [contar(np.asarray(p) >= limiar, a, v)
          for p, a, v in zip(probs, alvos, validos)]
    if not cs:
        return (float("nan"), float("nan"))
    arr = np.array(cs, dtype="float64")
    rng = np.random.RandomState(semente)
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(arr), len(arr))
        tp, fp, fn, _tn = arr[i].sum(axis=0)
        vals.append(tp / (tp + fp + fn + 1e-9))
    return (float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def melhor_limiar(probs, alvos, validos, limiares=LIMIARES, criterio="iou"):
    """Escolhe o limiar que maximiza o criterio. Tem de ser chamado com os
    dados de VALIDACAO -- escolher limiar no teste e contar duas vezes."""
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
# 3. BASELINES SEM REDE (nucleo testavel)
# =============================================================================

def posicao_do_nome(caminho):
    """(y0, x0) do nome AAAAMMDD_y####_x####.npz."""
    b = os.path.basename(caminho)
    if b.endswith(".npz"):
        b = b[:-4]
    partes = b.split("_")
    if len(partes) < 3 or not partes[-2].startswith("y") \
            or not partes[-1].startswith("x"):
        raise ValueError("nome fora do padrao: %s" % caminho)
    return int(partes[-2][1:]), int(partes[-1][1:])


def climatologia_na_grade(arquivos, forma, lado):
    """Frequencia de y = 1 por celula da GRADE, sobre as datas de treino.

    ISTO TEM DE SER NA GRADE, nao no referencial do patch. A primeira versao
    somava todos os patches de treino num mesmo array 256x256 -- patches de
    posicoes diferentes da cidade empilhados como se fossem o mesmo lugar. O
    resultado acendia 75,8% das celulas e o baseline saia fraco por engano.
    Mapa fixo e por COORDENADA, e a coordenada esta no nome do arquivo."""
    soma = np.zeros(forma, dtype="float64")
    cont = np.zeros(forma, dtype="float64")
    for cam in arquivos:
        y0, x0 = posicao_do_nome(cam)
        with np.load(cam, allow_pickle=False) as d:
            y = np.asarray(d["y"], dtype="float64")
            v = (np.asarray(d["valido"], dtype="float64")
                 if "valido" in d else np.ones_like(y))
        h, w = y.shape
        y1, x1 = min(y0 + h, forma[0]), min(x0 + w, forma[1])
        soma[y0:y1, x0:x1] += (y * v)[:y1 - y0, :x1 - x0]
        cont[y0:y1, x0:x1] += v[:y1 - y0, :x1 - x0]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cont > 0, soma / np.maximum(cont, 1e-9), 0.0)


def recortar(grade_freq, y0, x0, lado):
    """O pedaco do mapa fixo que corresponde a um patch."""
    out = np.zeros((lado, lado), dtype="float64")
    h = min(lado, grade_freq.shape[0] - y0)
    w = min(lado, grade_freq.shape[1] - x0)
    if h > 0 and w > 0:
        out[:h, :w] = grade_freq[y0:y0 + h, x0:x0 + w]
    return out


def prever_climatologia(freq_grade, arquivos, lado):
    return [recortar(freq_grade, *posicao_do_nome(c), lado=lado)
            for c in arquivos]


def prever_constante(forma, valor, n):
    return [np.full(forma, float(valor))] * n


def canal_como_probabilidade(Xs, i):
    """Usa um canal de entrada (ja normalizado em 0..1) direto como
    probabilidade. Serve para perguntar se a camada 1 do nlp_22 ja resolve o
    problema sozinha."""
    return [np.asarray(X[i], dtype="float64") for X in Xs]


# =============================================================================
# 4. SELECAO DE CANAIS (nucleo testavel)
# =============================================================================

def indices_dos_canais(canais, usar):
    """usar = tupla de nomes. Devolve os indices, na ordem de `canais`."""
    falta = [u for u in usar if u not in canais]
    if falta:
        raise ValueError("canais inexistentes: %s" % ", ".join(falta))
    return [i for i, c in enumerate(canais) if c in usar]


def experimentos(canais, grupos=GRUPOS):
    """A lista, na ordem do cabecalho deste arquivo."""
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
# 5. REDE E TREINO (precisa de torch)
# =============================================================================

def construir_rede(n_canais):
    import torch
    import torch.nn as nn

    if ARQUITETURA == "smp":
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name="efficientnet-b4",
                        encoder_weights="imagenet",
                        in_channels=n_canais, classes=1)

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
            self.up = nn.Upsample(scale_factor=2, mode="bilinear",
                                  align_corners=False)
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


def tversky_mascarada(logits, alvo, valido, alfa=TVERSKY_ALFA,
                      beta=TVERSKY_BETA, eps=1e-6):
    """Tversky contada so onde valido.

    Sem a mascara, os 55% de celula fora do dominio entram como verdadeiro
    negativo e a loss cai sem o modelo melhorar em nada."""
    import torch
    p = torch.sigmoid(logits) * valido
    a = alvo * valido
    tp = (p * a).sum(dim=(1, 2, 3))
    fp = (p * (1 - a)).sum(dim=(1, 2, 3))
    fn = ((1 - p) * a).sum(dim=(1, 2, 3))
    t = (tp + eps) / (tp + alfa * fp + beta * fn + eps)
    return (1.0 - t).mean()


class LoteDePatches(object):
    """Carrega .npz e serve lotes. Sem DataLoader para nao depender de
    multiprocessing no Windows e para manter a sobreamostragem explicita."""

    def __init__(self, arquivos, idx_canais, treino, semente=SEMENTE):
        self.arquivos = list(arquivos)
        self.idx = list(idx_canais)
        self.treino = treino
        self.rng = np.random.RandomState(semente)
        self.molhados, self.secos = [], []
        for a in self.arquivos:
            with np.load(a, allow_pickle=False) as d:
                (self.molhados if int(d["y"].sum()) >= 28
                 else self.secos).append(a)

    def __len__(self):
        return len(self.arquivos)

    def ler(self, caminho):
        with np.load(caminho, allow_pickle=False) as d:
            X = np.asarray(d["X"], dtype="float32")[self.idx]
            y = np.asarray(d["y"], dtype="float32")[None, :, :]
            v = (np.asarray(d["valido"], dtype="float32")[None, :, :]
                 if "valido" in d else np.ones_like(y))
        return X, y, v

    def lotes(self, tamanho, por_epoca=None):
        if self.treino:
            n = por_epoca or max(1, len(self.arquivos) // tamanho)
            for _ in range(n):
                esc = []
                for _ in range(tamanho):
                    quer = self.rng.rand() < FRAC_MOLHADO_NO_LOTE
                    fonte = (self.molhados if (quer and self.molhados)
                             else (self.secos or self.molhados))
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
    print("  %s: %d canais (%s), %.2f M parametros, %s"
          % (exp, len(idx), ", ".join(usar), n_par / 1e6, dev))

    tr = LoteDePatches(arquivos_por_conjunto["treino"], idx, True)
    va = LoteDePatches(arquivos_por_conjunto["validacao"], idx, False)
    opt = torch.optim.AdamW(rede.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCAS)

    hist_tr, hist_va, melhor, melhor_ep, espera = [], [], 1e9, 0, 0
    cam = os.path.join(destino, "%s.pth" % exp)
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
            torch.save({"modelo": rede.state_dict(), "canais": list(usar),
                        "epoca": ep + 1, "perda_val": p_va}, cam)
            marca = "  <- salvo"
        else:
            espera += 1
        print("    ep %3d/%d  treino %.4f  val %.4f  %4.1fs%s"
              % (ep + 1, EPOCAS, p_tr, p_va, time.time() - t0, marca))
        if espera >= PACIENCIA:
            print("    parada antecipada na epoca %d (melhor foi a %d)"
                  % (ep + 1, melhor_ep))
            break

    rede.load_state_dict(torch.load(cam, map_location=dev)["modelo"])
    rede.eval()
    plotar_perda(hist_tr, hist_va, melhor_ep,
                 os.path.join(destino, "%s_perda.png" % exp), exp)
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
                probs.append(p[i, 0]); alvos.append(y[i, 0])
                validos.append(v[i, 0])
    return probs, alvos, validos


def plotar_perda(tr, va, melhor_ep, caminho, exp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(tr) + 1), tr, label="treino")
    plt.plot(range(1, len(va) + 1), va, label="validacao")
    if melhor_ep:
        plt.axvline(melhor_ep, color="red", ls="--", alpha=0.6,
                    label="melhor (ep %d)" % melhor_ep)
    plt.xlabel("epoca"); plt.ylabel("Tversky mascarada")
    plt.title(exp); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(caminho, dpi=150); plt.close()


# =============================================================================
# 6. MAIN
# =============================================================================

def carregar_itens(dados=DADOS):
    man = json.load(open(os.path.join(dados, "manifest.json"),
                         encoding="utf-8"))
    chuva = {r["data"]: r.get("chuva_total_mm_media", 0.0)
             for r in man["datas"]}
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
        raise SystemExit("Nenhum patch em %s. Rode o nlp_33."
                         % os.path.join(dados, "patches"))
    return man, itens


def main():
    if MODO == "teste":
        return teste()
    os.makedirs(DESTINO, exist_ok=True)
    man, itens = carregar_itens()
    canais = man["canais"]
    print("%d patches, %d datas, canais: %s"
          % (len(itens), len({i[1] for i in itens}), ", ".join(canais)))

    saida, papel_d, papel_r = montar_split(itens)
    problemas = conferir_vazamento(saida, itens, papel_d, papel_r)
    if problemas:
        raise SystemExit("VAZAMENTO no split: " + "; ".join(problemas))
    print("\nSplit (sem vazamento):")
    for c in ("treino", "validacao", "teste_tempo", "teste_espaco",
              "teste_ambos"):
        print("  %-14s %5d patches" % (c, len(saida.get(c, []))))
    retidas = sorted(r for r, p in papel_r.items() if p == "retida")
    print("  regioes retidas: %s de %d" % (retidas, len(papel_r)))
    with open(os.path.join(DESTINO, "split.json"), "w",
              encoding="utf-8") as f:
        json.dump({"datas": papel_d,
                   "regioes": {str(k): v for k, v in papel_r.items()},
                   "contagem": {k: len(v) for k, v in saida.items()}},
                  f, ensure_ascii=False, indent=2)

    for c in ("treino", "validacao", "teste_ambos"):
        if not saida.get(c):
            raise SystemExit("Conjunto %r vazio. Com %d datas e %d regioes "
                             "nao da para montar o split bidimensional; "
                             "baixe LADO_REGIAO_PX ou FRAC_REGIOES_RETIDAS."
                             % (c, len({i[1] for i in itens}), len(papel_r)))

    linhas = []

    # ---- 1. baselines, sem rede
    print("\n" + "=" * 70)
    print("1. BASELINES SEM REDE")
    print("=" * 70)
    todos_idx = list(range(len(canais)))
    forma = tuple(man["grade"]["forma"])
    lado = int(man["patch"]["tamanho"])
    freq = climatologia_na_grade(saida["treino"], forma, lado)
    print("  climatologia na GRADE (%d x %d): %.2f%% das celulas com algum"
          % (forma[0], forma[1], 100.0 * float((freq > 0).mean())))
    print("  alagamento nas datas de treino; frequencia mediana dessas: %.3f"
          % (float(np.median(freq[freq > 0])) if np.any(freq > 0) else 0.0))

    conjuntos = {c: saida[c] for c in ("validacao", "teste_tempo",
                                       "teste_espaco", "teste_ambos")
                 if saida.get(c)}
    cache = {}
    for c, arqs in conjuntos.items():
        d = LoteDePatches(arqs, todos_idx, False)
        Xs, ys, vs = [], [], []
        for X, y, v in d.lotes(LOTE):
            for i in range(X.shape[0]):
                Xs.append(X[i]); ys.append(y[i, 0]); vs.append(v[i, 0])
        cache[c] = (Xs, ys, vs, list(arqs))

    i_susc = canais.index("susc_via") if "susc_via" in canais else None
    base = [("0_tudo_seco",
             lambda Xs, aq: prever_constante(Xs[0].shape[1:], 0.0, len(Xs))),
            ("0_climatologia",
             lambda Xs, aq: prever_climatologia(freq, aq, lado))]
    if i_susc is not None:
        base.append(("0_susc_sozinha",
                     lambda Xs, aq: canal_como_probabilidade(Xs, i_susc)))

    for nome, f_prev in base:
        Xv, yv, vv, av = cache["validacao"]
        t, m_val = melhor_limiar(f_prev(Xv, av), yv, vv)
        print("  %-16s limiar %.3f (IoU %.4f na validacao)"
              % (nome, t, m_val))
        for c in conjuntos:
            Xs, ys, vs, aq = cache[c]
            linhas.append(avaliar(f_prev(Xs, aq), ys, vs, t, nome, c, ""))

    if MODO == "baselines":
        gravar(linhas, DESTINO)
        print("\nSo os baselines, como pedido. O melhor IoU de climatologia e "
              "a barra que a U-Net tem de passar.")
        return 0

    # ---- 2 a 4. os modelos
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

    gravar(linhas, DESTINO)
    comparar(linhas, DESTINO)
    ler_o_resultado(linhas)
    return 0


def avaliar(probs, alvos, validos, limiar, exp, conjunto, canais_txt):
    """Conta, calcula, imprime com IC, e devolve a linha do csv."""
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


def comparar(linhas, destino):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    alvo = "teste_ambos"
    sel = [r for r in linhas if r["conjunto"] == alvo]
    if not sel:
        return
    sel.sort(key=lambda r: r["iou"])
    nomes = [r["experimento"] for r in sel]
    vals = [r["iou"] for r in sel]
    cores = ["#b8431c" if n.startswith("0_") else "#184f95" for n in nomes]
    plt.figure(figsize=(10, 0.6 * len(sel) + 2))
    plt.barh(range(len(sel)), vals, color=cores)
    plt.yticks(range(len(sel)), nomes)
    plt.xlabel("IoU em %s (so celulas validas)" % alvo)
    plt.title("U-Net (azul) contra baselines sem rede (laranja)")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(destino, "comparacao.png"), dpi=150)
    plt.close()


def ler_o_resultado(linhas):
    """Imprime a leitura que importa, para nao ficar so a tabela."""
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
    terr = iou("A_so_terreno", "teste_ambos")
    if clim is not None and comp is not None:
        print("Climatologia %.4f  x  modelo completo %.4f" % (clim, comp))
        if comp <= clim * 1.05:
            print("  -> A rede NAO bate o mapa fixo. Nada do que ela aprendeu")
            print("     depende da data. Resultado negativo, e e um resultado.")
        else:
            print("  -> A rede supera o mapa fixo em %.0f%%: ha sinal"
                  " dependente da data." % (100.0 * (comp / clim - 1)))
    if terr is not None and comp is not None:
        print("So terreno %.4f  x  completo %.4f" % (terr, comp))
        if terr >= comp * 0.95:
            print("  -> A chuva nao esta sendo usada. Confira se os canais de")
            print("     chuva variam entre datas (o nlp_33 avisa quando 'uniforme').")
    tt, te, ta = (iou("C_completo", "teste_tempo"),
                  iou("C_completo", "teste_espaco"),
                  iou("C_completo", "teste_ambos"))
    if None not in (tt, te, ta):
        print("Completo: tempo %.4f | espaco %.4f | ambos %.4f" % (tt, te, ta))
        if ta < min(tt, te) * 0.7:
            print("  -> Cai muito quando data E regiao sao novas. E o numero")
            print("     que vai para o artigo, nao os outros dois.")


# =============================================================================
# 7. TESTE
# =============================================================================

def teste():
    print("MODO TESTE: split, metricas e baselines sinteticos.\n")
    ok = True

    def falha(m):
        print("FALHOU " + m)

    # 1. regiao do patch
    if regiao_do_patch(0, 0, 512) != (0, 0) or \
            regiao_do_patch(600, 1100, 512) != (1, 2):
        falha("regiao_do_patch -> %s" % (regiao_do_patch(600, 1100, 512),))
        ok = False
    else:
        print("regiao_do_patch (bloco de 512 px): ok")

    # 2. a tabela 2x2 do split
    casos = {("treino", "treino"): "treino",
             ("val", "treino"): "validacao",
             ("teste", "treino"): "teste_tempo",
             ("treino", "retida"): "teste_espaco",
             ("teste", "retida"): "teste_ambos",
             ("val", "retida"): None}
    erros = [k for k, v in casos.items() if conjunto_do_patch(*k) != v]
    if erros:
        falha("conjunto_do_patch nos casos %s" % erros); ok = False
    else:
        print("conjunto_do_patch (val x regiao retida fica FORA): ok")

    # 3. O TESTE QUE IMPORTA: vazamento.
    # Monto 12 datas x 12 regioes e exijo que nenhuma data de treino apareca
    # nos testes temporais e nenhuma regiao de treino nos espaciais.
    itens = []
    for di in range(12):
        for ry in range(3):
            for rx in range(4):
                itens.append(("p_%02d_%d_%d.npz" % (di, ry, rx),
                              "2024%02d01" % (di + 1),
                              ry * 512, rx * 512, 10.0 * di))
    saida, pd_, pr_ = montar_split(itens, lado=512)
    probs = conferir_vazamento(saida, itens, pd_, pr_, lado=512)
    if probs:
        falha("vazamento no split: %s" % probs); ok = False
    elif not saida.get("teste_ambos"):
        falha("teste_ambos vazio -- o split 2D nao se formou"); ok = False
    else:
        print("montar_split (%d conjuntos, sem vazamento, teste_ambos com %d):"
              " ok" % (len(saida), len(saida["teste_ambos"])))

    # 4. e o detector de vazamento tem de PEGAR um split sujo, senao ele
    # nao prova nada
    sujo = dict(saida)
    sujo["teste_tempo"] = list(saida["teste_tempo"]) + [itens[0][0]]
    if not conferir_vazamento(sujo, itens, pd_, pr_, lado=512):
        falha("conferir_vazamento nao detectou patch de treino no teste")
        ok = False
    else:
        print("conferir_vazamento (detecta split sujo): ok")

    # 5. estratificacao: os tres conjuntos tem de receber datas de faixas
    # diferentes de chuva, nao as tres maiores no mesmo
    chuvas = {d: 10.0 * i for i, d in enumerate(sorted(pd_))}
    por = {}
    for d, p in pd_.items():
        por.setdefault(p, []).append(chuvas[d])
    if len(por) != 3:
        falha("dividir_datas -> %s" % sorted(por)); ok = False
    elif min(len(v) for v in por.values()) == 0:
        falha("conjunto de datas vazio"); ok = False
    else:
        print("dividir_datas (3 conjuntos: %s): ok"
              % ", ".join("%s=%d" % (k, len(v)) for k, v in sorted(por.items())))

    # 6. METRICA MASCARADA. Sem mascara, o acerto em area invalida conta e a
    # IoU muda. Se mascarar nao mudasse nada, a mascara nao estaria sendo
    # usada -- e esse e o bug que eu quero impossivel.
    alvo = np.array([[1, 0], [0, 0]])
    pred = np.array([[1, 1], [0, 0]])
    val = np.array([[1, 1], [0, 0]])
    sem = metricas(*contar(pred, alvo))
    com = metricas(*contar(pred, alvo, val))
    if abs(sem["acuracia"] - 0.75) > 1e-9:
        falha("acuracia sem mascara -> %.4f" % sem["acuracia"]); ok = False
    elif abs(com["acuracia"] - 0.5) > 1e-9:
        falha("acuracia com mascara -> %.4f" % com["acuracia"]); ok = False
    elif abs(com["iou"] - 0.5) > 1e-6:
        falha("iou com mascara -> %.4f" % com["iou"]); ok = False
    else:
        print("contar/metricas (a mascara MUDA o resultado: acuracia 0,75 ->"
              " 0,50): ok")

    # 7. o limiar e escolhido onde manda, e maximiza mesmo
    probs_v = [np.array([[0.9, 0.4], [0.1, 0.1]])]
    alvos_v = [np.array([[1, 0], [0, 0]])]
    val_v = [np.ones((2, 2))]
    t, m = melhor_limiar(probs_v, alvos_v, val_v)
    if not (0.4 < t <= 0.9):
        falha("melhor_limiar -> %.2f" % t); ok = False
    elif abs(m - 1.0) > 1e-6:
        falha("IoU no melhor limiar -> %.4f, esperado 1" % m); ok = False
    else:
        print("melhor_limiar (acha o limiar perfeito neste caso): ok")

    # 8. A CLIMATOLOGIA TEM DE SER NA GRADE.
    # A primeira versao somava todos os patches de treino num mesmo array
    # 256x256 -- posicoes diferentes da cidade empilhadas como se fossem o
    # mesmo lugar. Acendia 75,8% das celulas e o baseline saia fraco por
    # engano. Aqui dois patches em posicoes DIFERENTES, cada um com uma
    # celula molhada: na grade, cada molhada fica no seu lugar; empilhados,
    # as duas apareceriam nos dois patches.
    import tempfile as _t2
    d8 = _t2.mkdtemp()
    y_a = np.zeros((4, 4), dtype="uint8"); y_a[0, 0] = 1
    y_b = np.zeros((4, 4), dtype="uint8"); y_b[0, 0] = 1
    v4 = np.ones((4, 4), dtype="uint8")
    np.savez(os.path.join(d8, "20240101_y0000_x0000.npz"), y=y_a, valido=v4)
    np.savez(os.path.join(d8, "20240101_y0004_x0004.npz"), y=y_b, valido=v4)
    arqs8 = sorted(glob.glob(os.path.join(d8, "*.npz")))
    f = climatologia_na_grade(arqs8, (8, 8), 4)
    if posicao_do_nome(arqs8[1]) != (4, 4):
        falha("posicao_do_nome -> %s" % (posicao_do_nome(arqs8[1]),))
        ok = False
    elif abs(f[0, 0] - 1.0) > 1e-9 or abs(f[4, 4] - 1.0) > 1e-9:
        falha("climatologia_na_grade nao poe cada molhada no seu lugar -> %s"
              % f); ok = False
    elif f[0, 4] != 0.0 or f[4, 0] != 0.0:
        falha("houve contaminacao entre posicoes -> %s" % f); ok = False
    else:
        # e o recorte devolve o pedaco certo
        r = recortar(f, 4, 4, 4)
        if abs(r[0, 0] - 1.0) > 1e-9 or r.sum() != 1.0:
            falha("recortar -> %s" % r); ok = False
        else:
            print("climatologia_na_grade (por COORDENADA; sem contaminacao"
                  " entre posicoes): ok")

    # 8b. frequencia parcial e celula invalida fora da conta
    y_c = np.zeros((4, 4), dtype="uint8"); y_c[0, 0] = 1
    v_c = np.ones((4, 4), dtype="uint8"); v_c[1, 1] = 0
    np.savez(os.path.join(d8, "20240202_y0000_x0000.npz"),
             y=np.zeros((4, 4), dtype="uint8"), valido=v_c)
    f2 = climatologia_na_grade(
        sorted(glob.glob(os.path.join(d8, "*_y0000_x0000.npz"))), (8, 8), 4)
    if abs(f2[0, 0] - 0.5) > 1e-9:
        falha("frequencia de 1 em 2 datas -> %.4f" % f2[0, 0]); ok = False
    else:
        print("climatologia_na_grade (1 de 2 datas -> 0,5): ok")

    # 8c. o IC por bootstrap: um caso perfeito tem IC colado em 1
    lo, hi = ic_iou([np.array([[1.0]])], [np.array([[1]])],
                    [np.array([[1]])], 0.5, n=50)
    if abs(lo - 1.0) > 1e-6 or abs(hi - 1.0) > 1e-6:
        falha("ic_iou no caso perfeito -> [%.4f, %.4f]" % (lo, hi)); ok = False
    else:
        lo2, hi2 = ic_iou([np.array([[1.0]]), np.array([[0.0]])],
                          [np.array([[1]]), np.array([[1]])],
                          [np.array([[1]]), np.array([[1]])], 0.5, n=200)
        if not (lo2 < hi2):
            falha("ic_iou nao abriu intervalo em caso misto"); ok = False
        else:
            print("ic_iou (reamostra patches; abre intervalo: %.2f a %.2f):"
                  " ok" % (lo2, hi2))

    # 9. a ordem dos experimentos, e os grupos
    canais = ["precip_total_mm", "precip_max_mmh", "lulc_idx", "imperv",
              "decl_pct", "susc_via"]
    exps = dict(experimentos(canais))
    nomes = [n for n, _ in experimentos(canais)]
    if nomes[0] != "A_so_terreno" or nomes[1] != "B_so_chuva":
        falha("a ordem nao comeca pelos cegos -> %s" % nomes[:3]); ok = False
    elif "precip_total_mm" in exps["A_so_terreno"]:
        falha("so_terreno levou chuva"); ok = False
    elif set(exps["B_so_chuva"]) != {"precip_total_mm", "precip_max_mmh"}:
        falha("so_chuva -> %s" % (exps["B_so_chuva"],)); ok = False
    elif any(c in exps["D_sem_urbano"] for c in ("lulc_idx", "imperv",
                                                 "susc_via")):
        falha("sem_urbano manteve canal do grupo -> %s"
              % (exps["D_sem_urbano"],)); ok = False
    elif len(exps["C_completo"]) != 6:
        falha("completo -> %d canais" % len(exps["C_completo"])); ok = False
    else:
        print("experimentos (cegos primeiro; ablacao por GRUPO): ok")

    # 10. indices_dos_canais respeita a ordem do dataset, nao a de `usar`
    i = indices_dos_canais(canais, ("susc_via", "precip_total_mm"))
    if i != [0, 5]:
        falha("indices_dos_canais -> %s (tem de seguir a ordem do dataset)"
              % i); ok = False
    else:
        try:
            indices_dos_canais(canais, ("nao_existe",))
            falha("indices_dos_canais aceitou canal inexistente"); ok = False
        except ValueError:
            print("indices_dos_canais (ordem do dataset; recusa inexistente):"
                  " ok")

    # 11. a loss mascarada, se torch estiver instalado
    try:
        import torch
    except ImportError:
        print("tversky_mascarada: torch nao instalado aqui, NAO testado")
    else:
        lg = torch.full((1, 1, 4, 4), 10.0)       # preve tudo molhado
        al = torch.zeros((1, 1, 4, 4))
        al[0, 0, :2, :] = 1.0                     # metade de cima e agua
        vd = torch.zeros((1, 1, 4, 4))
        vd[0, 0, :2, :] = 1.0                     # so a metade de cima vale
        p_mask = float(tversky_mascarada(lg, al, vd))
        p_sem = float(tversky_mascarada(lg, al, torch.ones_like(vd)))
        if p_mask > 0.01:
            falha("loss mascarada deveria ser ~0 -> %.4f" % p_mask); ok = False
        elif p_sem <= p_mask + 0.05:
            falha("a mascara nao mudou a loss (%.4f vs %.4f)"
                  % (p_sem, p_mask)); ok = False
        else:
            print("tversky_mascarada (0 quando acerta o valido; a mascara"
                  " muda a loss): ok")
            rede = construir_rede(3)
            x = torch.randn(2, 3, 64, 64)
            s = rede(x).shape
            if tuple(s) != (2, 1, 64, 64):
                falha("a rede devolveu %s" % (tuple(s),)); ok = False
            else:
                print("construir_rede (%.2f M parametros, saida %s): ok"
                      % (sum(p.numel() for p in rede.parameters()) / 1e6,
                         tuple(s)))

    print("")
    print("TESTE PASSOU" if ok else "TESTE FALHOU")
    print("")
    print("-" * 70)
    print("ORDEM DE USO:")
    print("  1. nlp_31 com MODO = \"todos\"  (chuva de grade em todas as datas)")
    print("  2. nlp_33                      (o dataset, que agora recusa")
    print("                                  data com mancha e chuva zero)")
    print("  3. aqui com MODO = \"baselines\" -- so a climatologia, em minutos.")
    print("     O IoU dela e a barra que a U-Net tem de passar.")
    print("  4. aqui com MODO = \"completo\"")
    print("-" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--teste" in sys.argv:
        MODO = "teste"
    raise SystemExit(main())
