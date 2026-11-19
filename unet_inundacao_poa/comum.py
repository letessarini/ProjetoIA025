"""
Funcoes compartilhadas: leitura dos patches, split sem vazamento, metricas,
climatologia, redes e previsao.
"""

import glob
import json
import os
import sys
import time
import zipfile

import numpy as np

import config


# =============================================================================
# 1. DADOS
# =============================================================================

class PatchCorrompido(Exception):
    """Um .npz que nao abre nem tentando de novo."""


def carregar_itens(dados=None):
    """Le o manifest e lista os patches: (manifest, [(caminho, data, y0, x0, chuva_media)])."""
    dados = dados or config.DADOS
    if not os.path.isfile(os.path.join(dados, "manifest.json")):
        raise SystemExit("Nao achei %s (ajuste config.DADOS)." % os.path.join(dados, "manifest.json"))
    with open(os.path.join(dados, "manifest.json"), encoding="utf-8") as f:
        man = json.load(f)
    chuva = {r["data"]: r.get("chuva_total_mm_media", 0.0) for r in man["datas"]}
    itens = []
    for cam in sorted(glob.glob(os.path.join(dados, "patches", "*.npz"))):
        b = os.path.basename(cam)[:-4]
        try:
            data, sy, sx = b.split("_")
            itens.append((cam, data, int(sy[1:]), int(sx[1:]), chuva.get(data, 0.0)))
        except ValueError:
            continue
    if not itens:
        raise SystemExit("Nenhum patch em %s." % os.path.join(dados, "patches"))
    return man, itens


def abrir_npz(caminho, chaves, tentativas=3):
    """Le as chaves pedidas de um .npz, tentando de novo se a leitura falhar
    (ex.: 'Bad CRC-32' passageiro)."""
    erro = None
    for t in range(tentativas):
        try:
            with np.load(caminho, allow_pickle=False) as d:
                return {k: np.asarray(d[k]) for k in chaves if k in d.files}
        except (zipfile.BadZipFile, ValueError, OSError, EOFError) as e:
            erro = e
            time.sleep(0.5 * (t + 1))
    raise PatchCorrompido("%s (%s)" % (caminho, erro))


def ler_patch(caminho, idx):
    """Devolve X (canais idx), y e valido; y e valido com 1 canal."""
    d = abrir_npz(caminho, ("X", "y", "valido"))
    X = d["X"].astype("float32")[idx]
    y = d["y"].astype("float32")[None]
    v = d["valido"].astype("float32")[None] if "valido" in d else np.ones_like(y)
    return X, y, v


def alvos_e_validos(arquivos):
    """Listas com a mancha real (H, W) e a mascara valida (H, W) de cada patch."""
    alvos, validos = [], []
    for cam in arquivos:
        _, y, v = ler_patch(cam, [0])
        alvos.append(y[0]); validos.append(v[0])
    return alvos, validos


def posicao_do_nome(caminho):
    """(y0, x0) do canto do patch, lidos do nome AAAAMMDD_yNNNN_xNNNN.npz."""
    partes = os.path.basename(caminho)[:-4].split("_")
    return int(partes[-2][1:]), int(partes[-1][1:])


def data_do_nome(caminho):
    return os.path.basename(caminho).split("_")[0]


def data_br(d):
    return "%s/%s/%s" % (d[6:8], d[4:6], d[0:4])


def aumentar(X, y, v, rng):
    """Rotacao de 0/90/180/270 graus e espelhamento com 50% de chance (X, y e v juntos)."""
    k = rng.randint(4)
    X, y, v = (np.rot90(a, k, axes=(1, 2)) for a in (X, y, v))
    if rng.rand() < 0.5:
        X, y, v = (a[:, :, ::-1] for a in (X, y, v))
    return tuple(np.ascontiguousarray(a) for a in (X, y, v))


class LoteDePatches(object):
    """Gera lotes de patches (sem DataLoader, para funcionar igual em qualquer sistema).

    No treino, sorteia com reposicao e com uma fracao fixa de patches com agua
    (frac_molhado). Na avaliacao, percorre os arquivos em ordem."""

    def __init__(self, arquivos, idx, treino, aumento=False, frac_molhado=0.7, semente=None):
        self.arquivos = list(arquivos)
        self.idx = list(idx)
        self.treino = treino
        self.aumento = aumento
        self.frac_molhado = frac_molhado
        self.rng = np.random.RandomState(config.SEMENTE if semente is None else semente)
        self.molhados, self.secos = [], []
        self.corrompidos = []
        if treino:
            for a in self.arquivos:
                try:
                    molhado = int(abrir_npz(a, ("y",))["y"].sum()) >= config.MIN_PIXELS_MOLHADO
                except PatchCorrompido as e:
                    self._avisar(e)
                    continue
                (self.molhados if molhado else self.secos).append(a)

    def _avisar(self, erro):
        self.corrompidos.append(str(erro))
        print("    AVISO: patch corrompido, fica fora do treino: %s" % erro)
        print("           Rode o verificar_patches.py para conferir.")

    def lotes(self, tamanho, por_epoca=None):
        if self.treino:
            n = por_epoca or max(1, len(self.arquivos) // tamanho)
            for _ in range(n):
                esc = []
                for _ in range(tamanho):
                    quer_molhado = self.rng.rand() < self.frac_molhado
                    fonte = self.molhados if (quer_molhado and self.molhados) else (self.secos or self.molhados)
                    esc.append(fonte[self.rng.randint(len(fonte))])
                yield self._montar(esc)
        else:
            for i in range(0, len(self.arquivos), tamanho):
                yield self._montar(self.arquivos[i:i + tamanho])

    def _montar(self, caminhos):
        Xs, ys, vs = [], [], []
        for c in caminhos:
            while True:
                try:
                    X, y, v = ler_patch(c, self.idx)
                    break
                except PatchCorrompido as e:
                    if not self.treino:
                        raise          # na avaliacao, pular um patch mudaria o resultado
                    self._avisar(e)    # no treino, troca por outro patch do mesmo tipo
                    lista = self.molhados if c in self.molhados else self.secos
                    lista.remove(c)
                    fonte = lista or self.molhados or self.secos
                    c = fonte[self.rng.randint(len(fonte))]
            if self.treino and self.aumento:
                X, y, v = aumentar(X, y, v, self.rng)
            Xs.append(X); ys.append(y); vs.append(v)
        return np.stack(Xs), np.stack(ys), np.stack(vs)


def receita(experimento):
    """Configuracao completa de um experimento (PADRAO + o que ele muda)."""
    if experimento not in config.EXPERIMENTOS:
        raise SystemExit("Experimento desconhecido: %r (veja config.EXPERIMENTOS)" % experimento)
    return dict(config.PADRAO, **config.EXPERIMENTOS[experimento])


def canais_do_experimento(canais, cfg):
    """Nomes e indices dos canais usados: todos, menos os grupos em cfg['tirar']."""
    fora = set()
    for g in cfg["tirar"]:
        faltam = [c for c in config.GRUPOS[g] if c not in canais]
        if faltam:
            print("  AVISO: canais do grupo %r que nao existem no manifest: %s" % (g, faltam))
        fora |= set(config.GRUPOS[g])
    usar = [c for c in canais if c not in fora]
    return usar, [canais.index(c) for c in usar]


# =============================================================================
# 2. SPLIT
#    Datas -> treino / val / teste (estratificado pela chuva media).
#    Regioes de 512 x 512 px -> treino / retida (nunca vistas no treino).
#    O cruzamento das duas divisoes da os cinco conjuntos.
# =============================================================================

def regiao_do_patch(y0, x0, lado=None):
    lado = lado or config.LADO_REGIAO_PX
    return (int(y0) // int(lado), int(x0) // int(lado))


def dividir_datas(datas, chuva, frac=None, semente=None):
    """Ordena as datas pela chuva e sorteia os papeis, para que treino, validacao e
    teste tenham eventos de todas as intensidades."""
    frac = frac or config.FRAC_DATAS
    semente = config.SEMENTE if semente is None else semente
    ordem = sorted(range(len(datas)), key=lambda i: (-float(chuva[i]), str(datas[i])))
    n = len(ordem)
    alvo = [int(round(f * n)) for f in frac]
    alvo[0] = n - alvo[1] - alvo[2]
    rng = np.random.RandomState(semente)
    sequencia = ["treino"] * alvo[0] + ["val"] * alvo[1] + ["teste"] * alvo[2]
    rng.shuffle(sequencia)
    return {str(datas[i]): sequencia[pos] for pos, i in enumerate(ordem)}


def dividir_regioes(regioes, frac_retida=None, semente=None):
    frac_retida = config.FRAC_REGIOES_RETIDAS if frac_retida is None else frac_retida
    semente = config.SEMENTE if semente is None else semente
    regs = sorted(set(regioes))
    rng = np.random.RandomState(semente + 1)
    n_ret = max(1, int(round(frac_retida * len(regs))))
    idx = rng.permutation(len(regs))[:n_ret]
    retidas = {regs[i] for i in idx}
    return {r: ("retida" if r in retidas else "treino") for r in regs}


def conjunto_do_patch(papel_data, papel_regiao):
    """(papel da data, papel da regiao) -> conjunto."""
    if papel_regiao == "treino":
        return {"treino": "treino", "val": "validacao", "teste": "teste_tempo"}[papel_data]
    if papel_data == "treino":
        return "teste_espaco"
    if papel_data == "teste":
        return "teste_ambos"
    return None            # validacao x regiao retida: fica de fora


def montar_split(itens):
    """itens -> ({conjunto: [caminhos]}, papel de cada data, papel de cada regiao)."""
    vistos = {}
    for _, d, _, _, mm in itens:
        vistos.setdefault(d, float(mm))
    datas = sorted(vistos)
    papel_d = dividir_datas(datas, [vistos[d] for d in datas])
    regs = [regiao_do_patch(y0, x0) for _, _, y0, x0, _ in itens]
    papel_r = dividir_regioes(regs)
    saida = {}
    for (cam, d, _, _, _), r in zip(itens, regs):
        c = conjunto_do_patch(papel_d[d], papel_r[r])
        if c is not None:
            saida.setdefault(c, []).append(cam)
    return saida, papel_d, papel_r


def conferir_vazamento(saida, itens):
    """Lista de problemas (vazia = split sem vazamento)."""
    meta = {cam: (d, regiao_do_patch(y0, x0)) for cam, d, y0, x0, _ in itens}
    datas_de = lambda c: {meta[k][0] for k in saida.get(c, [])}
    regs_de = lambda c: {meta[k][1] for k in saida.get(c, [])}
    problemas = []
    if datas_de("treino") & (datas_de("teste_tempo") | datas_de("teste_ambos")):
        problemas.append("data de treino em teste temporal")
    if datas_de("validacao") & datas_de("teste_tempo"):
        problemas.append("data de validacao em teste_tempo")
    if regs_de("treino") & (regs_de("teste_espaco") | regs_de("teste_ambos")):
        problemas.append("regiao de treino em teste espacial")
    todos = [k for c in saida.values() for k in c]
    if len(todos) != len(set(todos)):
        problemas.append("o mesmo patch em mais de um conjunto")
    return problemas


def split_em_dict(saida, papel_d, papel_r):
    return {"datas": papel_d, "regioes": {str(k): v for k, v in papel_r.items()},
            "contagem": {k: len(v) for k, v in saida.items()}}


def comparar_split(caminho_ref, atual):
    """Compara o split atual com um split.json de referencia e mostra as diferencas."""
    with open(caminho_ref, encoding="utf-8") as f:
        ref = json.load(f)
    difs = []
    for chave in ("datas", "regioes"):
        a, b = ref.get(chave, {}), atual.get(chave, {})
        difs += ["%s %s: referencia %s, atual %s" % (chave[:-1], k, a.get(k, "-"), b.get(k, "-"))
                 for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)]
    if not difs:
        print("  split identico ao de referencia (%s)" % caminho_ref)
        return True
    print("  AVISO: split DIFERENTE do de referencia (%s):" % caminho_ref)
    for linha in difs[:12]:
        print("    " + linha)
    return False


# =============================================================================
# 3. METRICAS (contadas so nos pixels validos)
# =============================================================================

def contar(pred, alvo, valido):
    """(tp, fp, fn, tn) so onde valido == 1."""
    m = np.asarray(valido) > 0.5
    p = np.asarray(pred)[m].astype(bool)
    a = np.asarray(alvo)[m] > 0.5
    return (int(np.sum(p & a)), int(np.sum(p & ~a)), int(np.sum(~p & a)), int(np.sum(~p & ~a)))


def metricas(tp, fp, fn, tn, eps=1e-9):
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    n = tp + fp + fn + tn
    return {"iou": tp / (tp + fp + fn + eps), "f1": 2 * prec * rec / (prec + rec + eps),
            "precisao": prec, "recall": rec, "prevalencia": (tp + fn) / (n + eps),
            "frac_prevista": (tp + fp) / (n + eps), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# Busca de limiar por histograma: guardamos o histograma das probabilidades dos
# pixels com agua (hp) e secos (hn). Para um limiar t, os verdadeiros positivos
# sao a soma de hp de t para cima, o que permite testar todos os limiares de uma vez.

def novo_histograma():
    return np.zeros(config.N_BINS, dtype="int64"), np.zeros(config.N_BINS, dtype="int64")


def somar_histograma(hp, hn, prob, alvo, valido):
    m = np.asarray(valido) > 0.5
    caixa = np.minimum((np.asarray(prob)[m] * config.N_BINS).astype("int64"), config.N_BINS - 1)
    molhado = np.asarray(alvo)[m] > 0.5
    hp += np.bincount(caixa[molhado], minlength=config.N_BINS)
    hn += np.bincount(caixa[~molhado], minlength=config.N_BINS)


def melhor_limiar_histograma(hp, hn):
    """(limiar, IoU) que maximiza a IoU; limiar t = k / N_BINS, k = 1..N_BINS-1."""
    tp = hp[::-1].cumsum()[::-1]
    fp = hn[::-1].cumsum()[::-1]
    fn = hp.sum() - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    k = 1 + int(np.argmax(iou[1:]))       # k = 0 seria "tudo agua"
    return k / float(config.N_BINS), float(iou[k])


def melhor_limiar(probs, alvos, validos):
    hp, hn = novo_histograma()
    for p, a, v in zip(probs, alvos, validos):
        somar_histograma(hp, hn, p, a, v)
    return melhor_limiar_histograma(hp, hn)[0]


def _iou_de(arr):
    tp, fp, fn = arr[:, 0].sum(), arr[:, 1].sum(), arr[:, 2].sum()
    return tp / max(tp + fp + fn, 1e-9)


def ic_iou(contagens, n=None, semente=None):
    """IC 95% da IoU reamostrando PATCHES (pixels vizinhos nao sao independentes)."""
    n = n or config.N_BOOTSTRAP
    arr = np.asarray(contagens, dtype="float64")
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(config.SEMENTE if semente is None else semente)
    vals = []
    for _ in range(n):
        tp, fp, fn, _tn = arr[rng.randint(0, len(arr), len(arr))].sum(axis=0)
        vals.append(tp / (tp + fp + fn + 1e-9))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def ic_diferenca(cs_a, cs_b, n=None, semente=None):
    """Diferenca de IoU (a - b) e IC 95% pareado: o mesmo sorteio de patches para os dois
    modelos. Se o IC nao cruza o zero, a diferenca nao e explicada pelo acaso da amostra."""
    n = n or config.N_BOOTSTRAP_DIF
    a, b = np.asarray(cs_a, dtype="float64"), np.asarray(cs_b, dtype="float64")
    if len(a) == 0 or len(a) != len(b):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(config.SEMENTE if semente is None else semente)
    difs = []
    for _ in range(n):
        i = rng.randint(0, len(a), len(a))
        difs.append(_iou_de(a[i]) - _iou_de(b[i]))
    return float(_iou_de(a) - _iou_de(b)), float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))


def ic_diferenca_por_data(cs_a, cs_b, datas, n=None, semente=None):
    """Como ic_diferenca, mas sorteando DATAS inteiras (patches da mesma data sao
    correlacionados; este IC e mais exigente)."""
    n = n or config.N_BOOTSTRAP_DIF
    a, b = np.asarray(cs_a, dtype="float64"), np.asarray(cs_b, dtype="float64")
    if len(a) == 0 or len(a) != len(b) or len(datas) != len(a):
        return float("nan"), float("nan"), float("nan")
    grupos = {}
    for i, d in enumerate(datas):
        grupos.setdefault(d, []).append(i)
    blocos = list(grupos.values())
    rng = np.random.RandomState(config.SEMENTE if semente is None else semente)
    difs = []
    for _ in range(n):
        i = np.concatenate([blocos[k] for k in rng.randint(0, len(blocos), len(blocos))])
        difs.append(_iou_de(a[i]) - _iou_de(b[i]))
    return float(_iou_de(a) - _iou_de(b)), float(np.percentile(difs, 2.5)), float(np.percentile(difs, 97.5))


# =============================================================================
# 4. CLIMATOLOGIA (baseline sem rede)
# =============================================================================

def climatologia_na_grade(arquivos, forma):
    """Frequencia de inundacao de cada celula da grade nas datas de TREINO."""
    soma = np.zeros(forma, dtype="float64")
    cont = np.zeros(forma, dtype="float64")
    for cam in arquivos:
        y0, x0 = posicao_do_nome(cam)
        d = abrir_npz(cam, ("y", "valido"))
        y = d["y"].astype("float64")
        v = d["valido"].astype("float64") if "valido" in d else np.ones_like(y)
        h, w = y.shape
        y1, x1 = min(y0 + h, forma[0]), min(x0 + w, forma[1])
        soma[y0:y1, x0:x1] += (y * v)[:y1 - y0, :x1 - x0]
        cont[y0:y1, x0:x1] += v[:y1 - y0, :x1 - x0]
    return np.where(cont > 0, soma / np.maximum(cont, 1e-9), 0.0)


def recortar(grade, y0, x0, lado):
    out = np.zeros((lado, lado), dtype="float32")
    h, w = min(lado, grade.shape[0] - y0), min(lado, grade.shape[1] - x0)
    if h > 0 and w > 0:
        out[:h, :w] = grade[y0:y0 + h, x0:x0 + w]
    return out


# =============================================================================
# 5. REDES
# =============================================================================

def conferir_torch():
    """Para no inicio se o PyTorch nao estiver instalado."""
    try:
        import torch
    except ImportError:
        raise SystemExit("\nO PyTorch nao esta instalado neste Python (%s).\n"
                         "Veja https://pytorch.org/get-started/locally/" % sys.executable)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "nenhuma (treino na CPU, bem mais lento)"
    print("PyTorch %s | GPU: %s" % (torch.__version__, gpu))


def construir_rede(arquitetura, n_canais):
    import torch
    import torch.nn as nn

    if arquitetura == "unet_pequena":
        # U-Net de 4 niveis (~1,95 M parametros com 10 canais)
        def bloco(a, b):
            return nn.Sequential(
                nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
                nn.Conv2d(b, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(inplace=True))

        class UNetPequena(nn.Module):
            def __init__(self, cin, base=32):
                super().__init__()
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
                s1 = self.e1(x)                               # 256 px
                s2 = self.e2(self.pool(s1))                   # 128 px
                s3 = self.e3(self.pool(s2))                   # 64 px
                b = self.e4(self.pool(s3))                    # gargalo: 32 px
                x = self.d3(torch.cat([self.up(b), s3], 1))   # decoder + skip connections
                x = self.d2(torch.cat([self.up(x), s2], 1))
                x = self.d1(torch.cat([self.up(x), s1], 1))
                return self.saida(x)                          # logits

        return UNetPequena(n_canais)

    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise SystemExit("A arquitetura %r precisa do pacote segmentation-models-pytorch:\n"
                         "    pip install segmentation-models-pytorch" % arquitetura)
    classes = {"smp_unet": smp.Unet, "smp_unetpp": smp.UnetPlusPlus,
               "smp_deeplabv3p": smp.DeepLabV3Plus, "smp_unet_scse": smp.Unet}
    if arquitetura not in classes:
        raise SystemExit("Arquitetura desconhecida: %r" % arquitetura)
    extra = {"decoder_attention_type": "scse"} if arquitetura == "smp_unet_scse" else {}
    try:
        # encoder ResNet-34 com pesos da ImageNet; a 1a camada e adaptada a n_canais
        return classes[arquitetura](encoder_name="resnet34", encoder_weights="imagenet",
                                    in_channels=n_canais, classes=1, **extra)
    except Exception as e:
        print("  AVISO: pesos da ImageNet indisponiveis (%s). Treinando do zero." % e)
        return classes[arquitetura](encoder_name="resnet34", encoder_weights=None,
                                    in_channels=n_canais, classes=1, **extra)


def perda_tversky(logits, y, v, alfa, beta, eps=1e-6):
    """1 - indice de Tversky nos pixels validos (alfa pesa FP, beta pesa FN)."""
    import torch
    p = torch.sigmoid(logits) * v
    a = y * v
    tp = (p * a).sum(dim=(1, 2, 3))
    fp = (p * (1 - a)).sum(dim=(1, 2, 3))
    fn = ((1 - p) * a).sum(dim=(1, 2, 3))
    return (1.0 - (tp + eps) / (tp + alfa * fp + beta * fn + eps)).mean()


def perda_bce_dice(logits, y, v):
    """BCE + Dice (Tversky com alfa = beta = 0,5), nos pixels validos."""
    import torch.nn.functional as F
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    bce = (bce * v).sum() / v.sum().clamp_min(1.0)
    return bce + perda_tversky(logits, y, v, 0.5, 0.5)


def calcular_perda(logits, y, v, cfg):
    if cfg["perda"] == "tversky":
        return perda_tversky(logits, y, v, cfg["alfa"], cfg["beta"])
    return perda_bce_dice(logits, y, v)


def carregar_pesos(caminho, dev):
    import torch
    try:
        return torch.load(caminho, map_location=dev, weights_only=True)
    except TypeError:                        # versoes antigas do torch
        return torch.load(caminho, map_location=dev)


def prever(rede, dev, arquivos, idx, lote):
    """Probabilidades de agua (0 a 1), alvos e mascaras validas de cada patch."""
    import torch
    probs, alvos, validos = [], [], []
    with torch.no_grad():
        for X, y, v in LoteDePatches(arquivos, idx, treino=False).lotes(lote):
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                logits = rede(torch.from_numpy(X).to(dev))
            p = torch.sigmoid(logits.float()).cpu().numpy()
            for i in range(p.shape[0]):
                probs.append(p[i, 0]); alvos.append(y[i, 0]); validos.append(v[i, 0])
    return probs, alvos, validos


def carregar_modelo(experimento, canais, pasta=None):
    """Carrega um modelo treinado. Devolve (prever_janelas, idx, cfg), onde
    prever_janelas recebe um array (N, C, H, W) e devolve as probabilidades (N, H, W)."""
    import torch
    pasta = pasta or config.RESULTADOS
    cfg = receita(experimento)
    _, idx = canais_do_experimento(canais, cfg)
    cam = os.path.join(pasta, "%s.pth" % experimento)
    if not os.path.exists(cam):
        raise SystemExit("Nao achei %s. Treine esse experimento com o treinar_unet.py." % cam)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rede = construir_rede(cfg["arquitetura"], len(idx)).to(dev)
    rede.load_state_dict(carregar_pesos(cam, dev)["modelo"])
    rede.eval()
    lote = cfg["lote"]

    def prever_janelas(Xs):
        saida = []
        with torch.no_grad():
            for i in range(0, len(Xs), lote):
                x = torch.from_numpy(np.ascontiguousarray(Xs[i:i + lote], dtype="float32")).to(dev)
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                    logits = rede(x)
                saida.append(torch.sigmoid(logits.float()).cpu().numpy()[:, 0])
        return np.concatenate(saida) if saida else np.zeros((0,) + tuple(Xs.shape[2:]), dtype="float32")

    return prever_janelas, idx, cfg


def prever_arquivos(prever_janelas, arquivos, idx, bloco=32):
    """Aplica prever_janelas a uma lista de patches, em blocos (lista de (H, W))."""
    saida = []
    for i in range(0, len(arquivos), bloco):
        saida.extend(prever_janelas(np.stack([ler_patch(c, idx)[0] for c in arquivos[i:i + bloco]])))
    return saida


def ler_limiar(experimento, pasta=None):
    """Limiar escolhido na validacao pelo treinar_unet.py (lido do resultados.csv)."""
    import csv
    cam = os.path.join(pasta or config.RESULTADOS, "resultados.csv")
    if os.path.exists(cam):
        with open(cam, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                if r["experimento"] == experimento and r["conjunto"] == "validacao":
                    return float(r["limiar"])
    return None
