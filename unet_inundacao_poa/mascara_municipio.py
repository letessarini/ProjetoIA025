"""
================================================================================
MASCARA DO MUNICIPIO NA GRADE DOS PATCHES (a partir de um shapefile)
================================================================================

Converte o limite municipal (shapefile) numa mascara booleana do tamanho da
grade dos patches (True = pixel cujo centro esta dentro do municipio). A
mascara e usada pelo avaliacao_municipio.py.

  1. Le o shapefile e dissolve as feicoes num unico poligono (continente +
     ilhas). Limites em forma de linha sao fechados em poligono.
  2. Obtem a georreferencia da grade, nesta ordem: GEORREF_MANUAL; a parte
     "grade" do manifest.json (transform, limites ou origem + resolucao); um
     GeoTIFF ou sfincs.inp com exatamente o tamanho da grade (procurado perto
     dos dados se RASTER_REFERENCIA = "auto").
  3. Reprojeta o poligono para o sistema da grade e rasteriza pelo centro do
     pixel (regra par-impar, buracos ficam de fora).
  4. Confere o encaixe: entre as georreferencias candidatas (e as versoes com a
     grade invertida na vertical, comum em grades do SFINCS), escolhe a que poe
     mais area do municipio sobre pixels validos dos patches.
  5. Salva a mascara e uma figura de conferencia (limite sobre o terreno).

SAIDA (em config.RESULTADOS)
  mascara_municipio.npy              mascara booleana do tamanho da grade
  mascara_municipio_info.json        georreferencia usada, areas e coberturas
  mascara_municipio_conferencia.png  limite desenhado sobre o terreno

USO
  python mascara_municipio.py
  python mascara_municipio.py --shapefile caminho/limite.shp
================================================================================
"""

import argparse
import json
import os
import re
import time
import unicodedata
from collections import Counter

import numpy as np

import comum
import config

CRS_SHAPEFILE = None          # so se o .shp nao tiver .prj. Ex.: "EPSG:31982" (SIRGAS 2000 / UTM 22S)
FILTRO_MUNICIPIO = "auto"     # shapefile com varios municipios: ("NM_MUN", "Porto Alegre"); "auto" procura
GEORREF_MANUAL = None         # ex.: {"crs": "EPSG:31982", "transform": [20, 0, 470000, 0, -20, 6690000]}
RASTER_REFERENCIA = "auto"    # GeoTIFF ou sfincs.inp na MESMA grade dos patches; "auto" procura; None desliga
INVERTER_VERTICAL = "auto"    # "auto" testa os dois sentidos; True/False forca
DATA_CONFERENCIA = "auto"     # data usada para desenhar o terreno ("auto" = a com mais patches)


# =============================================================================
# 1. PACOTES
# =============================================================================

def conferir_pacotes():
    faltando = []
    for modulo in ("geopandas", "shapely", "pyproj"):
        try:
            __import__(modulo)
        except ImportError:
            faltando.append(modulo)
    if faltando:
        raise SystemExit("Faltam pacotes: %s. Instale com:\n    pip install %s"
                         % (", ".join(faltando), " ".join(faltando)))


# =============================================================================
# 2. GEOMETRIA DA GRADE (transformacao afim, como no rasterio)
#    x = a*coluna + b*linha + c ;  y = d*coluna + e*linha + f
#    (coluna, linha) contadas a partir do CANTO de cima a esquerda do pixel (0, 0)
# =============================================================================

def para_pixel(t, x, y):
    """Coordenadas do mapa -> (coluna, linha) fracionarias na grade."""
    a, b, c, d, e, f = t
    det = a * e - b * d
    x = np.asarray(x, dtype="float64") - c
    y = np.asarray(y, dtype="float64") - f
    return (e * x - b * y) / det, (-d * x + a * y) / det


def para_mapa(t, col, lin):
    a, b, c, d, e, f = t
    return a * col + b * lin + c, d * col + e * lin + f


def virar_vertical(t, altura):
    """Mesma grade, mas com a linha 0 do outro lado (de cabeca para baixo)."""
    a, b, c, d, e, f = t
    return (a, -b, c + b * altura, d, -e, f + e * altura)


def limites_da_grade(t, forma):
    h, w = forma
    xs, ys = para_mapa(t, np.array([0, w, 0, w]), np.array([0, 0, h, h]))
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def area_do_pixel(t):
    a, b, c, d, e, f = t
    return abs(a * e - b * d)


def rasterizar(aneis, t, forma):
    """True nos pixels cujo CENTRO cai dentro do poligono (mesma regra do rasterio).

    aneis: lista de arrays (n, 2) com os contornos externos E os buracos, em
    coordenadas do mapa. Usa a regra par-impar, entao buracos ficam de fora.
    Faz por linhas de varredura: cada aresta so e testada nas linhas que cruza.
    """
    h, w = forma
    U0, V0, U1, V1 = [], [], [], []
    for anel in aneis:
        anel = np.asarray(anel, dtype="float64")[:, :2]
        if len(anel) < 3:
            continue
        if not np.array_equal(anel[0], anel[-1]):
            anel = np.vstack([anel, anel[:1]])
        u, v = para_pixel(t, anel[:, 0], anel[:, 1])
        U0.append(u[:-1]); V0.append(v[:-1]); U1.append(u[1:]); V1.append(v[1:])
    saida = np.zeros(forma, dtype=bool)
    if not U0:
        return saida
    U0, V0, U1, V1 = (np.concatenate(a) for a in (U0, V0, U1, V1))
    # linhas r cujo centro (r + 0.5) esta em [vmin, vmax): regra meio-aberta, vertices nao contam 2x
    r0 = np.clip(np.ceil(np.minimum(V0, V1) - 0.5), 0, h).astype(np.int64)
    r1 = np.clip(np.ceil(np.maximum(V0, V1) - 0.5), 0, h).astype(np.int64)
    n = np.maximum(r1 - r0, 0)
    if n.sum() == 0:
        return saida
    qual = np.repeat(np.arange(len(n)), n)
    inicio = np.cumsum(n) - n
    linhas = r0[qual] + (np.arange(n.sum()) - np.repeat(inicio, n))
    vc = linhas + 0.5
    uc = U0[qual] + (vc - V0[qual]) * (U1[qual] - U0[qual]) / (V1[qual] - V0[qual])
    ordem = np.lexsort((uc, linhas))
    linhas, uc = linhas[ordem], uc[ordem]
    if len(linhas) % 2 or not np.array_equal(linhas[0::2], linhas[1::2]):
        raise RuntimeError("Contorno aberto: o poligono do municipio nao fecha.")
    ja = np.clip(np.ceil(uc[0::2] - 0.5), 0, w).astype(np.int64)
    jb = np.clip(np.ceil(uc[1::2] - 0.5), 0, w).astype(np.int64)
    marca = np.zeros((h, w + 1), dtype=np.int32)
    np.add.at(marca, (linhas[0::2], ja), 1)
    np.add.at(marca, (linhas[0::2], jb), -1)
    return np.cumsum(marca, axis=1)[:, :w] > 0


# =============================================================================
# 3. ONDE A GRADE FICA NO MAPA (georreferencia)
# =============================================================================

def _norm(txt):
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]", "", txt)


def _numeros(txt):
    return [float(s) for s in re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", str(txt))]


def _pega(dic, nomes):
    """Valor da primeira chave cujo nome normalizado esta em `nomes`."""
    if not isinstance(dic, dict):
        return None
    for k, v in dic.items():
        if _norm(k) in nomes and v is not None:
            return v
    return None


def transform_de_valor(v):
    """Aceita lista de 6 (rasterio ou GDAL), lista de 9 (matriz 3x3), dict a..f ou texto."""
    if isinstance(v, dict):
        if all(k in v for k in "abcdef"):
            return tuple(float(v[k]) for k in "abcdef")
        return None
    nums = _numeros(v) if isinstance(v, str) else [float(x) for x in np.ravel(v)]
    if len(nums) == 9:
        nums = nums[:6]
    if len(nums) != 6:
        return None
    a = nums
    estilo_rasterio = abs(a[2]) > 10 * abs(a[0]) and abs(a[5]) > 10 * abs(a[4]) and a[0] != 0 and a[4] != 0
    estilo_gdal = abs(a[0]) > 10 * abs(a[1]) and abs(a[3]) > 10 * abs(a[5]) and a[1] != 0 and a[5] != 0
    if estilo_rasterio:
        return tuple(a)
    if estilo_gdal:                       # GDAL: (c, a, b, f, d, e)
        return (a[1], a[2], a[0], a[4], a[5], a[3])
    return None


def transforms_de_limites(v, forma):
    """Limites (4 numeros) -> transformacoes possiveis; fica com as de pixel quadrado."""
    h, w = forma
    if isinstance(v, dict):
        grupos = [("left", "bottom", "right", "top"), ("xmin", "ymin", "xmax", "ymax"),
                  ("oeste", "sul", "leste", "norte"), ("minx", "miny", "maxx", "maxy")]
        nv = {_norm(k): x for k, x in v.items()}
        for g in grupos:
            if all(k in nv for k in g):
                v = [nv[k] for k in g]
                break
        else:
            return []
    nums = [float(x) for x in np.ravel(v)] if not isinstance(v, str) else _numeros(v)
    if len(nums) != 4:
        return []
    saida = []
    for nome, (x0, y0, x1, y1) in (("xmin,ymin,xmax,ymax", nums),
                                   ("xmin,xmax,ymin,ymax", (nums[0], nums[2], nums[1], nums[3]))):
        px, py = (x1 - x0) / w, (y1 - y0) / h
        if px > 0 and py > 0 and abs(px / py - 1) < 0.2:
            saida.append((nome, (px, 0.0, x0, 0.0, -py, y1)))
    return saida


def crs_de_valor(v):
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return "EPSG:%d" % int(v)
    if isinstance(v, dict):
        return crs_de_valor(_pega(v, {"epsg", "init", "wkt", "crs"}))
    v = str(v).strip()
    if re.fullmatch(r"\d{4,6}", v):
        return "EPSG:%s" % v
    return v or None


def caminhos_no_manifest(obj, achados=None):
    """Procura no manifest textos que parecem caminhos de GeoTIFF ou sfincs.inp."""
    achados = [] if achados is None else achados
    if isinstance(obj, dict):
        for v in obj.values():
            caminhos_no_manifest(v, achados)
    elif isinstance(obj, (list, tuple)):
        for v in obj[:5000]:
            caminhos_no_manifest(v, achados)
    elif isinstance(obj, str) and re.search(r"\.(tif|tiff|inp)$", obj.strip(), re.I):
        achados.append(obj.strip())
    return achados


def georref_do_manifest(man, forma):
    """Candidatos (nome, transform, crs) a partir do que estiver escrito no manifest."""
    lugares = [man.get("grade", {})]
    lugares += [v for k, v in man.items() if isinstance(v, dict) and k != "grade"]
    lugares.append({k: v for k, v in man.items() if k not in ("datas", "canais") and not isinstance(v, dict)})
    crs = None
    for lugar in lugares:
        crs = crs or crs_de_valor(_pega(lugar, {"crs", "epsg", "srs", "projecao", "sistema", "wkt", "proj",
                                                 "sistemadecoordenadas", "spatialref", "crswkt"}))
    cands = []
    for lugar in lugares:
        try:
            cands += _candidatos_de_um_lugar(lugar, forma, crs)
        except (ValueError, TypeError, IndexError, KeyError):
            continue
    return cands, crs


def _dois_numeros(v):
    if isinstance(v, dict):
        v = [_pega(v, {"x", "dx", "largura"}), _pega(v, {"y", "dy", "altura"})]
    nums = _numeros(v) if isinstance(v, str) else [float(x) for x in np.ravel(v)]
    return nums[0], nums[-1]


def _candidatos_de_um_lugar(lugar, forma, crs):
    h = forma[0]
    cands = []
    v = _pega(lugar, {"transform", "transformacao", "transformada", "affine", "afim", "geotransform", "gt",
                      "transformafim"})
    t = transform_de_valor(v) if v is not None else None
    if t:
        cands.append(("manifest: transform", t, crs))
    v = _pega(lugar, {"bounds", "limites", "extent", "extensao", "bbox", "envelope", "totalbounds", "limitesxy"})
    for nome, t in transforms_de_limites(v if v is not None else lugar, forma):
        cands.append(("manifest: limites (%s)" % nome, t, crs))
    res = _pega(lugar, {"res", "resolucao", "resolucaom", "pixel", "pixelm", "tamanhopixel", "tamanhodopixel",
                        "tamanhopixelm", "dx", "celula", "cellsize"})
    if res is None:
        return cands
    rx, ry = (abs(n) for n in _dois_numeros(res))
    x = _pega(lugar, {"x0", "xmin", "xorigem", "origemx", "left", "oeste", "ulx", "xul", "xesq"})
    y_topo = _pega(lugar, {"ymax", "top", "norte", "uly", "yul", "ytopo"})
    y_base = _pega(lugar, {"ymin", "bottom", "sul", "ybase"})
    y_amb = _pega(lugar, {"y0", "yorigem", "origemy"})
    origem = _pega(lugar, {"origem", "origin", "cantosuperioresquerdo"})
    if origem is not None:
        x, y_amb = _dois_numeros(origem)
    if x is None:
        return cands
    x = float(x)
    if y_topo is not None:
        cands.append(("manifest: origem em cima", (rx, 0.0, x, 0.0, -ry, float(y_topo)), crs))
    if y_base is not None:
        cands.append(("manifest: origem embaixo", (rx, 0.0, x, 0.0, -ry, float(y_base) + h * ry), crs))
    if y_amb is not None:   # nao da para saber se y0 e o canto de cima ou de baixo: testa os dois
        cands.append(("manifest: y0 como canto de cima", (rx, 0.0, x, 0.0, -ry, float(y_amb)), crs))
        cands.append(("manifest: y0 como canto de baixo", (rx, 0.0, x, 0.0, -ry, float(y_amb) + h * ry), crs))
    return cands


def ler_cabecalho_tif(caminho):
    """(forma, transform, crs) de um GeoTIFF sem ler os pixels. rasterio se tiver; senao Pillow."""
    try:
        import rasterio
        with rasterio.open(caminho) as r:
            crs = r.crs.to_string() if r.crs else None
            t = tuple(float(v) for v in tuple(r.transform)[:6])
            return (r.height, r.width), (None if t == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0) else t), crs
    except ImportError:
        pass
    except Exception:
        return None
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(caminho) as im:
            w, h = im.size
            tags = dict(im.tag_v2) if hasattr(im, "tag_v2") else {}
    except Exception:
        return None
    escala, ponto, matriz, chaves = (tags.get(k) for k in (33550, 33922, 34264, 34735))
    if matriz and len(matriz) >= 8:
        t = (matriz[0], matriz[1], matriz[3], matriz[4], matriz[5], matriz[7])
    elif escala and ponto and len(ponto) >= 6:
        i, j, _, X, Y, _ = ponto[:6]
        sx, sy = escala[0], escala[1]
        t = (sx, 0.0, X - i * sx, 0.0, -sy, Y + j * sy)
    else:
        return (h, w), None, None
    crs, centro = None, False
    if chaves and len(chaves) >= 4:
        k = list(chaves)
        for n in range(int(k[3])):
            kid, loc, _, val = k[4 + 4 * n: 8 + 4 * n]
            if loc != 0:
                continue
            if kid == 3072 and val not in (0, 32767):
                crs = "EPSG:%d" % val
            elif kid == 2048 and val not in (0, 32767) and crs is None:
                crs = "EPSG:%d" % val
            elif kid == 1025 and val == 2:          # PixelIsPoint: o ponto e o centro do pixel
                centro = True
    t = tuple(float(x) for x in t)
    if centro:
        a, b, c, d, e, f = t
        t = (a, b, c - (a + b) / 2, d, e, f - (d + e) / 2)
    return (h, w), t, crs


def ler_sfincs_inp(caminho):
    """(forma, transform com a linha 0 EM CIMA, crs) de um sfincs.inp."""
    p = {}
    with open(caminho, encoding="utf-8", errors="ignore") as f:
        for linha in f:
            if "=" in linha:
                k, v = linha.split("=", 1)
                p[k.strip().lower()] = v.strip()
    try:
        x0, y0 = float(p["x0"]), float(p["y0"])
        dx, dy = float(p["dx"]), float(p["dy"])
        mmax, nmax = int(float(p["mmax"])), int(float(p["nmax"]))
    except (KeyError, ValueError):
        return None
    th = np.radians(float(p.get("rotation", 0) or 0))
    # SFINCS: (x0, y0) e o canto de BAIXO a esquerda e a linha 0 da matriz e a de baixo
    t_baixo = (dx * np.cos(th), -dy * np.sin(th), x0, dx * np.sin(th), dy * np.cos(th), y0)
    crs = crs_de_valor(p.get("epsg")) if p.get("epsg") else None
    return (nmax, mmax), virar_vertical(t_baixo, nmax), crs


def procurar_referencias(forma, raizes, prof_max=4, limite_s=60, max_achados=6):
    """Procura GeoTIFF/sfincs.inp com exatamente o tamanho da grade."""
    pular = {"windows", "programfiles", "programfilesx86", "programdata", "appdata", "recyclebin",
             "systemvolumeinformation", "anaconda3", "miniconda3", "envs", "sitepackages", "nodemodules",
             "git", "patches", "pycache", "onedrive", "library", "pkgs"}
    achados, quase, vistos = [], [], set()
    t0 = time.time()
    for raiz in raizes:
        if not raiz or not os.path.isdir(raiz):
            continue
        base = os.path.realpath(raiz).rstrip("\\/").count(os.sep)
        for pasta, subpastas, arquivos in os.walk(raiz):
            real = os.path.realpath(pasta)
            if real in vistos or time.time() - t0 > limite_s or len(achados) >= max_achados:
                subpastas[:] = []
                continue
            vistos.add(real)
            subpastas[:] = [s for s in subpastas if _norm(s) not in pular and not s.startswith(("$", "."))
                            and real.rstrip("\\/").count(os.sep) - base < prof_max]
            for a in arquivos:
                cam = os.path.join(pasta, a)
                if re.search(r"\.tiff?$", a, re.I):
                    r = ler_cabecalho_tif(cam)
                elif a.lower() == "sfincs.inp":
                    r = ler_sfincs_inp(cam)
                else:
                    continue
                if not r or r[1] is None:
                    continue
                (achados if tuple(r[0]) == tuple(forma) else quase).append((cam,) + tuple(r))
    return achados, quase


def juntar_candidatos(man, forma):
    """Lista de candidatos (nome, transform, crs, confianca), do mais confiavel ao menos."""
    cands = []
    if GEORREF_MANUAL:
        t = transform_de_valor(GEORREF_MANUAL["transform"])
        if t is None:
            raise SystemExit("GEORREF_MANUAL['transform'] nao tem 6 numeros validos.")
        return [dict(nome="GEORREF_MANUAL", t=t, crs=crs_de_valor(GEORREF_MANUAL.get("crs")), confianca=10)]

    do_man, crs_man = georref_do_manifest(man, forma)
    for i, (nome, t, crs) in enumerate(do_man):
        cands.append(dict(nome=nome, t=t, crs=crs, confianca=8 - 0.1 * i))

    arquivos = [c for c in caminhos_no_manifest(man) if os.path.isfile(c)]
    if isinstance(RASTER_REFERENCIA, str) and RASTER_REFERENCIA != "auto":
        arquivos.insert(0, RASTER_REFERENCIA)
    quase = []
    for cam in arquivos:
        r = ler_sfincs_inp(cam) if cam.lower().endswith(".inp") else ler_cabecalho_tif(cam)
        if not r or r[1] is None:
            print("  (sem georreferencia em %s)" % cam)
            continue
        if tuple(r[0]) != tuple(forma):
            quase.append((cam,) + tuple(r))
            continue
        cands.append(dict(nome=os.path.basename(cam), t=r[1], crs=r[2] or crs_man, confianca=9, arquivo=cam))
    if RASTER_REFERENCIA == "auto" and not cands:
        print("Procurando um GeoTIFF/sfincs.inp com o tamanho da grade (%d x %d)..." % forma)
        raizes = [config.DADOS, os.path.dirname(os.path.abspath(config.DADOS)), config.RAIZ]
        achados, q = procurar_referencias(forma, raizes)
        quase += q
        for cam, f, t, crs in achados:
            print("  achei %s" % cam)
            cands.append(dict(nome=os.path.basename(cam), t=t, crs=crs or crs_man, confianca=7, arquivo=cam))
    if not cands and quase:
        print("  Arquivos georreferenciados com OUTRO tamanho (nao servem direto):")
        for cam, f, t, crs in quase[:8]:
            print("    %s  %s" % (cam, f))
    cands.sort(key=lambda c: -c["confianca"])
    # o mesmo transform vindo de lugares diferentes so precisa ser testado uma vez
    unicos = []
    for c in cands:
        if not any(np.allclose(c["t"], u["t"], rtol=0, atol=1e-6) and c["crs"] == u["crs"] for u in unicos):
            unicos.append(c)
    return unicos


# =============================================================================
# 4. O MUNICIPIO (shapefile)
# =============================================================================

def aneis_da_geometria(geom):
    """Todos os contornos (externos e buracos) de um Polygon/MultiPolygon/colecao."""
    aneis = []
    if geom is None or geom.is_empty:
        return aneis
    if geom.geom_type == "Polygon":
        aneis.append(np.asarray(geom.exterior.coords)[:, :2])
        aneis += [np.asarray(i.coords)[:, :2] for i in geom.interiors]
    elif hasattr(geom, "geoms"):
        for g in geom.geoms:
            aneis += aneis_da_geometria(g)
    return aneis


class Municipio(object):
    """O limite do municipio, com conversao (e cache) para o sistema de cada grade."""

    def __init__(self, gpd, geom, crs, origem=None):
        self.gpd, self.geom, self.crs, self.origem, self._cache = gpd, geom, crs, origem, {}

    def na_crs(self, crs):
        if crs is None or self.crs is None:
            return self.geom
        chave = str(crs)
        if chave not in self._cache:
            self._cache[chave] = self.gpd.GeoSeries([self.geom], crs=self.crs).to_crs(crs).iloc[0]
        return self._cache[chave]

    def aneis(self, crs):
        return aneis_da_geometria(self.na_crs(crs))

    def area(self, crs):
        return float(self.na_crs(crs).area)

    def limites(self, crs):
        return tuple(float(v) for v in self.na_crs(crs).bounds)

    def area_km2(self):
        try:
            return float(self.gpd.GeoSeries([self.geom], crs=self.crs).to_crs("EPSG:31982").area.iloc[0]) / 1e6
        except Exception:
            return float("nan")


def adivinhar_crs(limites):
    x0, y0, x1, y1 = limites
    if -180 <= x0 <= 180 and -90 <= y0 <= 90 and -180 <= x1 <= 180:
        return "EPSG:4674"           # SIRGAS 2000 em graus
    if 1e5 < x0 < 9e5 and 6.0e6 < y0 < 7.5e6:
        return "EPSG:31982"          # SIRGAS 2000 / UTM 22S (o mais comum em Porto Alegre)
    return None


def filtrar_municipio(gdf):
    if FILTRO_MUNICIPIO is None or len(gdf) <= 1:
        return gdf
    if isinstance(FILTRO_MUNICIPIO, (tuple, list)):
        col, val = FILTRO_MUNICIPIO
        sel = gdf[gdf[col].astype(str).map(_norm) == _norm(val)]
        if sel.empty:
            raise SystemExit("Nenhuma feicao com %s = %s no shapefile." % (col, val))
        return sel
    cod = str(config.MUNICIPIO["codigo_ibge"])
    alvos = {_norm(config.MUNICIPIO["nome"]), cod, cod[:6]}
    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue
        vals = gdf[col].astype(str).map(lambda s: _norm(re.sub(r"\.0$", "", s)))
        hit = vals.isin(alvos)
        if hit.any() and not hit.all():
            print("  O shapefile tem %d feicoes; usando as %d com %s = %s."
                  % (len(gdf), hit.sum(), col, config.MUNICIPIO["nome"]))
            return gdf[hit.values]
    return gdf


def ler_municipio(gpd, shapefile):
    from shapely.ops import polygonize, unary_union
    try:
        from shapely.validation import make_valid
    except ImportError:
        make_valid = lambda g: g.buffer(0)
    if not os.path.isfile(shapefile):
        raise SystemExit("Nao achei o shapefile: %s (ajuste config.SHAPEFILE_MUNICIPIO)" % shapefile)
    gdf = gpd.read_file(shapefile)
    print("Shapefile: %d feicao(oes), tipos %s" % (len(gdf), sorted(set(gdf.geom_type.dropna()))))
    gdf = filtrar_municipio(gdf)
    crs = gdf.crs.to_string() if gdf.crs is not None else None
    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        raise SystemExit("O shapefile nao tem geometrias.")
    if crs is None:
        crs = CRS_SHAPEFILE or adivinhar_crs(unary_union(geoms).bounds)
        print("  AVISO: o shapefile nao tem .prj. Usando %s %s" % (
            crs, "(definido em CRS_SHAPEFILE)" if CRS_SHAPEFILE else "(adivinhado pelas coordenadas; confira)"))
    poligonos = [make_valid(g) for g in geoms if g.geom_type in ("Polygon", "MultiPolygon")]
    linhas = [g for g in geoms if "LineString" in g.geom_type]
    if poligonos:
        geom = unary_union(poligonos)
    elif linhas:
        print("  O limite veio como LINHA; fechando em poligono...")
        faces = list(polygonize(unary_union(linhas)))
        if not faces:
            raise SystemExit("A linha do limite nao fecha um poligono. Use o poligono da malha municipal do\n"
                             "IBGE com FILTRO_MUNICIPIO = ('CD_MUN', '%s')." % config.MUNICIPIO["codigo_ibge"])
        geom = unary_union(faces)
    else:
        raise SystemExit("O shapefile so tem pontos; preciso de poligono ou linha do limite.")
    geom = make_valid(geom)
    print("  CRS do shapefile: %s | limites: %s" % (crs, ", ".join("%.1f" % v for v in geom.bounds)))
    return Municipio(gpd, geom, crs, shapefile)


# =============================================================================
# 5. CONFERENCIA (dominio valido de uma data) E ESCOLHA
# =============================================================================

def dominio_de_uma_data(man, itens, forma, lado):
    """Mosaico de uma data: area valida e um canal de terreno para desenhar o fundo."""
    cont = Counter(d for _, d, _, _, _ in itens)
    data = max(cont, key=lambda d: (cont[d], d)) if DATA_CONFERENCIA == "auto" else str(DATA_CONFERENCIA)
    canais = list(man["canais"])
    nome = next((c for c in ("cota_m", "hand_m", "decl_pct") if c in canais), canais[0])
    k = canais.index(nome)
    valido = np.zeros(forma, dtype=bool)
    fundo = np.full(forma, np.nan, dtype="float32")
    for cam, d, y0, x0, _ in itens:
        if d != data:
            continue
        X, _, v = comum.ler_patch(cam, [k])
        h, w = min(lado, forma[0] - y0), min(lado, forma[1] - x0)
        valido[y0:y0 + h, x0:x0 + w] = v[0, :h, :w] > 0.5
        fundo[y0:y0 + h, x0:x0 + w] = X[0, :h, :w]
    return data, nome, valido, fundo


def avaliar(cand, municipio, forma, valido):
    mascara = rasterizar(municipio.aneis(cand["crs"]), cand["t"], forma)
    px = area_do_pixel(cand["t"])
    area_poly = municipio.area(cand["crs"])
    n = int(mascara.sum())
    sobre = int((mascara & valido).sum())
    return dict(cand, mascara=mascara, pixels=n, pixels_validos=sobre,
                cobertura=n * px / area_poly if area_poly > 0 else float("nan"),
                frac_valida=sobre / n if n else 0.0)


def escolher(cands, municipio, forma, valido):
    testes = []
    for c in cands:
        orientacoes = {"auto": (False, True), True: (True,), False: (False,)}[INVERTER_VERTICAL]
        for inv in orientacoes:
            c2 = dict(c, invertida=inv, t=virar_vertical(c["t"], forma[0]) if inv else c["t"],
                      nome=c["nome"] + (" (invertida)" if inv else ""),
                      confianca=c["confianca"] - (0.5 if inv and INVERTER_VERTICAL == "auto" else 0))
            testes.append(avaliar(c2, municipio, forma, valido))
    print("\nCandidatos de georreferencia (quanto do municipio cai sobre pixels validos):")
    for r in testes:
        print("  %-48s municipio na grade %5.1f%% | sobre area valida %5.1f%% (%d px)"
              % (r["nome"][:48], 100 * min(r["cobertura"], 9.99), 100 * r["frac_valida"], r["pixels_validos"]))
    confiavel = max(testes, key=lambda r: r["confianca"])
    melhor = max(testes, key=lambda r: (r["pixels_validos"], r["confianca"]))
    if melhor is not confiavel and melhor["pixels_validos"] > 1.10 * confiavel["pixels_validos"] + 50:
        print("  -> '%s' encaixa bem melhor que '%s'; usando ela." % (melhor["nome"], confiavel["nome"]))
        escolhido = melhor
    else:
        escolhido = confiavel
    outros = [r for r in testes if r is not escolhido]
    alternativa = max(outros, key=lambda r: r["pixels_validos"]) if outros else None
    return escolhido, alternativa


def figura_conferencia(caminho, fundo, valido, data, nome_fundo, paineis, municipio):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    vals = fundo[valido & np.isfinite(fundo)]
    lo, hi = (np.percentile(vals, [2, 98]) if vals.size else (0, 1))
    fig, eixos = plt.subplots(1, len(paineis), figsize=(7.2 * len(paineis), 7.6), squeeze=False)
    for ax, r in zip(eixos[0], paineis):
        ax.imshow(np.where(valido, 0, 1), cmap=ListedColormap(["#FFFFFF", "#CFE3EC"]), vmin=0, vmax=1,
                  interpolation="nearest")
        ax.imshow(np.ma.masked_where(~valido, fundo), cmap="gist_earth", vmin=lo, vmax=hi, alpha=0.85,
                  interpolation="nearest")
        ax.imshow(np.ma.masked_where(~r["mascara"], r["mascara"].astype("uint8")), cmap=ListedColormap(["#C0392B"]),
                  vmin=0, vmax=1, alpha=0.22, interpolation="nearest")
        for anel in municipio.aneis(r["crs"]):
            u, v = para_pixel(r["t"], anel[:, 0], anel[:, 1])
            ax.plot(u - 0.5, v - 0.5, color="#C0392B", lw=0.9)
        ax.set_xlim(-0.5, valido.shape[1] - 0.5); ax.set_ylim(valido.shape[0] - 0.5, -0.5)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("%s%s\nmunicípio sobre área válida: %.0f%%" % (
            "USADA: " if r.get("usada") else "alternativa: ", r["nome"], 100 * r["frac_valida"]), fontsize=10)
    fig.suptitle("Encaixe do limite municipal na grade (%s, fundo: %s)\n"
                 "A linha vermelha deve seguir o terreno (margens e ilhas)."
                 % (comum.data_br(data), nome_fundo), fontsize=11)
    fig.legend(handles=[Line2D([], [], color="#C0392B", lw=1.2, label="limite do município"),
                        Patch(color="#CFE3EC", label="fora do domínio dos patches")],
               loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(caminho, dpi=170)
    plt.close(fig)


# =============================================================================
# 6. PROGRAMA PRINCIPAL
# =============================================================================

def gerar_mascara(municipio, man, itens, saida):
    forma = tuple(man["grade"]["forma"])
    lado = int(man["patch"]["tamanho"])
    print("\nGrade dos patches: %d x %d px" % forma)
    cands = juntar_candidatos(man, forma)
    if not cands:
        grade = json.dumps(man.get("grade", {}), ensure_ascii=False)
        raise SystemExit(
            "\nNAO ACHEI A GEORREFERENCIA DA GRADE.\n"
            "Parte 'grade' do manifest.json:\n  %s\n\n"
            "Informe a georreferencia em GEORREF_MANUAL, ou ponha em RASTER_REFERENCIA o caminho de\n"
            "um GeoTIFF de uma das camadas de entrada (cota, HAND...) com %d x %d pixels."
            % (grade[:1500], forma[0], forma[1]))

    data, nome_fundo, valido, fundo = dominio_de_uma_data(man, itens, forma, lado)
    print("Conferindo o encaixe na data %s (%.0f%% da grade e area valida)" % (data, 100 * valido.mean()))
    for c in cands:
        if c["crs"] is None:
            print("  AVISO: '%s' nao diz o sistema de coordenadas; supondo o mesmo do shapefile (%s)."
                  % (c["nome"], municipio.crs))
        tam = (abs(c["t"][0]), abs(c["t"][4]))
        if c["crs"] is not None and not ("4326" in str(c["crs"]) or "4674" in str(c["crs"])) \
                and not (0.5 < tam[0] / config.TAMANHO_PIXEL_M < 2):
            print("  AVISO: '%s' tem pixel de %.4g x %.4g (esperava ~%g m)."
                  % (c["nome"], tam[0], tam[1], config.TAMANHO_PIXEL_M))

    escolhido, alternativa = escolher(cands, municipio, forma, valido)
    if escolhido["pixels"] == 0:
        lg = limites_da_grade(escolhido["t"], forma)
        lm = municipio.limites(escolhido["crs"])
        raise SystemExit(
            "\nO municipio caiu FORA da grade. Provavelmente os sistemas de coordenadas nao batem.\n"
            "  grade     (%s): x %.1f a %.1f | y %.1f a %.1f\n"
            "  municipio (mesmo sistema): x %.1f a %.1f | y %.1f a %.1f\n"
            "Confira CRS_SHAPEFILE e a georreferencia da grade." % (escolhido["crs"] or municipio.crs, lg[0], lg[2],
                                                                    lg[1], lg[3], lm[0], lm[2], lm[1], lm[3]))

    mascara = escolhido["mascara"]
    np.save(saida, mascara)
    px_km2 = area_do_pixel(escolhido["t"]) / 1e6
    area_shp = municipio.area_km2()
    info = dict(shapefile=municipio.origem, crs_shapefile=municipio.crs, crs_grade=escolhido["crs"] or municipio.crs,
                fonte_georreferencia=escolhido["nome"], invertida=bool(escolhido["invertida"]),
                transform=[float(v) for v in escolhido["t"]], forma=list(forma),
                pixels_no_municipio=int(mascara.sum()), area_mascara_km2=float(mascara.sum() * px_km2),
                area_poligono_km2=area_shp, fracao_do_municipio_na_grade=float(min(escolhido["cobertura"], 1.0)),
                fracao_do_municipio_sobre_area_valida=float(escolhido["frac_valida"]),
                pixels_validos_no_municipio=int(escolhido["pixels_validos"]), data_conferencia=data)
    with open(os.path.splitext(saida)[0] + "_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    escolhido["usada"] = True
    paineis = [escolhido] + ([alternativa] if alternativa is not None else [])
    figura = os.path.splitext(saida)[0] + "_conferencia.png"
    figura_conferencia(figura, fundo, valido, data, nome_fundo, paineis, municipio)

    print("\n" + "=" * 78)
    print("MASCARA DO MUNICIPIO")
    print("=" * 78)
    print("  georreferencia usada : %s" % escolhido["nome"])
    print("  sistema da grade     : %s" % (escolhido["crs"] or municipio.crs))
    area_ref = config.MUNICIPIO["area_km2"]
    print("  area do poligono     : %.1f km2 (area oficial: %.1f km2)" % (area_shp, area_ref))
    print("  area na mascara      : %.1f km2 (%d pixels)" % (info["area_mascara_km2"], info["pixels_no_municipio"]))
    print("  municipio dentro da grade        : %.0f%%" % (100 * info["fracao_do_municipio_na_grade"]))
    print("  municipio sobre pixels validos   : %.0f%%  (pixels usados na avaliacao)"
          % (100 * info["fracao_do_municipio_sobre_area_valida"]))
    if info["fracao_do_municipio_na_grade"] < 0.5:
        print("  AVISO: menos da metade do municipio cai na grade. Confira a figura.")
    if area_shp == area_shp and not (0.6 < area_shp / area_ref < 1.4):
        print("  AVISO: a area do poligono difere muito da oficial; confira se o shapefile e o limite municipal.")
    print("\n  Salvo: %s" % saida)
    print("  Figura de conferencia: %s" % figura)
    return info


def main(shapefile=None):
    conferir_pacotes()
    import geopandas as gpd
    saida = os.path.join(config.RESULTADOS, "mascara_municipio.npy")
    os.makedirs(config.RESULTADOS, exist_ok=True)
    municipio = ler_municipio(gpd, shapefile or config.SHAPEFILE_MUNICIPIO)
    man, itens = comum.carregar_itens()
    return gerar_mascara(municipio, man, itens, saida)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Mascara do municipio na grade dos patches.")
    ap.add_argument("--shapefile", help="padrao: config.SHAPEFILE_MUNICIPIO")
    args = ap.parse_args()
    try:
        main(args.shapefile)
    except comum.PatchCorrompido as e:
        raise SystemExit("\nPatch corrompido: %s\nRode o verificar_patches.py." % e)
