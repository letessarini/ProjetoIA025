"""
Configuracao compartilhada por todos os scripts.

Normalmente so e preciso ajustar os caminhos da secao 1. As demais secoes
definem o split, o treino e os experimentos usados no trabalho.
"""

import os

RAIZ = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# 1. CAMINHOS
# =============================================================================

# Conjunto de patches: manifest.json + patches/AAAAMMDD_yNNNN_xNNNN.npz
DADOS = os.path.join(RAIZ, "dados", "DATASET_UNET")

# Onde ficam pesos (.pth), tabelas (.csv) e figuras (.png)
RESULTADOS = os.path.join(RAIZ, "resultados")

# Limite municipal (shapefile de poligono ou de linha)
SHAPEFILE_MUNICIPIO = os.path.join(RAIZ, "dados", "limite_municipal", "Porto Alegre.shp")

# Opcional: split.json de referencia para conferir se o split gerado e identico
SPLIT_REFERENCIA = None

# =============================================================================
# 2. GRADE E MUNICIPIO
# =============================================================================

TAMANHO_PIXEL_M = 20.0                  # resolucao da grade dos patches
MUNICIPIO = {"nome": "Porto Alegre", "codigo_ibge": "4314902", "area_km2": 495.4}

# =============================================================================
# 3. SPLIT (datas x regioes, sem vazamento)
# =============================================================================

SEMENTE = 20261022
FRAC_DATAS = (0.70, 0.15, 0.15)         # treino, validacao, teste
LADO_REGIAO_PX = 512                    # blocos de 512 x 512 px (10,24 km)
FRAC_REGIOES_RETIDAS = 0.25             # fracao dos blocos nunca vistos no treino

# =============================================================================
# 4. TREINO E AVALIACAO
# =============================================================================

EPOCAS = 80
PACIENCIA = 15                          # epocas sem melhora da IoU de validacao
LR = 3e-4
WEIGHT_DECAY = 1e-5
MIN_PIXELS_MOLHADO = 28                 # minimo de pixels de agua para o patch contar como "molhado"

N_BINS = 1000                           # busca de limiar com passo 0,001
N_BOOTSTRAP = 300                       # IC 95% da IoU (reamostragem de patches)
N_BOOTSTRAP_DIF = 1000                  # IC 95% da diferenca de IoU entre dois modelos
CONJUNTOS_AVALIACAO = ("validacao", "teste_tempo", "teste_espaco", "teste_ambos")

# =============================================================================
# 5. EXPERIMENTOS
# =============================================================================

# Grupos de canais usados nas ablacoes (nomes como no manifest.json)
GRUPOS = {
    "chuva": ("precip_total_mm", "precip_max_mmh"),
    "urbano": ("lulc_idx", "imperv", "susc_via"),
    "relevo": ("decl_pct", "hand_m", "log_area_contrib", "cota_m"),
}

# Receita base (R1). Cada experimento informa apenas o que muda em relacao a ela.
PADRAO = {
    "arquitetura": "unet_pequena",      # "unet_pequena" | "smp_unet" | "smp_unetpp" | "smp_deeplabv3p" | "smp_unet_scse"
    "perda": "bce_dice",                # "bce_dice" | "tversky"
    "alfa": 0.5, "beta": 0.5,           # pesos de FP e FN, usados quando perda = "tversky"
    "aumento": False,                   # rotacoes de 90 graus + espelhamento no treino
    "tirar": (),                        # grupos de canais removidos, ex.: ("chuva",)
    "lote": 16,                         # patches por lote
    "frac_molhado": 0.7,                # fracao de patches com agua sorteados em cada lote
    "epocas": EPOCAS,
    "semente": SEMENTE,                 # semente do treino (pesos iniciais e sorteio dos lotes);
                                        # o split usa sempre SEMENTE
}

EXPERIMENTOS = {
    # --- Funcao de perda e aumento de dados (U-Net simples) ---
    "R0_controle_tversky": {"perda": "tversky", "alfa": 0.3, "beta": 0.7},
    "R1_bce_dice":         {},
    "R2_bce_dice_aug":     {"aumento": True},
    "R2b_aug_150ep":       {"aumento": True, "epocas": 150},
    # --- Ablacao de grupos de variaveis, com aumento de dados ---
    "R3_sem_chuva":        {"tirar": ("chuva",), "aumento": True},
    "R4_sem_urbano":       {"tirar": ("urbano",), "aumento": True},
    "R5_sem_relevo":       {"tirar": ("relevo",), "aumento": True},
    # --- Ablacao de grupos de variaveis, sem aumento de dados ---
    "R3b_sem_chuva":       {"tirar": ("chuva",)},
    "R4b_sem_urbano":      {"tirar": ("urbano",)},
    "R5b_sem_relevo":      {"tirar": ("relevo",)},
    # --- Arquiteturas com encoder ResNet-34 pre-treinado (segmentation_models_pytorch) ---
    "R6_unet_resnet34":    {"arquitetura": "smp_unet"},
    "R7_unetpp_resnet34":  {"arquitetura": "smp_unetpp", "lote": 8},
    "R8_deeplabv3p_r34":   {"arquitetura": "smp_deeplabv3p"},
    "R9_unet_scse_r34":    {"arquitetura": "smp_unet_scse"},
    "R10_r34_sem_urbano":  {"arquitetura": "smp_unet", "tirar": ("urbano",)},
    # --- Repeticoes com outras sementes de treino ---
    "R1_s2":               {"semente": SEMENTE + 1},
    "R1_s3":               {"semente": SEMENTE + 2},
    "R6_s2":               {"arquitetura": "smp_unet", "semente": SEMENTE + 1},
    "R6_s3":               {"arquitetura": "smp_unet", "semente": SEMENTE + 2},
    "R10_s2":              {"arquitetura": "smp_unet", "tirar": ("urbano",), "semente": SEMENTE + 1},
    "R10_s3":              {"arquitetura": "smp_unet", "tirar": ("urbano",), "semente": SEMENTE + 2},
}

# Experimentos treinados por padrao, nesta ordem. Os que ja terminaram sao apenas
# recarregados e reavaliados.
RODAR = list(EXPERIMENTOS)

# Mesma receita com sementes diferentes: o relatorio mostra media e desvio-padrao
REPLICAS = {
    "R1 . U-Net simples": ["R1_bce_dice", "R1_s2", "R1_s3"],
    "R6 . U-Net ResNet-34": ["R6_unet_resnet34", "R6_s2", "R6_s3"],
    "R10 . ResNet-34 sem urbano": ["R10_r34_sem_urbano", "R10_s2", "R10_s3"],
}

REFERENCIA = "R1_bce_dice"              # modelo base das comparacoes pareadas
EXPERIMENTO_FINAL = "R6_unet_resnet34"  # usado nos mapas e na avaliacao do municipio
