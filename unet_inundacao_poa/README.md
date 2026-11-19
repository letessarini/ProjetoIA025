# Previsão da mancha de inundação em Porto Alegre com redes convolucionais

Código do projeto da disciplina IA025 (Aprendizado Profundo, Unicamp). Redes de
segmentação (U-Net e variantes) preveem a mancha de inundação a partir de chuva,
uso do solo e relevo, usando como referência manchas simuladas com o SFINCS.

## Estrutura

| Arquivo | Função |
|---|---|
| `config.py` | caminhos, split, hiperparâmetros e definição dos experimentos |
| `comum.py` | leitura dos patches, split sem vazamento, métricas, climatologia, redes |
| `treinar_unet.py` | treina e avalia os experimentos (R0–R10 e repetições com outras sementes) |
| `mapas_previsao.py` | mapas de mancha real, prevista e erro |
| `mascara_municipio.py` | converte o limite municipal (shapefile) numa máscara na grade dos patches |
| `avaliacao_municipio.py` | avaliação no mapa reconstruído do município |
| `figuras_resumo.py` | figuras-resumo da avaliação no município |
| `verificar_patches.py` | encontra arquivos `.npz` corrompidos |

## Instalação

```bash
pip install -r requirements.txt
```

Para usar a GPU, instale o PyTorch com CUDA conforme <https://pytorch.org/get-started/locally/>.
O `geopandas` só é necessário para o `mascara_municipio.py`.

## Dados

Ajuste os caminhos na seção 1 do `config.py`. O conjunto de patches deve ter:

```
DATASET_UNET/
  manifest.json
  patches/AAAAMMDD_yNNNN_xNNNN.npz
```

- `manifest.json`: `canais` (nomes dos canais de entrada, na ordem de `X`),
  `datas` (lista com `data` e `chuva_total_mm_media`), `grade.forma` (linhas e
  colunas da grade) e `patch.tamanho` (lado do patch em pixels). Para gerar a
  máscara do município, a parte `grade` também deve ter a georreferência
  (`transform` e `crs`), ou deve haver um GeoTIFF na mesma grade (veja
  `mascara_municipio.py`).
- cada `.npz`: `X` (canais × 256 × 256), `y` (mancha, 0/1) e `valido` (pixels
  dentro do domínio simulado). `yNNNN_xNNNN` é a posição do canto do patch na grade.

## Execução

```bash
python verificar_patches.py                 # opcional
python treinar_unet.py --modo teste_rapido  # confere o ambiente (2 épocas curtas)
python treinar_unet.py                      # todos os experimentos de config.RODAR
python mapas_previsao.py
python mascara_municipio.py
python avaliacao_municipio.py
python figuras_resumo.py
```

Experimentos já treinados (com `.pth` em `resultados/`) são apenas recarregados e
reavaliados; um treino interrompido recomeça do zero. Para treinar só alguns:
`python treinar_unet.py --experimentos R6_unet_resnet34 R6_s2 R6_s3`.

## Metodologia (resumo)

- **Split sem vazamento.** As datas são divididas em treino, validação e teste
  (70/15/15, estratificadas pela chuva). A grade é dividida em regiões de
  512 × 512 px, das quais 25% nunca são vistas no treino. O cruzamento gera
  `treino`, `validacao`, `teste_tempo` (datas novas), `teste_espaco` (lugares
  novos) e `teste_ambos`. O `split.json` gerado fica em `resultados/`; para
  conferir que outro computador gerou o mesmo split, aponte `SPLIT_REFERENCIA`
  para ele.
- **Treino.** AdamW (lr 3e-4) com cosseno, precisão mista na GPU, até 80 épocas
  com parada antecipada pela IoU de validação (paciência 15). Lotes com 70% de
  patches com água. Perda BCE + Dice (R0 usa Tversky com α = 0,3 e β = 0,7).
- **Limiar.** Escolhido na validação (passo 0,001) e aplicado sem mudança nos testes.
- **Baselines.** "Tudo seco" e climatologia (frequência de inundação de cada
  pixel nas datas de treino, com limiar também escolhido na validação).
- **Incerteza.** IC 95% da IoU por bootstrap de patches; comparações pareadas
  por bootstrap de patches e de datas; três sementes para R1, R6 e R10.
- **Escala do município.** As previsões dos patches são encaixadas no mapa de
  cada data e avaliadas dentro do limite municipal, junto com a média por
  patch, a soma dos patches e a previsão com janelas sobrepostas.

## Saídas principais (`resultados/`)

| Arquivo | Conteúdo |
|---|---|
| `resultados.csv` | métricas por experimento e conjunto, com IC 95% |
| `comparacoes_pareadas.csv` | diferenças de IoU contra a climatologia e contra R1 |
| `replicas.csv` | média e desvio-padrão entre sementes |
| `mapas_<exp>/` | mosaicos e painéis de erro |
| `mascara_municipio*.{npy,json,png}` | máscara do município e figura de conferência |
| `municipio_<exp>/` | avaliação no município (CSV e figuras) |
