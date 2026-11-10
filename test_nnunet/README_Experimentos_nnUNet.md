# IA025 — Segmentação de Inundações com nnU-Net v2
Documentação do pipeline de preparação, treinamento, validação e teste de modelos de segmentação semântica para identificação de pixels inundados em patches geoespaciais. O projeto inclui três configurações iniciais e um experimento adicional concluído com função de perda personalizada, além de uma avaliação de thresholds de probabilidade.

## 1. Objetivo e experimentos
| Experimento | Dados de entrada | Estratégia | Situação |
|---|---|---|---|
| Dataset101 | 9 canais | Baseline com patches positivos e negativos | Treinado e avaliado |
| Dataset102 | 7 canais | Remoção dos dois canais de precipitação | Treinado e avaliado |
| Dataset103 | 9 canais | Apenas patches positivos no treinamento | Treinado e avaliado |
| Dataset102_2 | 7 canais (Dataset102) | Tversky + Cross-Entropy, com α=0,7 e β=0,3 | Treinado e avaliado |

O Dataset102_2 reutiliza os dados e o ID **102**, sem criar um quarto dataset físico. Os resultados ficam separados pelo nome do trainer.

## 2. Dados e divisão por data
O conjunto original contém **1.458 patches**, com resolução **256 × 256**, provenientes de **54 datas**. A separação foi realizada por data de aquisição, evitando a sobreposição de datas entre treinamento, validação e teste.
| Informação | Dataset101 | Dataset102 | Dataset103 |
|---|---:|---:|---:|
| Canais | 9 | 7 | 9 |
| Patches de treinamento | 1.026 | 1.026 | 687 |
| Treino positivos | 687 | 687 | 687 |
| Treino negativos | 339 | 339 | 0 |
| Patches de validação | 216 | 216 | 216 |
| Validação positivos / negativos | 121 / 95 | 121 / 95 | 121 / 95 |
| Patches de teste | 216 | 216 | 216 |
| Teste positivos / negativos | 139 / 77 | 139 / 77 | 139 / 77 |

Divisão por datas: **38 de treinamento, 8 de validação e 8 de teste**. No Dataset103, foram removidos **339 patches negativos apenas do treinamento**, preservando a validação e o teste.

### Canais
O Dataset101 utiliza nove variáveis: precipitação total, precipitação máxima, uso do solo, impermeabilização, declividade, suscetibilidade de estradas, HAND, log da área contribuinte e elevação. O Dataset102 remove as duas variáveis de precipitação. O Dataset103 conserva os nove canais.

### Rótulos
- `0`: fundo / sem inundação;

- `1`: inundação;

- `2`: `ignore`, para pixels inválidos.

Um patch é classificado como **positivo** se possui ao menos um pixel com rótulo `1`; caso contrário, é **negativo**.

## 3. Ambiente
- Projeto: `~/ProjetoIA025`

- Ambiente Conda: `ia025`

- Framework: nnU-Net v2, PyTorch

- GPU: NVIDIA GeForce RTX 3050 Laptop, 6 GB

- Configuração: `2d`, fold `0`, CUDA
```bash
conda activate ia025
cd ~/ProjetoIA025
export nnUNet_raw="$HOME/ProjetoIA025/nnUNet_raw"
export nnUNet_preprocessed="$HOME/ProjetoIA025/nnUNet_preprocessed"
export nnUNet_results="$HOME/ProjetoIA025/nnUNet_results"
```

Essas variáveis devem ser definidas em cada nova sessão de terminal, caso não estejam configuradas permanentemente.

## 4. Conversão e organização dos dados
Os patches originais `.npz` foram convertidos para arquivos TIFF com o script **`utils/converter_npz_nnunet.py`**, em estrutura compatível com o leitor `NaturalImage2DIO` do nnU-Net v2.

- Com nove canais, cada patch gera `patch_ID_0000.tif` a `patch_ID_0008.tif`.

- Com sete canais, cada patch gera `patch_ID_0000.tif` a `patch_ID_0006.tif`.

- A máscara correspondente é armazenada como `patch_ID.tif`.

- `dataset.json` descreve canais, classes e quantidade de casos de treinamento do nnU-Net.

- `splits_final.json` define os IDs de treinamento e validação.

Estrutura de referência:
```text
nnUNet_raw/
├── Dataset101_Inundacao/
├── Dataset102_InundacaoSemChuva/
└── Dataset103_InundacaoSomentePositivos/
    ├── imagesTr/
    ├── labelsTr/
    ├── imagesTs/
    ├── labelsTs/
    ├── dataset.json
    └── splits_final.json
```

Os diretórios de treinamento incluem os casos usados em **treinamento + validação**. O teste é mantido em `imagesTs` e `labelsTs`. O `numTraining` é **1.242** nos Datasets 101 e 102 e **903** no Dataset103.

A integridade dos canais e das máscaras foi verificada após a conversão. A contagem de patches positivos e negativos foi conferida com **`utils/verificar_distribuicao_datasets.py`**.

## 5. Fingerprint, planejamento e pré-processamento

### 5.1. Planejamento automático
Primeiro comando documentado para o Dataset101:
```bash
nnUNetv2_plan_and_preprocess -d 101 -c 2d --verify_dataset_integrity --no_pp -npfp 2
```

Esse comando verifica a integridade, extrai o **fingerprint** e gera o planejamento, sem executar o pré-processamento devido a `--no_pp`.

O planejamento adotado utiliza **PlainConvUNet 2D**, sete estágios e patches **256 × 256**. O batch size foi ajustado para **4** em função da GPU de 6 GB, e foi utilizado um plano nomeado `nnUNetPlansZScore` com normalização ZScore.

### 5.2. Pré-processamento
Comando inicial registrado:
```bash
nnUNetv2_preprocess -d 101 -c 2d -np 2
```

Para o plano nomeado, o procedimento correspondente é:
```bash
nnUNetv2_preprocess -d 101 -c 2d -plans_name nnUNetPlansZScore
nnUNetv2_preprocess -d 102 -c 2d -plans_name nnUNetPlansZScore
nnUNetv2_preprocess -d 103 -c 2d -plans_name nnUNetPlansZScore
```

**Atenção:** o plano `nnUNetPlansZScore` deve existir e estar configurado antes do pré-processamento. As linhas exatas de todas as operações preparatórias originais dos Datasets 101 e 102 não estão integralmente preservadas nos registros.

Para o Dataset103, os comandos de preparação registrados incluem:
```bash
nnUNetv2_extract_fingerprint -d 103
nnUNetv2_plan_experiment -d 103
nnUNetv2_preprocess -d 103 -c 2d -plans_name nnUNetPlansZScore
```

O `splits_final.json` personalizado também foi colocado no diretório pré-processado.

## 6. Treinamentos concluídos
Parâmetros comuns: `nnUNetTrainer`, configuração `2d`, fold `0`, plano `nnUNetPlansZScore`, batch size `4`, patches `256 × 256`, ZScore, GPU CUDA e **1.000 épocas**. A perda original é **Dice + Cross-Entropy** (`DC_and_CE_loss`), com `ignore_label=2`.
```bash
# Dataset101 — baseline de 9 canais
nnUNetv2_train 101 2d 0 -tr nnUNetTrainer -p nnUNetPlansZScore -device cuda
# Dataset102 — 7 canais, sem precipitação
nnUNetv2_train 102 2d 0 -tr nnUNetTrainer -p nnUNetPlansZScore -device cuda
# Dataset103 — 9 canais, somente positivos no treinamento
nnUNetv2_train 103 2d 0 -tr nnUNetTrainer -p nnUNetPlansZScore -device cuda
```

### Validação
O **pseudo Dice** apresentado durante as épocas não equivale necessariamente ao Dice da validação completa. Para validar explicitamente o melhor checkpoint:
```bash
nnUNetv2_train 101 2d 0 -tr nnUNetTrainer -p nnUNetPlansZScore --val --val_best -device cuda
```

Substituir o ID para os demais experimentos.
| Dataset | Mean Validation Dice registrado |
|---|---:|
| 101 | 34,88% |
| 102 | 32,99% |
| 103 | 33,86% |

Os números acima são os valores de validação registrados no relatório; não devem ser comparados diretamente com o Dice global do teste, pois representam avaliações e agregações distintas.

## 7. Inferência e avaliação no teste
Os três modelos foram utilizados para inferir os mesmos **216 patches de teste**, com `checkpoint_best.pth` e fold `0`.
```bash
# Dataset101
nnUNetv2_predict \
  -i nnUNet_raw/Dataset101_Inundacao/imagesTs \
  -o resultados_teste/Dataset101 \
  -d 101 -c 2d -f 0 \
  -tr nnUNetTrainer -p nnUNetPlansZScore \
  -chk checkpoint_best.pth -device cuda
# Dataset102
nnUNetv2_predict \
  -i nnUNet_raw/Dataset102_InundacaoSemChuva/imagesTs \
  -o resultados_teste/Dataset102 \
  -d 102 -c 2d -f 0 \
  -tr nnUNetTrainer -p nnUNetPlansZScore \
  -chk checkpoint_best.pth -device cuda
# Dataset103
nnUNetv2_predict \
  -i nnUNet_raw/Dataset103_InundacaoSomentePositivos/imagesTs \
  -o resultados_teste/Dataset103 \
  -d 103 -c 2d -f 0 \
  -tr nnUNetTrainer -p nnUNetPlansZScore \
  -chk checkpoint_best.pth -device cuda
```

Avaliação por `avaliar_teste.py`, excluindo pixels com rótulo `2`:
```bash
python avaliar_teste.py --dataset 101 --max-figuras 20
python avaliar_teste.py --dataset 102 --max-figuras 20
python avaliar_teste.py --dataset 103 --max-figuras 20
```

O script gera `metricas_por_patch.csv`, `resumo.csv` e figuras. O parâmetro `--max-figuras` limita somente a quantidade de figuras, não a avaliação dos 216 patches.

### Resultados de teste
| Métrica | Dataset101 | Dataset102 | Dataset103 |
|---|---:|---:|---:|
| Dice global | 53,00% | **59,00%** | 56,86% |
| IoU global | 36,05% | **41,85%** | 39,72% |
| Precision global | **54,06%** | 49,53% | 51,93% |
| Recall global | 51,98% | **72,96%** | 62,81% |
| Dice médio por patch definido | 38,27% | **44,74%** | 40,12% |
| IoU médio por patch definido | 29,02% | **35,46%** | 31,80% |
| Pixels FP nos 77 patches negativos | **27.675** | 57.178 | 36.076 |

**Interpretação:** o Dataset102 apresentou maior Dice, IoU e Recall global, mas também mais falsos positivos. O Dataset101 apresentou maior Precision e menos falsos positivos nos patches negativos; o Dataset103 ficou em posição intermediária.

O Dice e o IoU globais são calculados a partir da soma dos TP, FP e FN dos patches. A média por patch considera apenas os casos em que a métrica está definida.

### Visualização e pós-processamento
As figuras mostram máscara real, máscara prevista e mapa de erros: verde = TP, vermelho = FP, amarelo = FN, cinza = ignore. Não foi aplicado pós-processamento personalizado nos testes concluídos; apenas a exportação padrão do nnU-Net.

## 8. Diagnóstico adicional dos erros do Dataset102
Nos 77 patches negativos do teste, **39 apresentaram pelo menos um falso positivo** (50,65%), somando **57.178 pixels FP**. A comparação visual de patches como `patch_0034`, `patch_0048` e `patch_1087` indicou regiões previstas mais extensas do que as máscaras reais. Essa observação motivou a investigação de uma perda com penalização assimétrica de FP e FN.

A análise da proporção de pixels inundados, realizada sobre as máscaras do Dataset102, mostrou:
| Indicador | Treinamento | Validação |
|---|---:|---:|
| Patches totais | 1.026 | 216 |
| Patches positivos | 687 | 121 |
| Pixels válidos | 47.066.762 | 9.908.792 |
| Pixels inundados | 1.118.611 | 178.863 |
| Fração inundada entre pixels válidos | **2,3766%** | **1,8051%** |
| Mediana da fração inundada nos patches positivos | 1,4542% | 0,8522% |
| Percentil 90 nos patches positivos | 9,2965% | 6,0128% |

Script utilizado: `utils/analisar_pixels_inundacao.py`. Saídas: `analise_pixels_inundacao/distribuicao_pixels.csv` e `analise_pixels_inundacao/distribuicao_pixels.png`.

**Hipótese:** a baixa ocupação de inundação e os falsos positivos extensos justificam testar uma função de perda que penalize mais FP. Isso não demonstra, por si só, a causa dos erros.

## 9. Dataset102_2 — Tversky + Cross-Entropy

### 9.1. Objetivo
Investigar se a substituição da Dice Loss por **Tversky Loss**, mantendo Cross-Entropy, reduz a superestimação das regiões inundadas sem prejudicar excessivamente o Recall.

Índice Tversky:

\[

TI = \frac{TP + \varepsilon}{TP + \alpha FP + \beta FN + \varepsilon}

\]

\[

L = (1 - TI) + L_{CE}, \quad \alpha = 0{,}7, \quad \beta = 0{,}3

\]

O maior peso atribuído a FP é uma **hipótese experimental**, não uma melhoria garantida.

### 9.2. Configuração
| Parâmetro | Valor |
|---|---|
| Dataset reutilizado | `102` — 7 canais |
| Treinamento / validação | 1.026 / 216 patches |
| Trainer | `nnUNetTrainerTversky` |
| Código personalizado | `nnunet_custom/nnUNetTrainerTversky.py` |
| Arquitetura | PlainConvUNet 2D, 7 estágios |
| Plano | `nnUNetPlansZScore` |
| Batch size / patch | 4 / 256 × 256 |
| Fold / GPU | 0 / CUDA |
| Épocas previstas | 1.000 |
| Loss | Tversky + Cross-Entropy |
| α / β | 0,7 / 0,3 |
| Ignore label | 2 |
| Batch Dice | `True` (agregação no batch) |
| Deep supervision | Habilitada |

A implementação personalizada calcula a Tversky sobre a classe inundação (`1`), soma TP/FP/FN sobre o batch, exclui os pixels com rótulo `2` e combina o termo Tversky com Cross-Entropy. A deep supervision utiliza os pesos definidos pelo trainer original.

### 9.3. Registro do trainer externo
A versão instalada do nnU-Net permite localizar trainers fora do pacote por meio de **`nnUNet_extTrainer`**:
```bash
export nnUNet_extTrainer="$HOME/ProjetoIA025/nnunet_custom"
```

Para verificar a descoberta da classe:
```bash
python - <<'PY'
from nnunetv2.run.run_training import recursive_find_trainer_class_by_name
trainer = recursive_find_trainer_class_by_name('nnUNetTrainerTversky')
print('Trainer encontrado:', trainer.__name__)
PY
```

A busca do trainer personalizado foi **confirmada** no ambiente `ia025`.

### 9.4. Testes realizados antes do treinamento
| Teste | Resultado |
|---|---|
| Localização do trainer externo | OK |
| Loss com máscara aleatória e backward | Loss finita, gradientes calculados |
| Todos os pixels ignorados | Loss `0.0`, gradientes finitos |
| Somente fundo | Loss `1.9041`, gradientes finitos |
| Somente inundação | Loss `1.1372`, gradientes finitos |
| Deep supervision simulada | Loss `1.4048`, gradientes finitos nas saídas com peso não nulo |
| Inicialização real | OK, `OptimizedModule`, `DeepSupervisionWrapper`, CUDA |
| Uma iteração real | OK: entrada `[4, 7, 256, 256]`, 6 targets, loss `1.7523192` |
| Encerramento de dataloaders | OK após encerramento explícito com `_finish()` |

A memória máxima alocada reportada nessa iteração foi **0,71 GB**. Esse número não deve ser interpretado como o pico garantido de um treinamento completo.

O primeiro teste de uma iteração produziu mensagens de erro ao encerrar workers em segundo plano; a repetição com fechamento explícito dos dataloaders terminou sem esses erros. Não houve indicação de falha na loss durante a iteração.

### 9.5. Comando para iniciar o treinamento completo
**Comando utilizado para o treinamento completo:**
```bash
conda activate ia025
cd ~/ProjetoIA025
export nnUNet_raw="$HOME/ProjetoIA025/nnUNet_raw"
export nnUNet_preprocessed="$HOME/ProjetoIA025/nnUNet_preprocessed"
export nnUNet_results="$HOME/ProjetoIA025/nnUNet_results"
export nnUNet_extTrainer="$HOME/ProjetoIA025/nnunet_custom"
nnUNetv2_train 102 2d 0 \
  -tr nnUNetTrainerTversky \
  -p nnUNetPlansZScore \
  -device cuda
```

Pasta de resultados esperada:
```text
nnUNet_results/Dataset102_InundacaoSemChuva/
└── nnUNetTrainerTversky__nnUNetPlansZScore__2d/
    └── fold_0/
```

### 9.6. Procedimento de validação
Para validar o melhor checkpoint utilizando o trainer personalizado:
```bash
nnUNetv2_train 102 2d 0 \
  -tr nnUNetTrainerTversky \
  -p nnUNetPlansZScore \
  --val --val_best -device cuda
```

A seleção do modelo e eventuais ajustes de hiperparâmetros devem ser feitos **na validação**. O teste deve ser utilizado para a comparação final, evitando escolher parâmetros com base nos exemplos de teste já inspecionados. Comparar Dice, IoU, Precision, Recall e falsos positivos, inclusive por data e em patches negativos.

### 9.7. Treinamento completo e seleção do checkpoint

O treinamento do Dataset102_2 foi concluído com **1.000 épocas**, utilizando o `nnUNetTrainerTversky`, os sete canais do Dataset102, o plano `nnUNetPlansZScore` e o fold `0`.

O melhor checkpoint (`checkpoint_best.pth`) foi salvo na **943ª época** (índice de época `942` no log). O melhor pseudo Dice EMA registrado foi **0,4403**. Esse valor é uma métrica de acompanhamento do treinamento e não equivale ao Dice global no teste.

### 9.8. Validação do melhor checkpoint

Foi executada a validação completa com o checkpoint selecionado:
```bash
nnUNetv2_train 102 2d 0 \
  -tr nnUNetTrainerTversky \
  -p nnUNetPlansZScore \
  --val --val_best -device cuda
```

**Mean Validation Dice: 32,85%** (`0,32853144744302726`). Para referência, o Dataset102 original registrou **32,99%** nessa avaliação.

### 9.9. Inferência e avaliação nos 216 patches de teste

A inferência utilizou o melhor checkpoint e salvou as máscaras previstas em `resultados_teste/Dataset102_Tversky/`:
```bash
nnUNetv2_predict \
  -i nnUNet_raw/Dataset102_InundacaoSemChuva/imagesTs \
  -o resultados_teste/Dataset102_Tversky \
  -d 102 -c 2d -f 0 \
  -tr nnUNetTrainerTversky \
  -p nnUNetPlansZScore \
  -chk checkpoint_best.pth -device cuda
```

Foram geradas **216 máscaras TIFF**. A avaliação foi realizada com:
```bash
python avaliar_teste.py --dataset 102 --modelo Tversky --max-figuras 216
```

O script avaliou **216 patches**, sendo **139 positivos** e **77 negativos**, desconsiderando os pixels com rótulo `2` (`ignore`). Os resultados foram gravados em `resultados_teste/avaliacao_Dataset102_Tversky/`, incluindo `resumo.csv`, `metricas_por_patch.csv` e figuras de comparação.

### 9.10. Resultados e comparação com o Dataset102 original
| Métrica | Dataset102 original (Dice + CE) | Dataset102_2 (Tversky + CE) |
|---|---:|---:|
| Dice global | **59,00%** | 53,33% |
| IoU global | **41,85%** | 36,36% |
| Precision global | 49,53% | **55,54%** |
| Recall global | **72,96%** | 51,29% |
| TP | **140.921** | 99.061 |
| FP | 143.599 | **79.291** |
| FN | **52.224** | 94.084 |
| TN | 9.572.048 | 9.636.356 |
| Dice médio por patch definido | **44,74%** | 38,82% |
| IoU médio por patch definido | **35,46%** | 29,11% |
| Pixels FP nos 77 patches negativos | 57.178 | **28.575** |

O modelo com Tversky + Cross-Entropy apresentou **redução de aproximadamente 44,8% nos falsos positivos totais** (de 143.599 para 79.291) e **50,0% nos falsos positivos em patches negativos** (de 57.178 para 28.575). A Precision aumentou de **49,53% para 55,54%**.

Entretanto, o Recall caiu de **72,96% para 51,29%**, com aumento dos falsos negativos de **52.224 para 94.084**. O Dice global diminuiu de **59,00% para 53,33%** e o IoU global de **41,85% para 36,36%**.

**Conclusão:** a configuração Tversky + Cross-Entropy (`α=0,7`, `β=0,3`) reduziu os falsos positivos, mas tornou a segmentação mais conservadora, deixando de identificar mais pixels realmente inundados. Portanto, **não superou o Dataset102 original em Dice ou IoU global**. Como próximo passo, propõe-se investigar o desbalanceamento de pixels inundados e não inundados e estratégias de amostragem no treinamento, mantendo a validação e o teste separados.

## 10. Observações de reprodutibilidade
1\. Manter os mesmos splits por data para comparar modelos.

2\. Preservar o `ignore_label=2` em todas as perdas e métricas.

3\. Não comparar diretamente valores numéricos de **losses** diferentes.

4\. Diferenciar pseudo Dice por época, Dice da validação completa e Dice global do teste.

5\. O Dataset102_2 é uma nova configuração de **treinamento**, não um novo conjunto de dados.

6\. Não considerar o Dataset102_2 concluído antes de registrar o treinamento completo, checkpoint, validação e avaliação final.

## 11. Avaliação de diferentes thresholds de probabilidade — Dataset102

Com o objetivo de verificar se o ajuste do limiar de classificação poderia melhorar o desempenho do Dataset102 original (7 canais, Dice + Cross-Entropy), foi realizado um experimento utilizando diferentes thresholds de probabilidade para a classe inundação, sem necessidade de novo treinamento.

Foram utilizadas as probabilidades geradas pelo modelo treinado (`checkpoint_best.pth`), avaliando os thresholds **0,3; 0,4; 0,5; 0,6 e 0,7**.

Inicialmente, foram avaliados os **216 patches de validação** (121 positivos e 95 negativos). Posteriormente, os mesmos thresholds foram aplicados aos **216 patches de teste** (139 positivos e 77 negativos), desconsiderando os pixels com rótulo `2` (`ignore`).

#### Resultados na validação
| Threshold | Dice global | IoU global | Precision | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| 0,3 | 41,15% | 25,90% | 32,89% | 54,92% | 200.410 | 80.625 |
| 0,4 | 41,29% | 26,02% | 33,32% | 54,26% | 194.228 | 81.803 |
| 0,5 | 41,38% | 26,09% | 33,70% | 53,61% | 188.649 | 82.976 |
| 0,6 | 41,48% | 26,17% | 34,10% | 52,93% | 182.985 | 84.184 |
| **0,7** | **41,55%** | **26,22%** | **34,57%** | 52,07% | **176.313** | 85.728 |

O threshold de **0,7 apresentou o maior Dice global na validação (41,55%)**, representando um aumento de 0,17 ponto percentual em relação ao threshold padrão de 0,5.

#### Resultados no teste
| Threshold | Dice global | IoU global | Precision | Recall | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| 0,3 | 58,83% | 41,67% | 48,44% | **74,90%** | 153.975 | **48.488** |
| 0,4 | 58,95% | 41,79% | 49,02% | 73,93% | 148.503 | 50.360 |
| **0,5** | 59,00% | 41,85% | 49,53% | 72,96% | 143.599 | 52.224 |
| **0,6** | **59,03%** | **41,87%** | 50,04% | 71,94% | 138.712 | 54.193 |
| 0,7 | 58,98% | 41,82% | **50,62%** | 70,63% | **133.052** | 56.733 |

O threshold de **0,5 reproduziu exatamente os resultados originais do Dataset102**, confirmando a consistência da avaliação.

#### Conclusão

No conjunto de teste, o threshold de **0,6 apresentou o maior Dice global (59,03%)**, representando um aumento de apenas 0,03 ponto percentual em relação ao threshold padrão de 0,5 (59,00%).

O aumento do threshold de 0,5 para 0,6 reduziu os falsos positivos de **143.599 para 138.712** e melhorou a Precision de **49,53% para 50,04%**. Entretanto, houve redução do Recall de **72,96% para 71,94%**, acompanhada do aumento dos falsos negativos de **52.224 para 54.193**.

Dessa forma, **o ajuste do threshold não proporcionou uma melhoria significativa no desempenho do modelo no conjunto de teste**, sendo mantido o threshold padrão de **0,5** como referência para o Dataset102.

## 12. Referência
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). **nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.** **Nature Methods**, 18(2), 203–211.