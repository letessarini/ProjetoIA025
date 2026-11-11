# ProjetoIA025 - pipeline plan

## Workflow rule

- Keep all important pending work here in `to-do.md`.
- Keep completed, validated, and design decisions in `DEVELOP.md`.
- Use this file for the open research and engineering tasks that still need execution.

## Contexto do projeto

Este repositório é um projeto de segmentação de inundação usando U-Net sobre patches espaciais gerados a partir de dados geoespaciais e hidrológicos.

O objetivo final é treinar um modelo que, a partir de um patch 2D com vários canais físicos/ambientais, preveja uma máscara binária de inundação (`y`), indicando quais pixels estão inundados.

A estrutura atual do projeto já contém:
- uma pasta `DATASET_UNET/` com manifest e patches em `.npz`
- um script principal de treino em `nlp_34_treinar_unet.py`
- utilitários em `utils/` para validar, separar e visualizar patches

A nova pipeline deve coexistir com essa estrutura atual sem quebrar o que já funciona.

## Dados que o pipeline entende

Os patches são arquivos `.npz` com a seguinte estrutura esperada:
- `X`: array com os canais de entrada, normalmente com formato `(C, H, W)`
  - exemplo observado: `(10, 256, 256)`
  - canais típicos: precipitação, uso do solo, impermeabilização, declividade, suscetibilidade, etc.
- `y`: máscara alvo, normalmente `(256, 256)`
  - valores binários: `0` = sem inundação, `1` = inundação
- `valido`: máscara de pixels válidos, normalmente `(256, 256)`
  - pixels fora do domínio ou inválidos não devem contar na perda/métrica
- `date`: data do patch
- `precip_val`: valor de precipitação associado
- `frac_valida`: fração válida do patch

O `manifest.json` da base contém:
- grade espacial da cidade
- canais disponíveis
- normalizações e preenchimentos para cada canal
- configuração de patches (`tamanho`, `passo`, `min_px_agua`, etc.)

## O que o modelo faz

O modelo é um classificador de pixels para segmentação semântica:
- Entrada: patch com vários canais físicos e geográficos
- Saída: máscara de inundação do mesmo tamanho do patch
- Objetivo: aprender a relação entre variáveis espaciais e a ocorrência de alagamento

A implementação atual usa uma U-Net de pequena escala com função de perda Tversky mascarada, adequada para problemas com classes desbalanceadas e área inundada pequena comparada ao total do patch.

## Pipeline planejada

A estrutura nova deve ficar dentro de `pipeline/` e deve funcionar como uma camada de organização, sem apagar ou alterar os scripts antigos.

### `pipeline/`
- `pipeline/orchestrator/`
  - ponto de entrada principal para rodar a pipeline inteira
  - orquestra dados, treino e avaliação
- `pipeline/data_preparation/`
  - descoberta do dataset
  - validação de arquivos `.npz`
  - checagem de canais/shape
  - preparação de splits treino/val/teste
  - organização dos patches em conjuntos
- `pipeline/models/`
  - definição/encapsulamento do modelo
  - treinamento
  - checkpoint salva
- `pipeline/metrics/`
  - cálculo de IoU, F1, precisão, recall, intervalos de confiança
  - geração de relatórios e gráficos

## Objetivos de desenvolvimento

### Fase 1 - organização inicial
- [x] Criar a pasta `pipeline/` e suas subpastas principais
- [x] Criar uma estrutura de referência que não interfira com o código atual
- [x] Criar `to-do.md` com a visão geral da pipeline
- [x] Criar `DEVELOP.md` com notas de desenvolvimento e decisões válidas
- [x] Criar `u-net-tuning-guide.md` com os parâmetros editáveis do U-Net e seus efeitos
- [x] Criar `pipeline/orchestrator/run_pipeline.py` como entry point
- [x] Mover a lógica de descoberta do dataset para `pipeline/data_preparation/`
- [x] Separar a lógica de treino em `pipeline/models/`
- [x] Separar a lógica de avaliação em `pipeline/metrics/`

### Fase 2 - robustez e reproduzibilidade
- [x] Construir um sistema de configuração centralizado
- [x] Garantir que o pipeline funcione com `DATASET_UNET/` ou `DATASET_UNET/DATASET_UNET/`
- [x] Padronizar nomes de pastas e artefatos de saída
- [x] Definir diretórios de resultados (`results/`, `artifacts/`, `checkpoints/`)
- [x] Garantir que os scripts antigos continuem funcionais como referência

### Fase 3 - execução real
- [x] Validar que cada patch lido tem os campos esperados
- [x] Garantir que o split não tem vazamento temporal ou espacial
- [x] Treinar o modelo com dados locais do repositório
- [x] Salvar métricas e pesos em diretórios padronizados
- [x] Produzir relatórios claros para comparação entre modelos

## Regras de implementação

- Não quebrar o código legado que já existe
- Não apagar `nlp_34_treinar_unet.py`
- Não depender de caminhos locais do autor
- Usar caminhos relativos ao repositório como padrão
- Manter a estrutura de dados atual e apenas encapsulá-la em módulos melhores

## Objetivos a preencher pelo usuário

- [ ] Definir exatamente o que queremos prever para o artigo: benchmark, publicação, ou apenas validação interna
- [ ] Definir quais métricas são prioritárias para reportar
- [ ] Definir se o pipeline deve rodar apenas treino, ou também validação e comparação de baselines
- [ ] Definir se queremos manter o nome dos experimentos atuais (`A_so_terreno`, `B_so_chuva`, etc.)
- [ ] Definir se queremos adicionar pipeline de comparação entre modelos e baselines sem rede

## Próximos passos imediatos

1. Criar módulos prontos para `data_preparation`
2. Extrair a descoberta de dataset e validação para um único módulo
3. Criar o primeiro orchestrator que apenas chama os passos de preparação e treino
4. Manter as versões existentes em `nlp_34_treinar_unet.py` e `utils/` como referência e fallback
5. Só depois refatorar para um design mais “produtivo” e reutilizável

## Tuning e experimentação pendente

- [ ] Rodar smoke test com `limit` pequeno para validação rápida
- [ ] Rodar treino full-data para avaliação real de performance
- [ ] Tunar `base_channels` e `depth` para capacidade e custo
- [ ] Ajustar `learning_rate` e `dropout` para convergência estável
- [ ] Ajustar `alpha`/`beta` da perda Tversky para class imbalance
- [ ] Verificar o efeito de `limit` em comparação entre execuções
- [ ] Registrar resultados em `results/` com metadata reproduzível
