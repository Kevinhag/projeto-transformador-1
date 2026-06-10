# Classificação de Flores com Transfer Learning

##### Gabriel A. O. Schlagenhaufer, Guilherme Ferraz, Igor T. Gutierrez, Kevin H. A. Guimarães, Patrick L. Froes

Projeto de visão computacional em duas fases:

1. **Fase 1 — Classificação Oxford Flowers-102**: treina e avalia três arquiteturas no dataset Oxford 102 Category Flower.
2. **Fase 2 — Detecção OOD**: avalia se os modelos treinados conseguem distinguir flores de folhas (out-of-distribution) num dataset COCO anotado.

## Modelos

| Nome | timm ID | Parâmetros |
|------|---------|-----------|
| MobileNetV3-Large | `mobilenetv3_large_100` | 4,33 M |
| EfficientViT-M4 | `efficientvit_m4` | 8,46 M |
| Swin-Tiny | `swin_tiny_patch4_window7_224` | 27,60 M |

## Estrutura

```
.
├── ml.py               # Fase 1: treino e avaliação no Flowers-102
├── transfer.py         # Fase 2: detecção OOD (flores x folhas)
├── avaliaEspecies.py   # Análise por espécie nos modelos treinados
├── plot.py             # Geração de todas as figuras e tabelas
├── settings.json       # Configurações globais (hiperparâmetros, paths, modelos)
├── requirements.txt    # Dependências
├── data/
│   ├── flowers-102/    # Dataset Oxford (baixado automaticamente)
│   └── dataset2/       # Dataset COCO flores x folhas
│       ├── train/
│       ├── valid/
│       └── test/
└── results/
    ├── <Modelo>_best.pt             # Checkpoints salvos pelo ml.py
    ├── history.json                 # Curvas de treino
    ├── metrics.json                 # Métricas finais
    ├── evaluation_results.json      # Resultados OOD (transfer.py)
    └── fig*.png / tabela1.png       # Figuras geradas pelo plot.py
```

## Instalação

```bash
# 1. Instale o PyTorch com suporte CUDA 12.8 (RTX 5070 / Blackwell)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. Instale as demais dependências
pip install timm numpy scipy pillow rich matplotlib roboflow
```

> Para outras GPUs, consulte [pytorch.org/get-started](https://pytorch.org/get-started/locally/) e ajuste o índice do `--index-url`.

## Uso

### Fase 1 — Treino (Oxford Flowers-102)

```bash
python ml.py
```

O dataset é baixado automaticamente na primeira execução. Os checkpoints e métricas são salvos em `results/`.

### Avaliação por espécie

```bash
python avaliaEspecies.py
```

Gera CSVs e gráficos das espécies com maior e menor acurácia em `results/`.

### Fase 2 — Detecção OOD (flores x folhas)

Coloque o dataset COCO anotado em:

```
data/dataset2/train/_annotations.coco.json
data/dataset2/valid/_annotations.coco.json
data/dataset2/test/_annotations.coco.json
```

Opcionalmente, configure o download automático via Roboflow no `settings.json`:

```json
"ood": {
  "roboflow": {
    "api_key": "SUA_CHAVE",
    "workspace": "seu-workspace",
    "project": "nome-do-projeto",
    "version": 1,
    "format": "coco"
  }
}
```

Depois execute:

```bash
python transfer.py
```

### Geração de figuras

```bash
python plot.py
```

Gera `fig1`–`fig5` e `tabela1` em `results/`.

## Configuração (`settings.json`)

| Chave | Padrão | Descrição |
|-------|--------|-----------|
| `data_dir` | `./data` | Diretório dos datasets |
| `output_dir` | `./results` | Diretório de saída |
| `num_classes` | `102` | Número de classes (Flowers-102) |
| `num_epochs` | `50` | Épocas de treino |
| `batch_size` | `8` | Tamanho do batch |
| `lr` | `1e-4` | Learning rate |
| `img_size` | `224` | Resolução de entrada |
| `warmup_epochs` | `5` | Épocas de warmup do scheduler |
| `ood.threshold` | `0.5` | Limiar de confiança para OOD |
| `ood.bbox_expand` | `0.10` | Expansão do bounding box (10%) |

## Resultados (50 épocas)

| Modelo | Acurácia (Flowers-102) |
|--------|----------------------|
| Swin-Tiny | 98,78% |
| MobileNetV3-Large | 97,97% |
| EfficientViT-M4 | 89,26% |
