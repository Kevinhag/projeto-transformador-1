"""
avaliaEspecies.py — Acurácia por espécie (Oxford 102 Flowers)

Reaproveita EXATAMENTE o mesmo split do ml.py (seed 42) e os checkpoints
salvos em results/<Modelo>_best.pt para:
  - montar a matriz de confusão no conjunto de teste
  - calcular a acurácia por espécie (classe)
  - listar e PLOTAR as espécies que MAIS acertam e as que MAIS erram
  - para as piores, mostrar com qual espécie ela mais se confunde
  - salvar um CSV por modelo e um gráfico (piores + melhores) por modelo

O gráfico usa o MESMO tema claro e as MESMAS cores por modelo do plot.py.

Uso:
  python avaliaEspecies.py            # avalia todos os modelos do settings.json
  python avaliaEspecies.py Swin-Tiny  # avalia só um modelo
"""

import os
import sys
import csv
import torch
import numpy as np
import matplotlib.pyplot as plt

# Reaproveita as funções do ml.py (mesmo split, mesma construção de modelo)
from ml import loadDatasets, buildModel, getDevice, CONFIG, MODELS, MODEL_CONFIG

# Reaproveita cores/estilo do plot.py para ficar idêntico às outras figuras.
# Se o plot.py não estiver acessível, cai num padrão equivalente.
try:
    from plot import COLORS, lighten
except Exception:
    COLORS = {
        "MobileNetV3-Large": "#2BA7B5",
        "EfficientViT-M4":   "#E8743B",
        "Swin-Tiny":         "#8B3FE0",
    }
    def lighten(hex_color, amount):
        h = hex_color.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
        return f"#{int(r+(255-r)*amount):02x}{int(g+(255-g)*amount):02x}{int(b+(255-b)*amount):02x}"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.size": 11, "axes.edgecolor": "#cccccc",
})

# Quantas espécies mostrar em cada extremo
N_SHOW = 15

# Mapeamento oficial id(1..102) -> nome da espécie (Oxford 102 Flowers)
ID_TO_NAME = {
    1: 'pink primrose', 2: 'hard-leaved pocket orchid', 3: 'canterbury bells',
    4: 'sweet pea', 5: 'english marigold', 6: 'tiger lily', 7: 'moon orchid',
    8: 'bird of paradise', 9: 'monkshood', 10: 'globe thistle', 11: 'snapdragon',
    12: "colt's foot", 13: 'king protea', 14: 'spear thistle', 15: 'yellow iris',
    16: 'globe-flower', 17: 'purple coneflower', 18: 'peruvian lily',
    19: 'balloon flower', 20: 'giant white arum lily', 21: 'fire lily',
    22: 'pincushion flower', 23: 'fritillary', 24: 'red ginger', 25: 'grape hyacinth',
    26: 'corn poppy', 27: 'prince of wales feathers', 28: 'stemless gentian',
    29: 'artichoke', 30: 'sweet william', 31: 'carnation', 32: 'garden phlox',
    33: 'love in the mist', 34: 'mexican aster', 35: 'alpine sea holly',
    36: 'ruby-lipped cattleya', 37: 'cape flower', 38: 'great masterwort',
    39: 'siam tulip', 40: 'lenten rose', 41: 'barbeton daisy', 42: 'daffodil',
    43: 'sword lily', 44: 'poinsettia', 45: 'bolero deep blue', 46: 'wallflower',
    47: 'marigold', 48: 'buttercup', 49: 'oxeye daisy', 50: 'common dandelion',
    51: 'petunia', 52: 'wild pansy', 53: 'primula', 54: 'sunflower',
    55: 'pelargonium', 56: 'bishop of llandaff', 57: 'gaura', 58: 'geranium',
    59: 'orange dahlia', 60: 'pink-yellow dahlia', 61: 'cautleya spicata',
    62: 'japanese anemone', 63: 'black-eyed susan', 64: 'silverbush',
    65: 'californian poppy', 66: 'osteospermum', 67: 'spring crocus',
    68: 'bearded iris', 69: 'windflower', 70: 'tree poppy', 71: 'gazania',
    72: 'azalea', 73: 'water lily', 74: 'rose', 75: 'thorn apple',
    76: 'morning glory', 77: 'passion flower', 78: 'lotus lotus', 79: 'toad lily',
    80: 'anthurium', 81: 'frangipani', 82: 'clematis', 83: 'hibiscus',
    84: 'columbine', 85: 'desert-rose', 86: 'tree mallow', 87: 'magnolia',
    88: 'cyclamen', 89: 'watercress', 90: 'canna lily', 91: 'hippeastrum',
    92: 'bee balm', 93: 'ball moss', 94: 'foxglove', 95: 'bougainvillea',
    96: 'camellia', 97: 'mallow', 98: 'mexican petunia', 99: 'bromelia',
    100: 'blanket flower', 101: 'trumpet creeper', 102: 'blackberry lily',
}


def nameOf(label0):
    """label 0-indexado (0..101) -> nome. torchvision usa id_mat - 1."""
    return ID_TO_NAME.get(label0 + 1, f"classe_{label0}")


@torch.no_grad()
def confusionMatrix(model, loader, device, numClasses):
    """Roda o modelo no loader e devolve a matriz de confusão [real, predito]."""
    model.eval()
    cm = np.zeros((numClasses, numClasses), dtype=np.int64)
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(imgs)
        preds = outputs.argmax(dim=1).cpu().numpy()
        for t, p in zip(labels.numpy(), preds):
            cm[t, p] += 1
    return cm


def perClassStats(cm):
    """Devolve lista de (label0, support, acertos, acc, confundeCom_label0)."""
    rows = []
    for c in range(cm.shape[0]):
        support = int(cm[c].sum())
        if support == 0:
            continue  # classe sem amostra no teste deste split
        correct = int(cm[c, c])
        acc = correct / support
        offDiag = cm[c].copy()
        offDiag[c] = -1
        confused = int(offDiag.argmax()) if offDiag.max() > 0 else None
        rows.append((c, support, correct, acc, confused))
    return rows


def buildModelLikeTraining(modelKey, numClasses, device):
    """Constrói o modelo com o MESMO dropout usado no treino (modelConfig)."""
    dropout = MODEL_CONFIG.get(modelKey, {}).get("dropout", 0.0)
    return buildModel(modelKey, numClasses, device, dropout=dropout)


def _styleAx(ax, title, color):
    ax.set_title(title, color=color, fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel("Acurácia por espécie (%)", color="#444")
    ax.set_xlim(0, 108)
    ax.grid(axis="x", alpha=0.25, linestyle="--", color="#bbb")
    ax.tick_params(colors="#444", labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plotSpecies(modelKey, rows, outDir):
    """Gráfico com dois painéis: piores N e melhores N espécies (tema claro)."""
    color = COLORS.get(modelKey, "#444")
    worst = rows[:N_SHOW]              # acc crescente -> as piores
    best  = rows[-N_SHOW:][::-1]       # acc decrescente -> as melhores

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    painels = [
        (axes[0], worst, f"{N_SHOW} espécies com PIOR acurácia", 0.0),
        (axes[1], best,  f"{N_SHOW} espécies com MELHOR acurácia", 0.4),
    ]
    for ax, data, titulo, shade in painels:
        labels = [nameOf(c) for c, *_ in data]
        accs   = [acc * 100 for *_, acc, _ in data]
        bars   = ax.barh(labels[::-1], accs[::-1], color=lighten(color, shade), height=0.72)
        for b, v in zip(bars, accs[::-1]):
            ax.text(b.get_width() + 1, b.get_y() + b.get_height() / 2,
                    f"{v:.0f}%", va="center", fontsize=9, color="#444")
        _styleAx(ax, titulo, color)

    fig.suptitle(f"{modelKey} — acurácia por espécie (teste Oxford 102)",
                 color=color, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outPath = os.path.join(outDir, f"species_{modelKey.replace(' ', '_')}.png")
    fig.savefig(outPath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Gráfico salvo: {outPath}")


def evaluateModel(modelKey, testLoader, device, numClasses, outDir):
    ckptPath = os.path.join(CONFIG["output_dir"], f"{modelKey.replace(' ', '_')}_best.pt")
    if not os.path.exists(ckptPath):
        print(f"  [pulando] checkpoint não encontrado: {ckptPath}")
        return

    model = buildModelLikeTraining(modelKey, numClasses, device)
    model.load_state_dict(torch.load(ckptPath, map_location=device))

    cm   = confusionMatrix(model, testLoader, device, numClasses)
    rows = perClassStats(cm)
    rows.sort(key=lambda r: (r[3], r[1]))  # ordena por acurácia crescente (e suporte)

    overall = sum(r[2] for r in rows) / sum(r[1] for r in rows)
    print(f"\n{'='*70}\n  {modelKey}  —  acurácia geral no teste: {overall*100:.2f}%\n{'='*70}")

    print(f"\n  ❌ {N_SHOW} espécies que MAIS erram:")
    for c, sup, cor, acc, conf in rows[:N_SHOW]:
        extra = f"  (confunde com '{nameOf(conf)}')" if conf is not None else ""
        print(f"    {acc*100:5.1f}%  {cor:2d}/{sup:<2d}  {nameOf(c)}{extra}")

    print(f"\n  ✅ {N_SHOW} espécies que MAIS acertam:")
    for c, sup, cor, acc, conf in rows[-N_SHOW:][::-1]:
        print(f"    {acc*100:5.1f}%  {cor:2d}/{sup:<2d}  {nameOf(c)}")

    # CSV completo
    csvPath = os.path.join(outDir, f"per_species_{modelKey.replace(' ', '_')}.csv")
    with open(csvPath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classe_id_0idx", "especie", "amostras", "acertos", "acuracia_%", "mais_confundida_com"])
        for c, sup, cor, acc, conf in sorted(rows, key=lambda r: -r[3]):
            w.writerow([c, nameOf(c), sup, cor, f"{acc*100:.2f}", nameOf(conf) if conf is not None else "-"])
    print(f"\n  CSV salvo: {csvPath}")

    plotSpecies(modelKey, rows, outDir)


def main():
    device     = getDevice()
    numClasses = CONFIG["num_classes"]
    outDir     = CONFIG["output_dir"]
    os.makedirs(outDir, exist_ok=True)

    # MESMO split do treino (seed 42) — pega só o testLoader
    _, _, testLoader = loadDatasets(CONFIG)

    selected = sys.argv[1:] if len(sys.argv) > 1 else list(MODELS.keys())
    for modelKey in selected:
        if modelKey not in MODELS:
            print(f"  [aviso] modelo '{modelKey}' não está no settings.json — pulando")
            continue
        evaluateModel(modelKey, testLoader, device, numClasses, outDir)


if __name__ == "__main__":
    main()