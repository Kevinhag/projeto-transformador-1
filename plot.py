"""
plot.py — Gera as figuras separadas do artigo (tema claro), lendo dos JSONs reais.

Fontes de dados:
  results/metrics.json            (gerado por ml.py)      -> Val/Test/Tempo Oxford
  results/history.json            (gerado por ml.py)      -> curvas por época
  results/evaluation_results.json (gerado por transfer.py)-> OOD (opcional)

Figuras geradas em results/:
  fig1_curvas_perda.png        loss treino/val por época (3 painéis)
  fig2_curvas_acuracia.png     acurácia treino/val por época (3 painéis)
  fig4_tempo_treino.png        tempo de treino por modelo
  fig3_comparativo_acuracia.png  Val/Test Oxford + rejeição OOD   (requer evaluation_results.json)
  fig5_ood_resultado.png       acertos OOD (barras + pizzas)      (requer evaluation_results.json)
  tabela1_resumo.png           tabela resumo consolidada          (requer evaluation_results.json)

Uso:
  python plot.py                      # lê de ./results
  python plot.py /caminho/results     # lê de outra pasta
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

# ---- cores por modelo (tema claro) ----
COLORS = {
    "MobileNetV3-Large": "#2EADC1",
    "EfficientViT-M4":   "#E67E49",
    "Swin-Tiny":         "#844BDF",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "savefig.facecolor": "white",
    "font.size": 11,
    "axes.edgecolor": "#cccccc",
})


def lighten(hex_color, amount):
    """Clareia uma cor misturando com branco. amount in [0,1]."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * amount)
    g = int(g + (255 - g) * amount)
    b = int(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def _style(ax, title=None, xlabel=None, ylabel=None, color="#222"):
    if title:
        ax.set_title(title, color=color, fontsize=13, fontweight="bold", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color="#444")
    if ylabel:
        ax.set_ylabel(ylabel, color="#444")
    ax.grid(alpha=0.25, linestyle="--", color="#bbb")
    ax.tick_params(colors="#444")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ==============================================================================
# FIG 1 e 2 — curvas por época (3 painéis lado a lado)
# ==============================================================================
def plotCurves(history, outDir, key, ylabel, title_metric, scale=1.0, fname=""):
    names  = list(history.keys())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, name in zip(axes, names):
        h      = history[name]
        epochs = range(1, len(h[f"train{key}"]) + 1)
        c      = COLORS.get(name, "#333")
        tr = [v * scale for v in h[f"train{key}"]]
        vl = [v * scale for v in h[f"val{key}"]]
        ax.plot(epochs, tr, color=c, lw=2.2, label="Treino")
        ax.plot(epochs, vl, color=c, lw=2.2, ls="--", label="Validação")
        _style(ax, title=name, xlabel="Época", ylabel=ylabel, color=c)
        ax.legend(frameon=False, loc="best", fontsize=10)
    fig.tight_layout()
    out = os.path.join(outDir, fname)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  salvo: {out}")


# ==============================================================================
# FIG 4 — tempo de treino (barras)
# ==============================================================================
def plotTime(metrics, outDir):
    names = list(metrics.keys())
    times = [metrics[n]["totalTime"] / 60 for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    bars = ax.bar([n.replace("-", "\n") for n in names], times,
                  color=[COLORS[n] for n in names], width=0.55)
    for b, v in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(times) * 0.01,
                f"{v:.1f} min", ha="center", va="bottom", fontweight="bold", fontsize=12)
    _style(ax, ylabel="Tempo (min)")
    ax.set_ylim(0, max(times) * 1.18)
    fig.tight_layout()
    out = os.path.join(outDir, "fig4_tempo_treino.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  salvo: {out}")


# ==============================================================================
# FIG 3 — comparativo Val/Test Oxford + rejeição OOD (barras agrupadas)
# ==============================================================================
def plotComparativo(metrics, evals, outDir):
    names = list(metrics.keys())
    x     = np.arange(len(names))
    w     = 0.26
    val   = [metrics[n]["bestValAcc"] * 100 for n in names]
    test  = [metrics[n]["testAcc"]    * 100 for n in names]
    ood   = [evals[n]["leaves"]["rejection_rate"] for n in names]

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for i, n in enumerate(names):
        c = COLORS[n]
        ax.bar(x[i] - w, val[i],  w, color=lighten(c, 0.45),
               label="Val. Oxford (%)" if i == 0 else "")
        ax.bar(x[i],     test[i], w, color=c,
               label="Test Oxford (%)" if i == 0 else "")
        ax.bar(x[i] + w, ood[i],  w, color=lighten(c, 0.25), hatch="//", edgecolor="white",
               label="Rejeição OOD — folhas (%)" if i == 0 else "")
        for xi, v in [(x[i] - w, val[i]), (x[i], test[i]), (x[i] + w, ood[i])]:
            ax.text(xi, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("-", "\n") for n in names])
    _style(ax, ylabel="Acurácia / Taxa (%)")
    ax.set_ylim(0, 115)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08),
              ncol=3, fontsize=10)
    fig.tight_layout()
    out = os.path.join(outDir, "fig3_comparativo_acuracia.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  salvo: {out}")


# ==============================================================================
# FIG 5 — resultado OOD: barras horizontais (acertos) + pizzas
# ==============================================================================
def plotOOD(evals, outDir):
    names   = list(evals.keys())
    correct = [evals[n]["overall"]["correct"] for n in names]
    total   = evals[names[0]]["overall"]["total"]
    accs    = [evals[n]["overall"]["accuracy"] for n in names]

    fig = plt.figure(figsize=(15, 5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0], wspace=0.25)

    # esquerda: barras horizontais
    ax1 = fig.add_subplot(gs[0])
    y   = np.arange(len(names))
    bars = ax1.barh(y, correct, color=[COLORS[n] for n in names], height=0.6)
    for b, c, a in zip(bars, correct, accs):
        ax1.text(b.get_width() + total * 0.01, b.get_y() + b.get_height() / 2,
                 f"{c}/{total}  ({a:.1f}%)", va="center", fontweight="bold", fontsize=11)
    ax1.axvline(total, ls="--", color="#888", lw=1.3, label=f"Total ({total})")
    ax1.set_yticks(y)
    ax1.set_yticklabels(names)
    ax1.set_xlim(0, total * 1.25)
    _style(ax1, title="Acertos na Triagem OOD", xlabel="Imagens classificadas corretamente")
    ax1.legend(frameon=False, loc="lower right")

    # direita: pizzas
    ax2 = fig.add_subplot(gs[1])
    ax2.axis("off")
    ax2.set_title("Acurácia OOD por Modelo (%)", fontsize=13, fontweight="bold", color="#222", pad=10)
    for i, n in enumerate(names):
        sub = fig.add_axes([0.60 + i * 0.135, 0.30, 0.12, 0.40])
        a   = accs[i]
        sub.pie([a, 100 - a], colors=[COLORS[n], lighten(COLORS[n], 0.78)],
                startangle=90, counterclock=False, wedgeprops={"linewidth": 0})
        sub.text(0, -1.45, f"{a:.1f}%", ha="center", fontweight="bold", color=COLORS[n], fontsize=12)
        sub.text(0, 1.35, n.replace("-", "\n"), ha="center", fontweight="bold", color=COLORS[n], fontsize=9)

    out = os.path.join(outDir, "fig5_ood_resultado.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  salvo: {out}")


# ==============================================================================
# TABELA 1 — resumo consolidado
# ==============================================================================
def plotResumo(metrics, evals, outDir):
    names = list(metrics.keys())
    cols  = ["Modelo", "Val. Oxford\n(%)", "Test Oxford\n(%)",
             "Rejeição OOD\nfolhas (%)", "Tempo\n(min)", "Acurácia OOD\ngeral (n=%d)" % evals[names[0]]["overall"]["total"]]
    rows  = []
    for n in names:
        ov = evals[n]["overall"]
        rows.append([
            n,
            f"{metrics[n]['bestValAcc']*100:.2f}",
            f"{metrics[n]['testAcc']*100:.2f}",
            f"{evals[n]['leaves']['rejection_rate']:.2f}",
            f"{metrics[n]['totalTime']/60:.1f}",
            f"{ov['correct']} ({ov['accuracy']:.1f}%)",
        ])

    fig, ax = plt.subplots(figsize=(13, 2.4))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)
    for (r, cc), cell in tbl.get_celld().items():
        cell.set_edgecolor("#999")
        if r == 0:
            cell.set_facecolor("#222")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            n = names[r - 1]
            cell.set_facecolor(lighten(COLORS[n], 0.82))
    fig.tight_layout()
    out = os.path.join(outDir, "tabela1_resumo.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  salvo: {out}")


# ==============================================================================
def main():
    outDir = sys.argv[1] if len(sys.argv) > 1 else "results"
    os.makedirs(outDir, exist_ok=True)

    metrics = json.load(open(os.path.join(outDir, "metrics.json")))
    history = json.load(open(os.path.join(outDir, "history.json")))

    print("Gerando figuras do Oxford (fase 1)...")
    plotCurves(history, outDir, "Loss", "Loss", "perda",
               scale=1.0, fname="fig1_curvas_perda.png")
    plotCurves(history, outDir, "Acc", "Acurácia (%)", "acurácia",
               scale=100.0, fname="fig2_curvas_acuracia.png")
    plotTime(metrics, outDir)

    evalPath = os.path.join(outDir, "evaluation_results.json")
    if os.path.exists(evalPath):
        evals = json.load(open(evalPath))
        print("Gerando figuras OOD (fase 2)...")
        plotComparativo(metrics, evals, outDir)
        plotOOD(evals, outDir)
        plotResumo(metrics, evals, outDir)
    else:
        print(f"\n[aviso] {evalPath} não encontrado — rode transfer.py para gerar "
              f"fig3, fig5 e tabela1.")

    print("\nConcluído.")


if __name__ == "__main__":
    main()
