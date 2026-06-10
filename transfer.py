"""
transfer.py — Fase 2: transferência + detecção OOD (flores x folhas) no conjunto COCO

Roda de forma automática, no mesmo padrão do ml.py:
  - lê toda a configuração de settings.json (inclusive um bloco "ood" opcional)
  - reaproveita getDevice/getTransforms/MODELS do ml.py (mesmo pré-processamento)
  - baixa o dataset COCO automaticamente do Roboflow se as credenciais estiverem
    em settings.json (análogo ao download=True do Flowers102 no ml.py)
  - detecta sozinho quais category_id são flor/folha pelos NOMES das categorias
  - carrega os checkpoints results/<Modelo>_best.pt e avalia
  - salva results/evaluation_results.json com o detalhamento completo

Uso:
    python transfer.py
"""

import os
import json
import torch
import torch.nn as nn
import timm
from PIL import Image
from rich import print
from rich.progress import (
    Progress, BarColumn, TextColumn, TimeRemainingColumn,
    MofNCompleteColumn, SpinnerColumn,
)

# Reaproveita o que já existe no ml.py (mesmo device e mesmo transform de validação)
from ml import getDevice, getTransforms, MODELS, MODEL_CONFIG, CONFIG

# ==============================================================================
# CONFIGURAÇÕES GLOBAIS  (bloco "ood" do settings.json, com defaults)
# ==============================================================================
_settings = json.load(open("settings.json", "r"))
OOD = _settings.get("ood", {})

DATASET_DIR   = OOD.get("dataset_dir",   os.path.join(CONFIG["data_dir"], "dataset2"))
SPLITS        = OOD.get("splits",        ["train", "valid", "test"])
THRESHOLD     = OOD.get("threshold",     0.5)
BBOX_EXPAND   = OOD.get("bbox_expand",   0.10)
IMG_SIZE      = CONFIG.get("img_size",   224)
NUM_CLASSES   = CONFIG.get("num_classes", 102)
OUTPUT_DIR    = CONFIG.get("output_dir", "./results")

# Palavras-chave para detectar a categoria automaticamente (não precisa editar ids)
FLOWER_WORDS  = [w.lower() for w in OOD.get("flower_keywords", ["flor", "flower", "flores"])]
LEAF_WORDS    = [w.lower() for w in OOD.get("leaf_keywords",
                 ["folha", "folhas", "hoja", "hojas", "leaf", "leaves"])]

# Credenciais opcionais do Roboflow para download automático
ROBOFLOW      = OOD.get("roboflow", {})


# ==============================================================================
# DOWNLOAD AUTOMÁTICO DO DATASET (Roboflow) — análogo ao download=True do ml.py
# ==============================================================================
def ensureDataset():
    """Garante que o dataset COCO exista localmente; baixa do Roboflow se configurado."""
    have = any(
        os.path.exists(os.path.join(DATASET_DIR, s, "_annotations.coco.json"))
        for s in SPLITS
    )
    if have:
        print(f" Dataset encontrado em: {DATASET_DIR}")
        return DATASET_DIR

    if ROBOFLOW.get("api_key") and ROBOFLOW.get("workspace") and ROBOFLOW.get("project"):
        print(" Dataset não encontrado localmente — baixando do Roboflow...")
        from roboflow import Roboflow
        rf      = Roboflow(api_key=ROBOFLOW["api_key"])
        project = rf.workspace(ROBOFLOW["workspace"]).project(ROBOFLOW["project"])
        version = int(ROBOFLOW.get("version", 1))
        dataset = project.version(version).download(
            ROBOFLOW.get("format", "coco"), location=DATASET_DIR
        )
        loc = getattr(dataset, "location", DATASET_DIR)
        print(f" Download concluído em: {loc}")
        return loc

    # Sem dados e sem credenciais: instrui o usuário e encerra
    print("[red] Dataset COCO não encontrado e Roboflow não configurado.[/red]")
    print(f"   Coloque os splits em {DATASET_DIR}/<split>/_annotations.coco.json")
    print("   OU preencha o bloco 'ood.roboflow' do settings.json com api_key/workspace/project.")
    raise SystemExit(1)


# ==============================================================================
# DETECÇÃO AUTOMÁTICA DO MAPA DE CATEGORIAS (flor=0 / folha=1)
# ==============================================================================
def buildCategoryMap(cocoCategories):
    """Lê as categorias do COCO e mapeia category_id -> 0 (flor) / 1 (folha) pelo nome."""
    catMap = {}
    for cat in cocoCategories:
        # Pula a supercategoria do Roboflow (ex.: id 0 "Deteccion-Flores",
        # cujo nome contém "flores" e enganaria o casamento por palavra-chave)
        if str(cat.get("supercategory", "")).lower() in ("", "none"):
            continue
        name = str(cat.get("name", "")).lower()
        if any(w in name for w in LEAF_WORDS):
            catMap[cat["id"]] = 1
        elif any(w in name for w in FLOWER_WORDS):
            catMap[cat["id"]] = 0
    if not catMap:
        print("[red] Não consegui detectar flor/folha pelos nomes das categorias.[/red]")
        print("   Ajuste 'ood.flower_keywords'/'ood.leaf_keywords' no settings.json.")
        raise SystemExit(1)
    return catMap


# ==============================================================================
# CARREGAMENTO DAS AMOSTRAS
# ==============================================================================
def loadSamples(datasetDir):
    samples = []
    catMapInfo = None
    for split in SPLITS:
        annFile = os.path.join(datasetDir, split, "_annotations.coco.json")
        imgDir  = os.path.join(datasetDir, split)
        if not os.path.exists(annFile):
            continue
        with open(annFile) as f:
            coco = json.load(f)
        if catMapInfo is None:
            catMapInfo = buildCategoryMap(coco["categories"])
            nomes = ", ".join(f"{cid}->{'flor' if c == 0 else 'folha'}" for cid, c in catMapInfo.items())
            print(f" Mapa de categorias detectado: {nomes}")
        idToFile = {img["id"]: img["file_name"] for img in coco["images"]}
        for ann in coco["annotations"]:
            if ann["category_id"] not in catMapInfo:
                continue
            samples.append({
                "imgDir":    imgDir,
                "filename":  idToFile[ann["image_id"]],
                "bbox":      ann["bbox"],
                "trueClass": catMapInfo[ann["category_id"]],  # 0=flor, 1=folha
            })
    nFlor  = sum(1 for s in samples if s["trueClass"] == 0)
    nFolha = sum(1 for s in samples if s["trueClass"] == 1)
    print(f" Amostras: {len(samples)} (flores={nFlor}, folhas={nFolha})\n")
    return samples


# ==============================================================================
# CARREGAMENTO DO MODELO (mesma estrutura de cabeçalho do buildModel do ml.py)
# ==============================================================================
def loadModel(modelKey, device):
    timmName = MODELS[modelKey]
    ckptPath = os.path.join(OUTPUT_DIR, f"{modelKey.replace(' ', '_')}_best.pt")
    if not os.path.exists(ckptPath):
        print(f"[red] Checkpoint não encontrado: {ckptPath} — rode o ml.py primeiro.[/red]")
        raise SystemExit(1)

    model   = timm.create_model(timmName, pretrained=False, num_classes=NUM_CLASSES)
    dropout = MODEL_CONFIG.get(modelKey, {}).get("dropout", 0.0)
    if dropout > 0.0:  # reconstrói o mesmo head usado no treino, p/ casar o state_dict
        originalClassifier = model.get_classifier()
        model.reset_classifier(0)
        model.head = nn.Sequential(nn.Dropout(p=dropout), originalClassifier)

    model.load_state_dict(torch.load(ckptPath, map_location="cpu", weights_only=True))
    return model.to(device).eval()


# ==============================================================================
# AVALIAÇÃO DE UM MODELO
# ==============================================================================
@torch.no_grad()
def evaluateModel(model, samples, transform, device):
    cm = [[0, 0], [0, 0]]   # cm[verdadeiro][predito], 0=flor 1=folha
    sampleResults = []

    with Progress(
        SpinnerColumn(), TextColumn("[bold green]  inferência[/bold green]"),
        BarColumn(), MofNCompleteColumn(), TimeRemainingColumn(), transient=True,
    ) as progress:
        task = progress.add_task("eval", total=len(samples))
        for s in samples:
            image = Image.open(os.path.join(s["imgDir"], s["filename"])).convert("RGB")
            x, y, w, h = s["bbox"]
            x1, y1 = max(0, x - w * BBOX_EXPAND), max(0, y - h * BBOX_EXPAND)
            x2, y2 = x + w * (1 + BBOX_EXPAND), y + h * (1 + BBOX_EXPAND)
            crop   = image.crop((x1, y1, x2, y2))
            tensor = transform(crop).unsqueeze(0).to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                prob = torch.softmax(model(tensor), dim=1).max().item()

            pred = 0 if prob >= THRESHOLD else 1
            true = s["trueClass"]
            cm[true][pred] += 1
            sampleResults.append({
                "filename":   s["filename"],
                "pred":       "flor" if pred == 0 else "folha",
                "true":       "flor" if true == 0 else "folha",
                "confidence": round(prob * 100, 1),
                "correct":    pred == true,
            })
            progress.update(task, advance=1)

    florTotal  = cm[0][0] + cm[0][1]
    folhaTotal = cm[1][0] + cm[1][1]
    correct    = cm[0][0] + cm[1][1]
    total      = florTotal + folhaTotal

    overall   = correct / total * 100 if total else 0.0
    florDet   = cm[0][0] / florTotal * 100 if florTotal else 0.0
    folhaRej  = cm[1][1] / folhaTotal * 100 if folhaTotal else 0.0
    balanced  = (florDet + folhaRej) / 2

    return {
        "overall":      {"correct": correct, "total": total, "accuracy": round(overall, 2)},
        "flowers":      {"n": florTotal,  "accepted": cm[0][0], "detection_rate": round(florDet, 2)},
        "leaves":       {"n": folhaTotal, "rejected": cm[1][1], "rejection_rate": round(folhaRej, 2)},
        "balanced_acc": round(balanced, 2),
        "confusion":    {"flor_flor": cm[0][0], "flor_folha": cm[0][1],
                         "folha_flor": cm[1][0], "folha_folha": cm[1][1]},
        "samples":      sampleResults,
    }


# ==============================================================================
# FUNÇÃO PRINCIPAL
# ==============================================================================
def main():
    print("\n" + "=" * 62)
    print(" Transferência de Aprendizado — Detecção OOD (flores x folhas)")
    print("=" * 62)
    print(f"  Modelos : {', '.join(MODELS.keys())}")
    print(f"  Limiar  : {THRESHOLD}")
    print(f"  Expansão: {BBOX_EXPAND*100:.0f}%\n")

    device          = getDevice()
    datasetDir      = ensureDataset()
    _, valTransform = getTransforms(IMG_SIZE)   # mesmo transform de validação do ml.py
    samples         = loadSamples(datasetDir)

    resultsByModel = {}
    for modelKey in MODELS:
        print(f"  Modelo: {modelKey}")
        model = loadModel(modelKey, device)
        res   = evaluateModel(model, samples, valTransform, device)
        resultsByModel[modelKey] = res

        ov = res["overall"]
        print(f"    Acurácia geral        : {ov['correct']}/{ov['total']} = {ov['accuracy']:.1f}%")
        print(f"    Detecção de flor       : {res['flowers']['accepted']}/{res['flowers']['n']} = {res['flowers']['detection_rate']:.1f}%")
        print(f"    Rejeição de folha (OOD): {res['leaves']['rejected']}/{res['leaves']['n']} = {res['leaves']['rejection_rate']:.1f}%")
        print(f"    Acurácia balanceada    : {res['balanced_acc']:.1f}%\n")

        del model
        torch.cuda.empty_cache()

    print("=" * 62)
    print("  RESUMO FINAL  (geral | rejeição OOD)")
    for name, r in resultsByModel.items():
        print(f"  {name:<22} {r['overall']['correct']}/{r['overall']['total']} = "
              f"{r['overall']['accuracy']:.1f}%   |   rejeição folha = {r['leaves']['rejection_rate']:.1f}%")
    print("=" * 62)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    outPath = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(outPath, "w", encoding="utf-8") as f:
        json.dump(resultsByModel, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em: {outPath}")
    print("Agora rode 'python plot.py' para gerar fig3, fig5 e tabela1.")


if __name__ == "__main__":
    main()