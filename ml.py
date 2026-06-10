import os
import time
import json
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision import transforms
from torchvision.datasets import Flowers102
from rich import print
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)

# ==============================================================================
# CONFIGURAÇÕES GLOBAIS
# ==============================================================================
_settings    = json.load(open("settings.json", "r"))
CONFIG       = _settings["config"]
MODELS       = _settings["models"]
MODEL_CONFIG = _settings.get("modelConfig", {})  # configs específicas por modelo (opcional)


# ==============================================================================
# DEVICE
# ==============================================================================
def getDevice():
    if torch.cuda.is_available():
        device  = torch.device("cuda")
        gpuName = torch.cuda.get_device_name(0)
        vRam    = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpuName} - {vRam:.1f} GB VRAM")
    else:
        device = torch.device("cpu")
        print("Rodando em CPU")
    return device


# ==============================================================================
# TRANSFORMS
# ==============================================================================
def getTransforms(imgSize=224):
    # Média e desvio padrão por canal RGB calculados no ImageNet
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    # Pipeline de treino: augmentations aleatórias para aumentar diversidade
    trainTransforms = transforms.Compose([
        transforms.RandomResizedCrop(imgSize, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Pipeline de validação/teste: sem aleatoriedade — avaliação consistente
    valTransforms = transforms.Compose([
        transforms.Resize(int(imgSize * 1.14)),
        transforms.CenterCrop(imgSize),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return trainTransforms, valTransforms


# ==============================================================================
# WRAPPER DE DATASET
# ==============================================================================
class SubsetWithTransform(torch.utils.data.Dataset):
    """
    Aplica um transform a um Subset retornado pelo random_split.

    Precisa estar no escopo global (fora de funções) para ser
    serializável pelo multiprocessing no Python 3.14+.
    """
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# ==============================================================================
# DATASET — redistribuição 70 / 15 / 15
# ==============================================================================
def loadDatasets(cfg):
    """
    O split oficial do Oxford 102 Flowers é problemático:
        - Treino:    1.020 imagens (10 por classe) — muito pouco
        - Validação: 1.020 imagens (10 por classe) — muito pouco
        - Teste:     6.149 imagens                 — grande demais

    Estratégia adotada:
        1. Baixa os 3 splits originais sem transform nenhum
        2. Junta tudo em um único dataset de 8.189 imagens com ConcatDataset
        3. Divide aleatoriamente em 70% / 15% / 15%
        4. Aplica os transforms corretos em cada parte usando um wrapper
    """
    trainTransforms, valTransforms = getTransforms(cfg["img_size"])

    rawTrain = Flowers102(root=cfg["data_dir"], split="train", transform=None, download=True)
    rawVal   = Flowers102(root=cfg["data_dir"], split="val",   transform=None, download=True)
    rawTest  = Flowers102(root=cfg["data_dir"], split="test",  transform=None, download=True)

    totalImages = len(rawTrain) + len(rawVal) + len(rawTest)
    print(f" Imagens no dataset: {totalImages}")

    fullDataset = ConcatDataset([rawTrain, rawVal, rawTest])

    trainSize = int(0.70 * totalImages)
    valSize   = int(0.15 * totalImages)
    testSize  = totalImages - trainSize - valSize

    print(f" Redistribuição: Train {trainSize} | Val {valSize} | Test {testSize}")

    generator = torch.Generator().manual_seed(cfg["seed"])
    trainSubset, valSubset, testSubset = random_split(
        fullDataset, [trainSize, valSize, testSize], generator=generator
    )

    trainDataset = SubsetWithTransform(trainSubset, trainTransforms)
    valDataset   = SubsetWithTransform(valSubset,   valTransforms)
    testDataset  = SubsetWithTransform(testSubset,  valTransforms)

    trainLoader = DataLoader(trainDataset, batch_size=cfg["batch_size"], shuffle=True,
                             num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    valLoader   = DataLoader(valDataset,   batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=True)
    testLoader  = DataLoader(testDataset,  batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=True)

    return trainLoader, valLoader, testLoader


# ==============================================================================
# CONSTRUÇÃO DO MODELO
# ==============================================================================
def buildModel(modelKey, numClasses, device, dropout=0.0):
    """
    Carrega um modelo pré-treinado no ImageNet via timm e adapta para 102 classes.

    Args:
        dropout: se > 0, injeta uma camada Dropout antes do classificador final.
                 Útil para modelos com tendência a overfitting (ex: EfficientViT).
    """
    timmName = MODELS[modelKey]
    print(f"\n   Carregando {modelKey}  ({timmName})")

    model = timm.create_model(timmName, pretrained=True, num_classes=numClasses)

    # Injeta Dropout antes do classificador se solicitado
    # Dropout desativa aleatoriamente `dropout`% dos neurônios durante o treino,
    # forçando o modelo a não depender de nenhum neurônio específico — regularização forte
    if dropout > 0.0:
        # timm expõe o classificador final como model.classifier ou model.head
        # dependendo da arquitetura — get_classifier() é o método universal
        originalClassifier = model.get_classifier()
        model.reset_classifier(0)  # remove o classificador original

        # Reconstrói: Dropout → camada linear original
        model.head = nn.Sequential(
            nn.Dropout(p=dropout),
            originalClassifier,
        )
        print(f"   Dropout {dropout} adicionado antes do classificador")

    model = model.to(device)

    totalParams     = sum(p.numel() for p in model.parameters())
    trainableParams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" Parâmetros: {totalParams/1e6:.2f} M total | {trainableParams/1e6:.2f} M treináveis")

    return model


# ==============================================================================
# MIXUP — augmentation que mistura dois exemplos aleatórios
# ==============================================================================
def mixupBatch(imgs, labels, numClasses, alpha):
    """
    Mixup: cria um novo exemplo interpolando dois exemplos aleatórios.

        img_novo   = lambda * img_a   + (1 - lambda) * img_b
        label_novo = lambda * label_a + (1 - lambda) * label_b

    Onde lambda ~ Beta(alpha, alpha).

    Isso força o modelo a aprender representações suaves e contínuas entre
    classes, em vez de fronteiras rígidas — reduz overfitting significativamente
    em datasets pequenos e modelos com poucos parâmetros como o EfficientViT.

    Args:
        imgs:       batch de imagens  (batch, C, H, W)
        labels:     batch de rótulos  (batch,) — inteiros
        numClasses: total de classes para converter para one-hot
        alpha:      parâmetro da distribuição Beta (0.4 é um bom valor inicial)

    Returns:
        mixedImgs:   imagens misturadas
        mixedLabels: labels suavizados em formato one-hot (batch, numClasses)
    """
    # Sorteia lambda da distribuição Beta — valores próximos de 0.5 misturam mais
    lam = np.random.beta(alpha, alpha)

    batchSize = imgs.size(0)

    # Embaralha os índices do batch para parear exemplos aleatórios
    shuffledIdx = torch.randperm(batchSize, device=imgs.device)

    # Interpola as imagens
    mixedImgs = lam * imgs + (1 - lam) * imgs[shuffledIdx]

    # Converte labels para one-hot e interpola
    labelsOneHot         = torch.zeros(batchSize, numClasses, device=imgs.device)
    shuffledLabels       = labels[shuffledIdx]
    labelsOneHot.scatter_(1, labels.unsqueeze(1), 1)

    shuffledOneHot = torch.zeros(batchSize, numClasses, device=imgs.device)
    shuffledOneHot.scatter_(1, shuffledLabels.unsqueeze(1), 1)

    mixedLabels = lam * labelsOneHot + (1 - lam) * shuffledOneHot

    return mixedImgs, mixedLabels


# ==============================================================================
# LOOP DE UMA ÉPOCA DE TREINO
# ==============================================================================
def trainEpoch(model, loader, criterion, optimizer, scheduler, device, scaler,
               useMixup=False, mixupAlpha=0.4, numClasses=102):
    """
    Executa uma época completa: forward → loss → backward → update.

    Args:
        useMixup:   se True, aplica Mixup nos batches de treino
        mixupAlpha: parâmetro alpha do Mixup (força da mistura)
        numClasses: necessário para criar os labels one-hot do Mixup
    """
    model.train()

    runningLoss = 0.0
    correct     = 0
    total       = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]  treino[/bold green]"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[yellow]{task.fields[loss]}[/yellow]"),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("treino", total=len(loader), loss="loss: -.----")

        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):

                if useMixup:
                    # Aplica Mixup: mistura pares de imagens e labels
                    mixedImgs, mixedLabels = mixupBatch(imgs, labels, numClasses, mixupAlpha)
                    outputs = model(mixedImgs)
                    # Com labels suavizados, a loss é calculada como produto escalar
                    # entre a distribuição predita (softmax) e o label interpolado
                    logProbs = torch.nn.functional.log_softmax(outputs, dim=1)
                    loss     = -(mixedLabels * logProbs).sum(dim=1).mean()
                else:
                    outputs = model(imgs)
                    loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            runningLoss += loss.item() * imgs.size(0)
            _, preds = outputs.max(dim=1)
            correct  += preds.eq(labels).sum().item()
            total    += imgs.size(0)

            progress.update(task, advance=1, loss=f"loss: {loss.item():.3f}")

    scheduler.step()

    return runningLoss / total, correct / total


# ==============================================================================
# AVALIAÇÃO (VALIDAÇÃO / TESTE)
# ==============================================================================
@torch.no_grad()
def evalEpoch(model, loader, criterion, device):
    """
    Avalia o modelo sem atualizar os pesos.
    Usado para validação (monitorar overfitting) e teste final (métrica real).
    """
    model.eval()

    runningLoss = 0.0
    correct     = 0
    total       = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]   eval [/bold blue]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("eval", total=len(loader))

        for imgs, labels in loader:
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss    = criterion(outputs, labels)

            runningLoss += loss.item() * imgs.size(0)
            _, preds = outputs.max(dim=1)
            correct  += preds.eq(labels).sum().item()
            total    += imgs.size(0)

            progress.update(task, advance=1)

    return runningLoss / total, correct / total


# ==============================================================================
# LOOP COMPLETO DE TREINO DE UM MODELO
# ==============================================================================
def trainModel(modelKey, trainLoader, valLoader, cfg, device):
    """
    Orquestra o treinamento completo.
    Mescla as configs globais com as configs específicas do modelo (se existirem).
    """
    print(f"\n{'='*62}")
    print(f"  Iniciando treino: {modelKey}")
    print(f"{'='*62}")

    # Mescla config global com config específica do modelo
    # Valores do modelConfig sobrescrevem os do CONFIG global
    modelCfg = {**cfg, **MODEL_CONFIG.get(modelKey, {})}

    # Extrai parâmetros específicos de regularização
    dropout        = modelCfg.get("dropout", 0.0)
    labelSmoothing = modelCfg.get("label_smoothing", 0.1)
    useMixup       = "mixup_alpha" in MODEL_CONFIG.get(modelKey, {})
    mixupAlpha     = modelCfg.get("mixup_alpha", 0.4)

    # Exibe aviso se o modelo tem config customizada
    if modelKey in MODEL_CONFIG:
        overrides = MODEL_CONFIG[modelKey]
        print(f"   Config customizada: {overrides}")

    model = buildModel(modelKey, modelCfg["num_classes"], device, dropout=dropout)

    # CrossEntropyLoss — label_smoothing pode ser maior para modelos com overfitting
    criterion = nn.CrossEntropyLoss(label_smoothing=labelSmoothing)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=modelCfg["lr"],
        weight_decay=modelCfg["weight_decay"]
    )

    warmupEpochs    = modelCfg.get("warmup_epochs", 5)
    cosineEpochs    = modelCfg["num_epochs"] - warmupEpochs
    warmupScheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmupEpochs
    )
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosineEpochs)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmupScheduler, cosineScheduler], milestones=[warmupEpochs]
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    history = {
        "trainLoss": [], "trainAcc": [],
        "valLoss":   [], "valAcc":   [],
    }

    bestValAcc = 0.0
    bestState  = None
    t0         = time.time()

    for epoch in range(1, modelCfg["num_epochs"] + 1):
        epStart = time.time()

        trLoss, trAcc = trainEpoch(
            model, trainLoader, criterion, optimizer, scheduler, device, scaler,
            useMixup=useMixup, mixupAlpha=mixupAlpha, numClasses=modelCfg["num_classes"]
        )
        vlLoss, vlAcc = evalEpoch(model, valLoader, criterion, device)

        history["trainLoss"].append(trLoss)
        history["trainAcc"].append(trAcc)
        history["valLoss"].append(vlLoss)
        history["valAcc"].append(vlAcc)

        epTime = time.time() - epStart
        marker = " ⭐" if vlAcc > bestValAcc else ""

        print(
            f"  Época {epoch:02d}/{modelCfg['num_epochs']}  |  "
            f"Train loss {trLoss:.4f}  acc {trAcc:.4f}  |  "
            f"Val loss {vlLoss:.4f}  acc {vlAcc:.4f}  |  "
            f"{epTime:.1f}s{marker}"
        )

        if vlAcc > bestValAcc:
            bestValAcc = vlAcc
            bestState  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    totalTime = time.time() - t0
    print(f"\n  Treino concluído | Melhor Val Acc: {bestValAcc:.4f} | Tempo: {totalTime:.1f}s")

    model.load_state_dict(bestState)

    return model, history, bestValAcc, totalTime


# ==============================================================================
# FUNÇÃO PRINCIPAL
# ==============================================================================
def main():
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    print("\n" + "="*62)
    print(" Oxford 102 Flowers — ML Benchmark")
    print("="*62)
    print(f"  Modelos : {', '.join(MODELS.keys())}")
    print(f"  Épocas  : {CONFIG['num_epochs']}")
    print(f"  Batch   : {CONFIG['batch_size']}")
    print(f"  LR      : {CONFIG['lr']}")
    print(f"  Img size: {CONFIG['img_size']}x{CONFIG['img_size']}")

    device = getDevice()
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    trainLoader, valLoader, testLoader = loadDatasets(CONFIG)

    allResults = {}

    for modelKey in MODELS:
        model, history, bestValAcc, totalTime = trainModel(
            modelKey, trainLoader, valLoader, CONFIG, device
        )

        print(f"  Avaliando no Test set...")
        _, testAcc = evalEpoch(model, testLoader, nn.CrossEntropyLoss(), device)
        print(f"  Test Accuracy: {testAcc * 100:.2f}%")

        allResults[modelKey] = {
            "history":    history,
            "bestValAcc": bestValAcc,
            "testAcc":    testAcc,
            "totalTime":  totalTime,
        }

        ckptPath = os.path.join(CONFIG["output_dir"], f"{modelKey.replace(' ', '_')}_best.pt")
        torch.save(model.state_dict(), ckptPath)
        print(f"  Checkpoint salvo: {ckptPath}")

        del model
        torch.cuda.empty_cache()

    metricsPath = os.path.join(CONFIG["output_dir"], "metrics.json")
    with open(metricsPath, "w") as f:
        def toPy(obj):
            if isinstance(obj, (np.float32, np.float64)): return float(obj)
            if isinstance(obj, (np.int32,   np.int64)):   return int(obj)
            return obj
        json.dump(
            {
                modelKey: {
                    key: ([toPy(v) for v in val] if isinstance(val, list) else toPy(val))
                    for key, val in results.items()
                    if key != "history"
                }
                for modelKey, results in allResults.items()
            },
            f, indent=2
        )
    print(f"\nMétricas salvas em: {metricsPath}")

    historyPath = os.path.join(CONFIG["output_dir"], "history.json")
    with open(historyPath, "w") as f:
        json.dump(
            {modelKey: results["history"] for modelKey, results in allResults.items()},
            f, indent=2
        )
    print(f"Histórico salvo em: {historyPath}")

    print("\n" + "="*62)
    print("  RESULTADO FINAL")
    print("="*62)
    bestModel = max(allResults, key=lambda k: allResults[k]["testAcc"])
    for name, res in allResults.items():
        marker = " ← 🏆" if name == bestModel else ""
        print(
            f"  {name:<22}  "
            f"Val {res['bestValAcc']*100:.2f}%  |  "
            f"Test {res['testAcc']*100:.2f}%  |  "
            f"{res['totalTime']/60:.1f} min"
            f"{marker}"
        )
    print("="*62)
    print("\nConcluído! Resultados em:", CONFIG["output_dir"])


if __name__ == "__main__":
    main()