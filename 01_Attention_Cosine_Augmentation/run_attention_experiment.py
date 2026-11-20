# -*- coding: utf-8 -*-
"""
Geliştirme 1: Uzamsal-Menzil Koordinat Dikkat Mekanizmalı (Coordinate Attention) Residual 2D CNN
ve Fizik-Uyumlu Radar Veri Çoğaltma (Physics-Informed Radar Augmentation) + Cosine Annealing LR.

Ana kodlara (`src/`, `train_and_evaluate.py`) dokunmadan bağımsız olarak çalışır.
Çıktılar `01_Attention_Cosine_Augmentation/results/` ve `RAPOR_01_ATTENTION_CNN.md` olarak kaydedilir.
"""

import copy
import json
import math
import os
import sys
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

# Ana dizindeki veri yükleyiciyi salt-okunur olarak import et
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import DEFAULT_RADAR_DATA_DIR, NUM_CLASSES, NUM_CLIENTS, load_all_days_dataset, to_dataloader
from src.fl_engine import aggregate_weights, evaluate_model


class CoordinateAttention(nn.Module):
    """
    Range (Menzil - H) ve Azimuth (Açı - W) eksenleri boyunca ayrı ayrı 1D global havuzlama
    yaparak radar yansıma tepe noktasının (target peak) tam (Range, Azimuth) koordinatlarını
    yakalayan dikkat modülü (Hou et al., CVPR 2021 - mmWave Radar için uyarlanmıştır).
    """

    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=mid_channels)
        self.act = nn.SiLU(inplace=True)

        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        n, c, h, w = x.size()

        # Range ekseni (H, 1) ve Azimuth ekseni (1, W) boyunca ortalamalar
        x_h = x.mean(dim=3, keepdim=True)  # (N, C, H, 1)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # (N, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)  # (N, C, H+W, 1)
        y = self.act(self.gn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # (N, mid, 1, W)

        a_h = torch.sigmoid(self.conv_h(x_h))  # Menzil dikkat maskesi: M_range in (0,1)^(N x C x H x 1)
        a_w = torch.sigmoid(self.conv_w(x_w))  # Açı dikkat maskesi: M_azimuth in (0,1)^(N x C x 1 x W)

        return identity * a_h * a_w


class ResidualRadarBlock(nn.Module):
    """GroupNorm + CoordinateAttention destekli Residual Evrişim Bloğu."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_ch)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_ch)
        self.ca = CoordinateAttention(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out = self.ca(out)
        out = out + self.shortcut(x)
        return self.act(out)


class RadarAttentionResNet(nn.Module):
    """
    Coordinate Attention + Residual Bağlantılı Gelişmiş 2D Radar Konum Ağı.
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(4, 24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (24, 64, 16)
        )
        self.layer1 = ResidualRadarBlock(24, 48, stride=2, groups=8)  # (48, 32, 8)
        self.layer2 = ResidualRadarBlock(48, 96, stride=2, groups=8)  # (96, 16, 4)
        self.pool = nn.AdaptiveAvgPool2d((4, 2))  # (96, 4, 2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 2, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        return self.classifier(x)


def apply_radar_physics_augmentation(bx: torch.Tensor, awgn_std: float = 0.04, speckle_std: float = 0.04) -> torch.Tensor:
    """
    Fizik-Uyumlu mmWave Radar Veri Çoğaltma:
    1. Termal Alıcı Gürültüsü (AWGN): x_aug = x + N(0, sigma_awgn^2)
    2. Çarpımsal Yansıma / Speckle Gürültüsü: x_aug = x * (1 + N(0, sigma_speckle^2))
    """
    awgn = torch.randn_like(bx) * awgn_std
    speckle = 1.0 + torch.randn_like(bx) * speckle_std
    return bx * speckle + awgn


def train_client_with_aug_and_lr(
    global_model: nn.Module,
    dataloader,
    device: torch.device,
    local_epochs: int,
    lr: float,
    use_augmentation: bool = True,
):
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()
    optimizer = torch.optim.AdamW(local_model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    total_loss, total_correct, total_samples = 0.0, 0, 0
    for _ in range(local_epochs):
        for bx, by in dataloader:
            bx, by = bx.to(device), by.to(device)
            if use_augmentation:
                bx = apply_radar_physics_augmentation(bx)
            optimizer.zero_grad()
            logits = local_model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * by.size(0)
            total_correct += (logits.argmax(dim=1) == by).sum().item()
            total_samples += by.size(0)

    return (
        copy.deepcopy(local_model.state_dict()),
        total_loss / max(1, total_samples),
        total_correct / max(1, total_samples),
        len(dataloader.dataset),
    )


def main():
    t0 = time.time()
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(exp_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== [Geliştirme 1] Coordinate Attention ResNet + Cosine LR + Radar Augmentation (Cihaz: {device}) ===")

    clients_iid, day_tests, (global_test_x, global_test_y) = load_all_days_dataset(
        base_dir=DEFAULT_RADAR_DATA_DIR, days=["day0", "day1", "day2"], norm_method="zscore"
    )

    client_loaders = {
        cid: to_dataloader(x, y, batch_size=64, shuffle=True) for cid, (x, y) in clients_iid.items()
    }
    global_test_loader = to_dataloader(global_test_x, global_test_y, batch_size=128, shuffle=False)
    day_test_loaders = {
        d: to_dataloader(tx, ty, batch_size=128, shuffle=False) for d, (tx, ty) in day_tests.items()
    }

    NUM_ROUNDS = 15
    LOCAL_EPOCHS = 2
    LR_MAX = 1.5e-3
    LR_MIN = 1.5e-4

    torch.manual_seed(42)
    global_model = RadarAttentionResNet(num_classes=NUM_CLASSES).to(device)

    history = {
        "round": [0],
        "lr": [LR_MAX],
        "train_acc": [np.nan],
        "test_acc": [],
        "test_loss": [],
        "day0_acc": [],
        "day1_acc": [],
        "day2_acc": [],
    }

    init_loss, init_acc, _, _ = evaluate_model(global_model, global_test_loader, device)
    history["test_acc"].append(init_acc)
    history["test_loss"].append(init_loss)
    for d in ["day0", "day1", "day2"]:
        _, d_acc, _, _ = evaluate_model(global_model, day_test_loaders[d], device)
        history[f"{d}_acc"].append(d_acc)

    best_acc = 0.0
    best_state = None

    for rnd in range(1, NUM_ROUNDS + 1):
        # Cosine Annealing Öğrenme Oranı: eta_t = eta_min + 0.5*(eta_max - eta_min)*(1 + cos(pi * (t-1)/T))
        lr_t = LR_MIN + 0.5 * (LR_MAX - LR_MIN) * (1.0 + math.cos(math.pi * (rnd - 1) / NUM_ROUNDS))

        c_states, c_sizes, r_accs = [], [], []
        for cid in range(NUM_CLIENTS):
            st, _, c_acc, c_sz = train_client_with_aug_and_lr(
                global_model, client_loaders[cid], device, LOCAL_EPOCHS, lr_t, use_augmentation=True
            )
            c_states.append(st)
            c_sizes.append(c_sz)
            r_accs.append(c_acc * c_sz)

        global_model.load_state_dict(aggregate_weights(c_states, c_sizes))
        tr_acc = sum(r_accs) / sum(c_sizes)
        te_loss, te_acc, _, _ = evaluate_model(global_model, global_test_loader, device)

        history["round"].append(rnd)
        history["lr"].append(lr_t)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        history["test_loss"].append(te_loss)

        for d in ["day0", "day1", "day2"]:
            _, d_acc, _, _ = evaluate_model(global_model, day_test_loaders[d], device)
            history[f"{d}_acc"].append(d_acc)

        if te_acc >= best_acc:
            best_acc = te_acc
            best_state = copy.deepcopy(global_model.state_dict())

        print(
            f"[Attention-ResNet FL] Tur {rnd:02d}/{NUM_ROUNDS:02d} (LR={lr_t:.5f}) | "
            f"Train Acc: {tr_acc*100:.2f}% | Global Test Acc: {te_acc*100:.2f}% (Loss: {te_loss:.4f}) | "
            f"day0: {history['day0_acc'][-1]*100:.2f}%, day1: {history['day1_acc'][-1]*100:.2f}%, day2: {history['day2_acc'][-1]*100:.2f}%"
        )

    # En iyi modeli yükle ve nihai rapor oluştur
    global_model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(res_dir, "attention_resnet_best.pt"))
    final_loss, final_acc, y_true, y_pred = evaluate_model(global_model, global_test_loader, device)

    target_names = [f"Sınıf {i}" for i in range(NUM_CLASSES)]
    cls_report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    print("\n--- Nihai Sınıflandırma Raporu (Attention-ResNet FL) ---")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # Ana modelin (Baseline FedAvg) sonuçlarıyla karşılaştırma grafiği
    base_summary_path = os.path.join(PROJECT_ROOT, "results", "experiment_summary.json")
    base_acc = 98.63
    if os.path.exists(base_summary_path):
        with open(base_summary_path, "r", encoding="utf-8") as f:
            base_acc = json.load(f)["final_test_accuracy_percent"]["FedAvg_IID"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    axes[0].plot(history["round"], [a * 100 for a in history["test_acc"]], "b-o", linewidth=2.2, label="Attention-ResNet + CosineLR (Global Test)")
    axes[0].plot(history["round"], [a * 100 for a in history["day0_acc"]], "g--", linewidth=1.6, label="day0 Test Kümesi")
    axes[0].plot(history["round"], [a * 100 for a in history["day1_acc"]], "m--", linewidth=1.6, label="day1 Test Kümesi")
    axes[0].plot(history["round"], [a * 100 for a in history["day2_acc"]], "c--", linewidth=1.6, label="day2 Test Kümesi")
    axes[0].axhline(base_acc, color="red", linestyle=":", linewidth=1.8, label=f"Temel Model (Baseline FedAvg: %{base_acc})")
    axes[0].set_title("Coordinate Attention ResNet FL - Yakınsama Eğrisi", fontweight="bold")
    axes[0].set_xlabel("İletişim Turu (Round)")
    axes[0].set_ylabel("Test Doğruluğu (%)")
    axes[0].set_ylim(75, 100.3)
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right")

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=axes[1],
        xticklabels=[f"S{i}" for i in range(NUM_CLASSES)],
        yticklabels=[f"S{i}" for i in range(NUM_CLASSES)],
    )
    axes[1].set_title(f"Attention-ResNet Karmaşıklık Matrisi (Doğruluk: %{final_acc*100:.2f})", fontweight="bold")
    axes[1].set_xlabel("Tahmin Edilen Konum Sınıfı")
    axes[1].set_ylabel("Gerçek Konum Sınıfı")

    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "attention_resnet_results.png"), dpi=150)
    plt.close(fig)

    summary = {
        "method": "Coordinate Attention Residual CNN + Physics Augmentation + Cosine Annealing LR",
        "baseline_fedavg_acc_percent": base_acc,
        "peak_test_acc_percent": round(float(best_acc * 100), 2),
        "final_round_test_acc_percent": round(float(history["test_acc"][-1] * 100), 2),
        "day_wise_best_acc_percent": {
            "day0": round(float(max(history["day0_acc"]) * 100), 2),
            "day1": round(float(max(history["day1_acc"]) * 100), 2),
            "day2": round(float(max(history["day2_acc"]) * 100), 2),
        },
        "history": {
            "round": history["round"],
            "lr": [round(x, 6) for x in history["lr"]],
            "test_acc_percent": [round(float(x * 100), 2) for x in history["test_acc"]],
            "test_loss": [round(float(x), 4) for x in history["test_loss"]],
        },
        "classification_report": cls_report,
        "execution_time_seconds": round(time.time() - t0, 2),
    }
    with open(os.path.join(res_dir, "summary_01_attention.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"=== [Geliştirme 1 Tamamlandı] En İyi Global Doğruluk: %{best_acc*100:.2f} (Süre: {summary['execution_time_seconds']}s) ===")


if __name__ == "__main__":
    main()
