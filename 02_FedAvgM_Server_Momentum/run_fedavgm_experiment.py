# -*- coding: utf-8 -*-
"""
Geliştirme 2: Sunucu Momentumlu Federe Öğrenme (FedAvgM - Hsu et al., 2019) ve
Cosine Learning Rate Decay ile Aşırı Non-IID (Dirichlet alpha=0.3) Yakınsama İyileştirmesi.

Matematiksel Formülasyon:
Klasik FedAvg'da sunucu ağırlık güncellemesi doğrudan ağırlıklı ortalamadır:
  Delta_w^(t) = w^(t) - sum_{k=1}^K (n_k / n) w_k^(t+1)
  w^(t+1) = w^(t) - Delta_w^(t)

Non-IID veri dağılımında istemci gradyanları birbirine zıt yönlere çektiği için
FedAvgM sunucu tarafında bir hız/momentum vektörü v^(t) tutar:
  v^(t+1) = beta * v^(t) + Delta_w^(t)
  w^(t+1) = w^(t) - eta_server * v^(t+1)

Ana koda dokunmadan `02_FedAvgM_Server_Momentum/` altında bağımsız çalışır.
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
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.dataset import (
    DEFAULT_RADAR_DATA_DIR,
    NUM_CLASSES,
    NUM_CLIENTS,
    create_non_iid_partition,
    load_all_days_dataset,
    to_dataloader,
)
from src.models import RadarPositionCNN
from src.fl_engine import aggregate_weights, evaluate_model, train_local_client


def run_fedavgm_non_iid(
    client_loaders,
    test_loader,
    device: torch.device,
    num_rounds: int = 15,
    local_epochs: int = 2,
    lr_max: float = 1.2e-3,
    lr_min: float = 2e-4,
    server_momentum_beta: float = 0.7,
    server_lr: float = 1.0,
    prox_mu: float = 0.02,
    seed: int = 42,
    label: str = "FedAvgM",
):
    torch.manual_seed(seed)
    global_model = RadarPositionCNN(num_classes=NUM_CLASSES).to(device)

    # Sunucu momentum tamponu v^(t)
    velocity = {
        k: torch.zeros_like(v, dtype=torch.float32)
        for k, v in global_model.state_dict().items()
        if v.is_floating_point()
    }

    init_loss, init_acc, _, _ = evaluate_model(global_model, test_loader, device)
    history = {
        "round": [0],
        "test_acc": [init_acc],
        "test_loss": [init_loss],
        "train_acc": [np.nan],
    }

    for rnd in range(1, num_rounds + 1):
        lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * (rnd - 1) / num_rounds))

        old_state = copy.deepcopy(global_model.state_dict())
        client_states, client_sizes, round_accs = [], [], []

        for cid in range(NUM_CLIENTS):
            c_st, _, c_acc, c_sz = train_local_client(
                global_model=global_model,
                dataloader=client_loaders[cid],
                device=device,
                local_epochs=local_epochs,
                lr=lr_t,
                prox_mu=prox_mu,
            )
            client_states.append(c_st)
            client_sizes.append(c_sz)
            round_accs.append(c_acc * c_sz)

        avg_state = aggregate_weights(client_states, client_sizes)

        # Sunucu Momentum Güncellemesi:
        # pseudo-gradient Delta_w = w_old - w_avg
        # v = beta * v + (1 - beta_damp) * Delta_w  (Nesterov-benzeri sönümlemeli sunucu momentumu)
        new_state = {}
        for k in old_state.keys():
            if old_state[k].is_floating_point():
                delta_w = old_state[k].float() - avg_state[k].float()
                velocity[k] = server_momentum_beta * velocity[k] + (1.0 - 0.3 * server_momentum_beta) * delta_w
                updated = old_state[k].float() - server_lr * velocity[k]
                new_state[k] = updated.to(old_state[k].dtype)
            else:
                new_state[k] = avg_state[k].clone()

        global_model.load_state_dict(new_state)
        tr_acc = sum(round_accs) / sum(client_sizes)
        te_loss, te_acc, _, _ = evaluate_model(global_model, test_loader, device)

        history["round"].append(rnd)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        history["test_loss"].append(te_loss)

        print(
            f"[{label}] Tur {rnd:02d}/{num_rounds:02d} | Train Acc: {tr_acc*100:.2f}% | "
            f"Non-IID Global Test Acc: {te_acc*100:.2f}% (Loss: {te_loss:.4f})"
        )

    return global_model, history


def main():
    t0 = time.time()
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(exp_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== [Geliştirme 2] FedAvgM (Sunucu Momentumu) Non-IID Analizi (Cihaz: {device}) ===")

    clients_iid, _, (global_test_x, global_test_y) = load_all_days_dataset(
        base_dir=DEFAULT_RADAR_DATA_DIR, days=["day0", "day1", "day2"], norm_method="zscore"
    )
    clients_non_iid = create_non_iid_partition(clients_iid, alpha=0.3, seed=42)

    non_iid_loaders = {
        cid: to_dataloader(x, y, batch_size=64, shuffle=True) for cid, (x, y) in clients_non_iid.items()
    }
    test_loader = to_dataloader(global_test_x, global_test_y, batch_size=128, shuffle=False)

    print("\n--- 1. Standart FedAvg (Non-IID, alpha=0.3, beta=0.0) ---")
    _, hist_fedavg = run_fedavgm_non_iid(
        non_iid_loaders,
        test_loader,
        device,
        num_rounds=15,
        local_epochs=2,
        lr_max=1e-3,
        lr_min=1e-3,
        server_momentum_beta=0.0,
        prox_mu=0.0,
        label="FedAvg-NonIID",
    )

    print("\n--- 2. FedAvgM + Proksimal + CosineLR (Non-IID, alpha=0.3, beta=0.6) ---")
    model_fedavgm, hist_fedavgm = run_fedavgm_non_iid(
        non_iid_loaders,
        test_loader,
        device,
        num_rounds=15,
        local_epochs=2,
        lr_max=1.3e-3,
        lr_min=2e-4,
        server_momentum_beta=0.6,
        prox_mu=0.02,
        label="FedAvgM-NonIID",
    )

    torch.save(model_fedavgm.state_dict(), os.path.join(res_dir, "fedavgm_non_iid_best.pt"))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    axes[0].plot(hist_fedavg["round"], [a * 100 for a in hist_fedavg["test_acc"]], "r--^", linewidth=2, label="Standart FedAvg (Non-IID, α=0.3)")
    axes[0].plot(hist_fedavgm["round"], [a * 100 for a in hist_fedavgm["test_acc"]], "b-o", linewidth=2.2, label="FedAvgM (Sunucu Momentumu β=0.6 + CosineLR)")
    axes[0].set_title("Non-IID (α=0.3) Veri Dağılımında FedAvg vs. FedAvgM Test Doğruluğu", fontweight="bold")
    axes[0].set_xlabel("İletişim Turu (Round)")
    axes[0].set_ylabel("Global Test Doğruluğu (%)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right")

    axes[1].plot(hist_fedavg["round"], hist_fedavg["test_loss"], "r--^", linewidth=2, label="Standart FedAvg (Non-IID, α=0.3)")
    axes[1].plot(hist_fedavgm["round"], hist_fedavgm["test_loss"], "b-o", linewidth=2.2, label="FedAvgM (Sunucu Momentumu β=0.6 + CosineLR)")
    axes[1].set_title("Non-IID (α=0.3) Veri Dağılımında Test Kaybı (Cross-Entropy Loss)", fontweight="bold")
    axes[1].set_xlabel("İletişim Turu (Round)")
    axes[1].set_ylabel("Global Test Kaybı (Loss)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "fedavgm_comparison.png"), dpi=150)
    plt.close(fig)

    summary = {
        "method": "FedAvgM (Server Momentum beta=0.6 + Proximal mu=0.02 + Cosine Annealing LR)",
        "fedavg_non_iid": {
            "round_1_acc": round(float(hist_fedavg["test_acc"][1] * 100), 2),
            "peak_acc": round(float(max(hist_fedavg["test_acc"]) * 100), 2),
            "final_acc": round(float(hist_fedavg["test_acc"][-1] * 100), 2),
            "final_loss": round(float(hist_fedavg["test_loss"][-1]), 4),
            "acc_curve": [round(float(x * 100), 2) for x in hist_fedavg["test_acc"]],
        },
        "fedavgm_non_iid": {
            "round_1_acc": round(float(hist_fedavgm["test_acc"][1] * 100), 2),
            "peak_acc": round(float(max(hist_fedavgm["test_acc"]) * 100), 2),
            "final_acc": round(float(hist_fedavgm["test_acc"][-1] * 100), 2),
            "final_loss": round(float(hist_fedavgm["test_loss"][-1]), 4),
            "acc_curve": [round(float(x * 100), 2) for x in hist_fedavgm["test_acc"]],
        },
        "execution_time_seconds": round(time.time() - t0, 2),
    }

    with open(os.path.join(res_dir, "summary_02_fedavgm.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"\n=== [Geliştirme 2 Tamamlandı] FedAvg Non-IID Peak: %{summary['fedavg_non_iid']['peak_acc']} -> "
        f"FedAvgM Non-IID Peak: %{summary['fedavgm_non_iid']['peak_acc']} (Süre: {summary['execution_time_seconds']}s) ==="
    )


if __name__ == "__main__":
    main()
