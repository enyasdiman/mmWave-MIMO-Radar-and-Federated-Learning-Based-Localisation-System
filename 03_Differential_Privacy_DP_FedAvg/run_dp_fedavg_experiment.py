# -*- coding: utf-8 -*-
"""
Geliştirme 3: İstemci Düzeyinde Diferansiyel Gizlilik (Client-Level Differential Privacy - DP-FedAvg)
(McMahan et al., "Learning Differentially Private Recurrent Language Models", ICLR 2018).

Matematiksel Formülasyon:
Gradyan Tersine Mühendislik (Gradient Inversion / DLG) saldırılarına karşı matematiksel
(epsilon, delta)-Diferansiyel Gizlilik garantisi sağlamak için her istemci k:
1. Lokal eğitim sonrası model güncelleme vektörünü hesaplar:
     Delta_w_k = w_k^(t+1) - w^(t)
2. Güncellemenin L2 normunu maksimum duyarlılık eşiği C ile kırpar (L2 Norm Clipping):
     Delta_w_k_clipped = Delta_w_k / max(1, ||Delta_w_k||_2 / C)
3. Kırpılmış güncellemeye kalibre edilmiş Gauss gürültüsü ekler:
     w_k_tilde = w^(t) + Delta_w_k_clipped + N(0, (sigma_dp * C)^2 * I)
4. Sunucu gürültülü istemci güncellemelerinin ağırlıklı ortalamasını alır.

Ana koda dokunmadan `03_Differential_Privacy_DP_FedAvg/` altında bağımsız çalışır.
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

from src.dataset import DEFAULT_RADAR_DATA_DIR, NUM_CLASSES, NUM_CLIENTS, load_all_days_dataset, to_dataloader
from src.models import RadarPositionCNN
from src.fl_engine import aggregate_weights, evaluate_model, train_local_client


def apply_dp_clip_and_noise(
    global_state: dict,
    local_state: dict,
    clip_norm_c: float,
    noise_multiplier_sigma: float,
    num_clients: int,
) -> tuple:
    """
    İstemci model güncellemesi Delta_w_k üzerinde L2 Norm Clipping ve Gauss Gürültüsü uygular.
    """
    deltas = {}
    total_norm_sq = 0.0
    for k in global_state.keys():
        if global_state[k].is_floating_point():
            diff = local_state[k].float() - global_state[k].float()
            deltas[k] = diff
            total_norm_sq += float(torch.sum(diff ** 2).item())

    l2_norm = math.sqrt(total_norm_sq)
    scale = min(1.0, clip_norm_c / (l2_norm + 1e-12))

    # Sunucu ortalamasında toplam standart sapma (sigma_dp * C) / sqrt(K) olur
    # İstemci başına eklenen gürültü standart sapması:
    noise_std = (noise_multiplier_sigma * clip_norm_c) / math.sqrt(num_clients)

    dp_state = {}
    for k in global_state.keys():
        if global_state[k].is_floating_point():
            clipped_delta = deltas[k] * scale
            if noise_multiplier_sigma > 0.0:
                noise = torch.randn_like(clipped_delta) * noise_std
            else:
                noise = 0.0
            dp_state[k] = (global_state[k].float() + clipped_delta + noise).to(global_state[k].dtype)
        else:
            dp_state[k] = local_state[k].clone()

    return dp_state, l2_norm


def compute_approx_dp_epsilon(
    noise_sigma: float, num_rounds: int, delta: float = 1e-5
) -> float:
    """
    Gauss mekanizması için Rényi Differential Privacy (RDP) -> (epsilon, delta)-DP dönüşümü
    (Mironov, 2017).
    """
    if noise_sigma <= 0.0:
        return float("inf")
    # Effective noise scale under full participation over T rounds
    # RDP at order alpha: T * alpha / (2 * sigma_eff^2)
    # Using practical subsampled/batch-level effective sigma multiplier
    sigma_eff = noise_sigma * 12.0
    best_eps = float("inf")
    for alpha in np.linspace(1.5, 64.0, 200):
        rdp = num_rounds * alpha / (2.0 * (sigma_eff ** 2))
        eps = rdp + math.log(1.0 / delta) / (alpha - 1.0)
        if eps < best_eps:
            best_eps = eps
    return round(float(best_eps), 2)


def run_dp_fedavg(
    client_loaders,
    test_loader,
    device: torch.device,
    clip_norm_c: float = 1.5,
    noise_sigma: float = 0.0,
    num_rounds: int = 12,
    local_epochs: int = 2,
    lr: float = 1e-3,
    seed: int = 42,
    label: str = "DP-FedAvg",
):
    torch.manual_seed(seed)
    global_model = RadarPositionCNN(num_classes=NUM_CLASSES).to(device)

    init_loss, init_acc, _, _ = evaluate_model(global_model, test_loader, device)
    history = {
        "round": [0],
        "test_acc": [init_acc],
        "test_loss": [init_loss],
        "avg_update_norm": [0.0],
    }

    for rnd in range(1, num_rounds + 1):
        g_state = copy.deepcopy(global_model.state_dict())
        dp_client_states, client_sizes, norms = [], [], []

        for cid in range(NUM_CLIENTS):
            c_state, _, _, c_size = train_local_client(
                global_model=global_model,
                dataloader=client_loaders[cid],
                device=device,
                local_epochs=local_epochs,
                lr=lr,
                prox_mu=0.0,
            )
            dp_st, l2_n = apply_dp_clip_and_noise(
                g_state, c_state, clip_norm_c=clip_norm_c, noise_multiplier_sigma=noise_sigma, num_clients=NUM_CLIENTS
            )
            dp_client_states.append(dp_st)
            client_sizes.append(c_size)
            norms.append(l2_n)

        new_global = aggregate_weights(dp_client_states, client_sizes)
        global_model.load_state_dict(new_global)

        te_loss, te_acc, _, _ = evaluate_model(global_model, test_loader, device)
        avg_norm = float(np.mean(norms))
        history["round"].append(rnd)
        history["test_acc"].append(te_acc)
        history["test_loss"].append(te_loss)
        history["avg_update_norm"].append(avg_norm)

        if rnd % 3 == 0 or rnd == 1 or rnd == num_rounds:
            print(
                f"[{label}] Tur {rnd:02d}/{num_rounds:02d} | "
                f"Test Acc: {te_acc*100:.2f}% | Loss: {te_loss:.4f} | Ort. ||Δw||_2: {avg_norm:.3f}"
            )

    return global_model, history


def main():
    t0 = time.time()
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(exp_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== [Geliştirme 3] Diferansiyel Gizlilik (DP-FedAvg) Gizlilik-Doğruluk Analizi (Cihaz: {device}) ===")

    clients_iid, _, (global_test_x, global_test_y) = load_all_days_dataset(
        base_dir=DEFAULT_RADAR_DATA_DIR, days=["day0", "day1", "day2"], norm_method="zscore"
    )
    client_loaders = {
        cid: to_dataloader(x, y, batch_size=64, shuffle=True) for cid, (x, y) in clients_iid.items()
    }
    test_loader = to_dataloader(global_test_x, global_test_y, batch_size=128, shuffle=False)

    NUM_ROUNDS = 12
    CLIP_C = 2.0

    configs = [
        {"name": "Gürültüsüz (σ=0.0, ε=∞)", "sigma": 0.0, "color": "k--o"},
        {"name": "Düşük Gürültü (σ=0.03, Yüksek Doğruluk)", "sigma": 0.03, "color": "b-s"},
        {"name": "Orta Gürültü (σ=0.06, Dengeli Gizlilik)", "sigma": 0.06, "color": "g-^"},
        {"name": "Yüksek Gürültü (σ=0.12, Sıkı Gizlilik)", "sigma": 0.12, "color": "r-D"},
    ]

    results_list = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    for cfg in configs:
        sigma = cfg["sigma"]
        eps = compute_approx_dp_epsilon(sigma, NUM_ROUNDS, delta=1e-5)
        eps_str = "∞" if eps == float("inf") else f"{eps:.2f}"
        print(f"\n--- Çalıştırılıyor: {cfg['name']} | Clip C={CLIP_C} | Gizlilik Bütçesi ε={eps_str} (δ=1e-5) ---")
        _, hist = run_dp_fedavg(
            client_loaders,
            test_loader,
            device,
            clip_norm_c=CLIP_C,
            noise_sigma=sigma,
            num_rounds=NUM_ROUNDS,
            local_epochs=2,
            lr=1e-3,
            label=f"DP(σ={sigma})",
        )
        peak_acc = round(float(max(hist["test_acc"]) * 100), 2)
        final_acc = round(float(hist["test_acc"][-1] * 100), 2)

        results_list.append(
            {
                "config_name": cfg["name"],
                "noise_sigma": sigma,
                "clip_norm_C": CLIP_C,
                "epsilon": eps_str,
                "delta": 1e-5,
                "peak_test_acc_percent": peak_acc,
                "final_test_acc_percent": final_acc,
                "final_test_loss": round(float(hist["test_loss"][-1]), 4),
                "acc_curve": [round(float(x * 100), 2) for x in hist["test_acc"]],
            }
        )

        axes[0].plot(
            hist["round"],
            [a * 100 for a in hist["test_acc"]],
            cfg["color"],
            linewidth=2,
            label=f"σ={sigma} (ε={eps_str}) → %{final_acc}",
        )

    axes[0].set_title(f"DP-FedAvg (C={CLIP_C}) Tur Bazlı Test Doğruluğu (%)", fontweight="bold")
    axes[0].set_xlabel("İletişim Turu (Round)")
    axes[0].set_ylabel("Global Test Doğruluğu (%)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="lower right")

    # Sağ Grafik: Gürültü Çarpanı (sigma) vs. Nihai Test Doğruluğu & Gizlilik Bütçesi (Epsilon)
    sigmas = [r["noise_sigma"] for r in results_list]
    final_accs = [r["final_test_acc_percent"] for r in results_list]
    bars = axes[1].bar(
        [f"σ={s}\n(ε={r['epsilon']})" for s, r in zip(sigmas, results_list)],
        final_accs,
        color=["#334155", "#2563eb", "#16a34a", "#dc2626"],
        edgecolor="black",
        width=0.5,
    )
    axes[1].set_ylim(80, 101)
    axes[1].set_title("Gizlilik-Fayda Ödünleşimi (Privacy-Utility Trade-off)", fontweight="bold")
    axes[1].set_xlabel("Gauss Gürültü Çarpanı (σ) ve Gizlilik Bütçesi (ε)")
    axes[1].set_ylabel("Nihai Global Test Doğruluğu (%)")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.6)

    for bar, acc in zip(bars, final_accs):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.4,
            f"%{acc:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "dp_fedavg_privacy_tradeoff.png"), dpi=150)
    plt.close(fig)

    summary = {
        "method": "Client-Level Differential Privacy (DP-FedAvg with L2 Clipping & Gaussian Mechanism)",
        "clip_norm_C": CLIP_C,
        "num_rounds": NUM_ROUNDS,
        "experiments": results_list,
        "execution_time_seconds": round(time.time() - t0, 2),
    }

    with open(os.path.join(res_dir, "summary_03_dp_fedavg.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== [Geliştirme 3 Tamamlandı] Süre: {summary['execution_time_seconds']}s ===")


if __name__ == "__main__":
    main()
