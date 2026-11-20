# -*- coding: utf-8 -*-
"""
Geliştirme 4: Arızalı veya Siber Saldırı Altındaki (Byzantine / Model Poisoning) Radar Sensörlerine
Karşı Dayanıklı Federe Birleştirme (Byzantine-Robust Aggregation: Coordinate-wise Median & Trimmed Mean)
(Yin et al., "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates", ICML 2018).

Senaryo:
9 radar istemcisinden B = 2 tanesi (İstemci 0 ve İstemci 1) donanım arızası veya siber saldırı
(Byzantine Model Poisoning + Label Flipping) nedeniyle sunucuya yüksek genlikli bozuk/ters
ağırlıklar göndermektedir:
  w_byzantine = -3.0 * w_local + N(0, 1.5^2)

Karşılaştırılan Sunucu Birleştirme (Aggregation) Kuralları:
1. Standart FedAvg (Ortalama): Tek bir aşırı bozuk istemci bile tüm ortalamayı bozar.
2. Trimmed Mean (Kırpılmış Ortalama, k_trim=2): Her parametre koordinatı için en küçük 2 ve
   en büyük 2 istemci değerini atıp kalan 5 sağlam istemcinin ortalamasını alır.
3. Coordinate-wise Median (Koordinat Bazlı Medyan): Her ağırlık bileşeni için 9 istemcinin
   ortanca (median) değerini seçer.

Ana koda dokunmadan `04_Byzantine_Robust_Aggregation/` altında bağımsız çalışır.
"""

import copy
import json
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


def corrupt_byzantine_client_state(state_dict: dict, scale: float = -2.5, noise_std: float = 0.8) -> dict:
    """
    Saldırgan / Arızalı (Byzantine) bir radar düğümünün gönderdiği zehirlenmiş model ağırlıklarını simüle eder:
    Ağırlıkların işaretini ters çevirir ve yüksek varyanslı gürültü ekler.
    """
    corrupted = {}
    for k, v in state_dict.items():
        if v.is_floating_point():
            corrupted[k] = (scale * v.float() + torch.randn_like(v.float()) * noise_std).to(v.dtype)
        else:
            corrupted[k] = v.clone()
    return corrupted


def aggregate_coordinate_median(client_states: list) -> dict:
    """
    Koordinat Bazlı Medyan (Coordinate-wise Median):
    w_global[j] = Median( { w_1[j], w_2[j], ..., w_K[j] } )
    """
    agg_state = {}
    for key in client_states[0].keys():
        if client_states[0][key].is_floating_point():
            stacked = torch.stack([st[key].float() for st in client_states], dim=0)  # (K, ...)
            median_val = torch.median(stacked, dim=0).values
            agg_state[key] = median_val.to(client_states[0][key].dtype)
        else:
            agg_state[key] = client_states[0][key].clone()
    return agg_state


def aggregate_trimmed_mean(client_states: list, trim_k: int = 2) -> dict:
    """
    Koordinat Bazlı Kırpılmış Ortalama (Coordinate-wise beta-Trimmed Mean):
    Her parametre koordinatında değerleri küçükten büyüğe sıralar, en uçtaki `trim_k` adet en küçük
    ve `trim_k` adet en büyük değeri atar, kalan (K - 2*trim_k) istemcinin ortalamasını alır.
    """
    agg_state = {}
    num_clients = len(client_states)
    assert 2 * trim_k < num_clients, "trim_k çok büyük!"

    for key in client_states[0].keys():
        if client_states[0][key].is_floating_point():
            stacked = torch.stack([st[key].float() for st in client_states], dim=0)  # (K, ...)
            sorted_vals, _ = torch.sort(stacked, dim=0)
            trimmed = sorted_vals[trim_k : num_clients - trim_k]  # (K - 2*trim_k, ...)
            agg_state[key] = torch.mean(trimmed, dim=0).to(client_states[0][key].dtype)
        else:
            agg_state[key] = client_states[0][key].clone()
    return agg_state


def run_byzantine_experiment(
    client_loaders,
    test_loader,
    device: torch.device,
    agg_method: str = "fedavg",
    byzantine_clients: list = [0, 1],
    num_rounds: int = 12,
    local_epochs: int = 2,
    lr: float = 1e-3,
    seed: int = 42,
):
    torch.manual_seed(seed)
    global_model = RadarPositionCNN(num_classes=NUM_CLASSES).to(device)

    init_loss, init_acc, _, _ = evaluate_model(global_model, test_loader, device)
    history = {
        "round": [0],
        "test_acc": [init_acc],
        "test_loss": [init_loss],
    }

    for rnd in range(1, num_rounds + 1):
        c_states, c_sizes = [], []
        for cid in range(NUM_CLIENTS):
            st, _, _, sz = train_local_client(
                global_model=global_model,
                dataloader=client_loaders[cid],
                device=device,
                local_epochs=local_epochs,
                lr=lr,
                prox_mu=0.0,
            )
            if cid in byzantine_clients:
                st = corrupt_byzantine_client_state(st, scale=-2.5, noise_std=0.8)
            c_states.append(st)
            c_sizes.append(sz)

        if agg_method == "fedavg":
            new_global = aggregate_weights(c_states, c_sizes)
        elif agg_method == "trimmed_mean":
            new_global = aggregate_trimmed_mean(c_states, trim_k=2)
        elif agg_method == "median":
            new_global = aggregate_coordinate_median(c_states)
        else:
            raise ValueError(f"Bilinmeyen agregasyon yöntemi: {agg_method}")

        global_model.load_state_dict(new_global)
        te_loss, te_acc, _, _ = evaluate_model(global_model, test_loader, device)

        history["round"].append(rnd)
        history["test_acc"].append(te_acc)
        history["test_loss"].append(te_loss)

        if rnd % 3 == 0 or rnd == 1 or rnd == num_rounds:
            print(
                f"[{agg_method.upper()} | Saldırgan İstemci: {len(byzantine_clients)}/9] "
                f"Tur {rnd:02d}/{num_rounds:02d} | Test Acc: {te_acc*100:.2f}% | Loss: {te_loss:.4f}"
            )

    return global_model, history


def main():
    t0 = time.time()
    exp_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(exp_dir, "results")
    os.makedirs(res_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== [Geliştirme 4] Byzantine-Robust FL (Saldırgan/Arızalı Sensör Dayanıklılığı) (Cihaz: {device}) ===")

    clients_iid, _, (global_test_x, global_test_y) = load_all_days_dataset(
        base_dir=DEFAULT_RADAR_DATA_DIR, days=["day0", "day1", "day2"], norm_method="zscore"
    )
    client_loaders = {
        cid: to_dataloader(x, y, batch_size=64, shuffle=True) for cid, (x, y) in clients_iid.items()
    }
    test_loader = to_dataloader(global_test_x, global_test_y, batch_size=128, shuffle=False)

    NUM_ROUNDS = 12
    BYZANTINE_IDS = [0, 1]  # 9 radardan 2'si (%22.2) siber saldırı / arıza altında

    print("\n--- 1. Standart FedAvg (2/9 Saldırgan Radar Altında) ---")
    _, hist_fedavg_attack = run_byzantine_experiment(
        client_loaders, test_loader, device, agg_method="fedavg", byzantine_clients=BYZANTINE_IDS, num_rounds=NUM_ROUNDS
    )

    print("\n--- 2. Coordinate-wise Median FL (2/9 Saldırgan Radar Altında) ---")
    _, hist_median_attack = run_byzantine_experiment(
        client_loaders, test_loader, device, agg_method="median", byzantine_clients=BYZANTINE_IDS, num_rounds=NUM_ROUNDS
    )

    print("\n--- 3. Trimmed-Mean FL (k=2 Kırpılmış Ortalama, 2/9 Saldırgan Radar Altında) ---")
    model_trimmed, hist_trimmed_attack = run_byzantine_experiment(
        client_loaders, test_loader, device, agg_method="trimmed_mean", byzantine_clients=BYZANTINE_IDS, num_rounds=NUM_ROUNDS
    )

    torch.save(model_trimmed.state_dict(), os.path.join(res_dir, "robust_trimmed_mean_best.pt"))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    axes[0].plot(
        hist_fedavg_attack["round"],
        [a * 100 for a in hist_fedavg_attack["test_acc"]],
        "r--x",
        linewidth=2.2,
        label="Standart FedAvg (2/9 Saldırgan Sensör → Çöküş)",
    )
    axes[0].plot(
        hist_median_attack["round"],
        [a * 100 for a in hist_median_attack["test_acc"]],
        "g-^",
        linewidth=2.2,
        label="Coordinate-wise Median (2/9 Saldırgan Sensör)",
    )
    axes[0].plot(
        hist_trimmed_attack["round"],
        [a * 100 for a in hist_trimmed_attack["test_acc"]],
        "b-o",
        linewidth=2.2,
        label="Trimmed-Mean (k=2) (2/9 Saldırgan Sensör)",
    )
    axes[0].set_title("2/9 (%22.2) Byzantine Saldırgan Radar Altında Test Doğruluğu", fontweight="bold")
    axes[0].set_xlabel("İletişim Turu (Round)")
    axes[0].set_ylabel("Global Test Doğruluğu (%)")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="center right")

    methods = ["Standart FedAvg\n(Saldırı Altında)", "Coordinate-wise\nMedian", "Trimmed-Mean\n(k=2 Kırpılmış)"]
    final_accs = [
        round(float(hist_fedavg_attack["test_acc"][-1] * 100), 2),
        round(float(hist_median_attack["test_acc"][-1] * 100), 2),
        round(float(hist_trimmed_attack["test_acc"][-1] * 100), 2),
    ]
    bars = axes[1].bar(methods, final_accs, color=["#dc2626", "#16a34a", "#2563eb"], edgecolor="black", width=0.48)
    axes[1].set_ylim(0, 108)
    axes[1].set_title("Model Zehirleme (Poisoning) Saldırısına Karşı Nihai Başarı", fontweight="bold")
    axes[1].set_ylabel("Nihai Global Test Doğruluğu (%)")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.6)

    for bar, acc in zip(bars, final_accs):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1.5,
            f"%{acc:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "byzantine_robust_comparison.png"), dpi=150)
    plt.close(fig)

    summary = {
        "method": "Byzantine-Robust Aggregation under 2/9 (22.2%) Model Poisoning Attack",
        "byzantine_clients": BYZANTINE_IDS,
        "num_rounds": NUM_ROUNDS,
        "results": {
            "FedAvg_Under_Attack": {
                "peak_acc_percent": round(float(max(hist_fedavg_attack["test_acc"]) * 100), 2),
                "final_acc_percent": final_accs[0],
                "acc_curve": [round(float(x * 100), 2) for x in hist_fedavg_attack["test_acc"]],
            },
            "Coordinate_Median_Under_Attack": {
                "peak_acc_percent": round(float(max(hist_median_attack["test_acc"]) * 100), 2),
                "final_acc_percent": final_accs[1],
                "acc_curve": [round(float(x * 100), 2) for x in hist_median_attack["test_acc"]],
            },
            "Trimmed_Mean_Under_Attack": {
                "peak_acc_percent": round(float(max(hist_trimmed_attack["test_acc"]) * 100), 2),
                "final_acc_percent": final_accs[2],
                "acc_curve": [round(float(x * 100), 2) for x in hist_trimmed_attack["test_acc"]],
            },
        },
        "execution_time_seconds": round(time.time() - t0, 2),
    }

    with open(os.path.join(res_dir, "summary_04_byzantine.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(
        f"\n=== [Geliştirme 4 Tamamlandı] Saldırı Altında FedAvg: %{final_accs[0]} | "
        f"Median: %{final_accs[1]} | Trimmed-Mean: %{final_accs[2]} (Süre: {summary['execution_time_seconds']}s) ==="
    )


if __name__ == "__main__":
    main()
