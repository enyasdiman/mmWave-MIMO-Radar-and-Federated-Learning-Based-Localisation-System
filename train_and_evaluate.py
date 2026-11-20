# -*- coding: utf-8 -*-
"""
Federated Radar Position Recognition - Tam Kapsamlı Eğitim, Analiz ve Görselleştirme Betiği.

Çalıştırdığı Deneyler:
1. Veri Seti Analizi: day0, day1 ve day2 üzerindeki 9 radar istemcisinin (toplam 9.000 eğitim, 3.500 test
   örneği) yüklenmesi, AWR1842 radar fiziksel parametrelerinin doğrulanması ve örnek Range-Azimuth
   ısı haritalarının görselleştirilmesi.
2. IID Senaryosu (Orijinal 9 İstemci Dağılımı):
   - Merkezi Eğitim (Centralized Baseline)
   - FedAvg (9 İstemci, 3 Gün Birleşik Veri)
3. Non-IID Senaryosu (Dirichlet alpha=0.3 ile Heterojen İstemci Dağılımı):
   - FedAvg (Non-IID)
   - FedProx (Non-IID, mu=0.05) -> İstemci sapmasını (client drift) azaltma etkisi
4. Çıktılar:
   - `results/experiment_summary.json`: Tüm metrikler ve sınıflandırma raporları
   - `results/radar_range_azimuth_samples.png`: Sınıflara ait örnek Range-Azimuth haritaları
   - `results/iid_vs_non_iid_distribution.png`: İstemci-Sınıf dağılım ısı haritası
   - `results/training_curves_comparison.png`: Merkezi vs FedAvg vs FedProx doğruluk/kayıp eğrileri
   - `results/confusion_matrix_fedavg.png`: Nihai Global Model Karmaşıklık Matrisi
   - `results/best_global_radar_cnn.pt`: Eğitilmiş PyTorch model ağırlıkları
"""

import json
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import (
    DEFAULT_RADAR_DATA_DIR,
    NUM_CLASSES,
    NUM_CLIENTS,
    create_non_iid_partition,
    get_client_class_distribution,
    load_all_days_dataset,
    to_dataloader,
)
from src.models import RadarPositionCNN
from src.fl_engine import (
    evaluate_model,
    run_centralized_training,
    run_federated_training,
)
from src.awr1842_bridge import (
    DEFAULT_CFG_PATH,
    parse_awr1842_config,
    predict_radar_frame,
    raw_tlv_azimuth_to_heatmap,
)


def plot_radar_samples(x_data: np.ndarray, y_data: np.ndarray, save_path: str):
    """Her konum sınıfından (Sınıf 0..9) birer Range-Azimuth ısı haritası çizer."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.flatten()
    for cls_id in range(NUM_CLASSES):
        idx = np.where(y_data == cls_id)[0][0]
        sample_map = x_data[idx, 0]  # (256, 63)
        ax = axes[cls_id]
        im = ax.imshow(sample_map, aspect="auto", cmap="viridis", origin="lower")
        ax.set_title(f"Konum Sınıfı {cls_id}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Azimuth Bin (0..62)")
        ax.set_ylabel("Range Bin (0..255)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle("mmWave MIMO Radar (256x63) Range-Azimuth Konum İmzaları", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_client_distributions(iid_dist: np.ndarray, non_iid_dist: np.ndarray, save_path: str):
    """IID ve Non-IID (Dirichlet) istemci veri dağılımlarını yan yana görselleştirir."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    sns.heatmap(
        iid_dist,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        xticklabels=[f"S{i}" for i in range(NUM_CLASSES)],
        yticklabels=[f"İstemci {i+1}" for i in range(NUM_CLIENTS)],
    )
    axes[0].set_title("Orijinal Veri Dağılımı (IID - Dengeli, 3 Gün Birleşik)", fontweight="bold")
    axes[0].set_xlabel("Konum Sınıfı")
    axes[0].set_ylabel("Radar İstemcisi")

    sns.heatmap(
        non_iid_dist,
        annot=True,
        fmt="d",
        cmap="Oranges",
        ax=axes[1],
        xticklabels=[f"S{i}" for i in range(NUM_CLASSES)],
        yticklabels=[f"İstemci {i+1}" for i in range(NUM_CLIENTS)],
    )
    axes[1].set_title("Simüle Edilen Non-IID Dağılım (Dirichlet α = 0.3)", fontweight="bold")
    axes[1].set_xlabel("Konum Sınıfı")
    axes[1].set_ylabel("Radar İstemcisi")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_comparisons(
    hist_central: dict,
    hist_fedavg_iid: dict,
    hist_fedavg_noniid: dict,
    hist_fedprox_noniid: dict,
    save_path: str,
):
    """Tüm algoritmaların Global Test Doğruluğu ve Kayıp eğrilerini karşılaştırır."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. Test Doğruluğu
    ax = axes[0]
    ax.plot(hist_central["round"], [a * 100 for a in hist_central["test_acc"]], "k--o", label="Merkezi (Centralized - IID)", linewidth=2, markersize=4)
    ax.plot(hist_fedavg_iid["round"], [a * 100 for a in hist_fedavg_iid["test_acc"]], "b-s", label="FedAvg (IID - 9 İstemci)", linewidth=2, markersize=4)
    ax.plot(hist_fedavg_noniid["round"], [a * 100 for a in hist_fedavg_noniid["test_acc"]], "r-^", label="FedAvg (Non-IID, α=0.3)", linewidth=2, markersize=4)
    ax.plot(hist_fedprox_noniid["round"], [a * 100 for a in hist_fedprox_noniid["test_acc"]], "g-D", label="FedProx (Non-IID, α=0.3, μ=0.05)", linewidth=2, markersize=4)
    ax.set_title("Global Test Doğruluğu Karşılaştırması (%)", fontsize=12, fontweight="bold")
    ax.set_xlabel("İletişim Turu (Round) / Epoch")
    ax.set_ylabel("Test Doğruluğu (%)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")

    # 2. Test Kaybı
    ax2 = axes[1]
    ax2.plot(hist_central["round"], hist_central["test_loss"], "k--o", label="Merkezi (Centralized - IID)", linewidth=2, markersize=4)
    ax2.plot(hist_fedavg_iid["round"], hist_fedavg_iid["test_loss"], "b-s", label="FedAvg (IID - 9 İstemci)", linewidth=2, markersize=4)
    ax2.plot(hist_fedavg_noniid["round"], hist_fedavg_noniid["test_loss"], "r-^", label="FedAvg (Non-IID, α=0.3)", linewidth=2, markersize=4)
    ax2.plot(hist_fedprox_noniid["round"], hist_fedprox_noniid["test_loss"], "g-D", label="FedProx (Non-IID, α=0.3, μ=0.05)", linewidth=2, markersize=4)
    ax2.set_title("Global Test Kaybı (Cross-Entropy Loss)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("İletişim Turu (Round) / Epoch")
    ax2.set_ylabel("Test Kaybı (Loss)")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str):
    plt.figure(figsize=(9, 7.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Sınıf {i}" for i in range(NUM_CLASSES)],
        yticklabels=[f"Sınıf {i}" for i in range(NUM_CLASSES)],
    )
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Tahmin Edilen Konum Etiketi")
    plt.ylabel("Gerçek Konum Etiketi")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    start_time = time.time()
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Çalışma Ortamı: PyTorch {torch.__version__} | Cihaz: {device} ===")

    # 1. AWR1842 Radar Donanım Parametrelerini Doğrula
    radar_params = parse_awr1842_config(DEFAULT_CFG_PATH)
    print("\n--- 1. TI AWR1842 mmWave MIMO Radar Donanım Konfigürasyonu ---")
    for k, v in radar_params.items():
        print(f"  {k}: {v}")

    # 2. Tüm Günlerin (day0, day1, day2) Veri Setini Yükle
    print("\n--- 2. Veri Seti Yükleniyor (day0 + day1 + day2) ---")
    clients_iid, day_tests, (global_test_x, global_test_y) = load_all_days_dataset(
        base_dir=DEFAULT_RADAR_DATA_DIR,
        days=["day0", "day1", "day2"],
        norm_method="zscore",
    )

    total_train_samples = sum(len(v[1]) for v in clients_iid.values())
    print(f"  Toplam Eğitim Verisi (9 İstemci x 3 Gün): {total_train_samples} örnek")
    print(f"  Toplam Global Test Verisi (3 Gün Birleşik): {len(global_test_y)} örnek")
    for d_name, (dtx, dty) in day_tests.items():
        print(f"    - {d_name} Test Kümesi: {dtx.shape}")

    # Örnek Radar Range-Azimuth haritalarını kaydet
    plot_radar_samples(
        global_test_x,
        global_test_y,
        os.path.join(results_dir, "radar_range_azimuth_samples.png"),
    )

    # 3. Non-IID Veri Bölümlemesi Oluştur (Dirichlet alpha=0.3)
    clients_non_iid = create_non_iid_partition(clients_iid, alpha=0.3, seed=42)
    iid_dist_mat = get_client_class_distribution(clients_iid)
    non_iid_dist_mat = get_client_class_distribution(clients_non_iid)
    plot_client_distributions(
        iid_dist_mat,
        non_iid_dist_mat,
        os.path.join(results_dir, "iid_vs_non_iid_distribution.png"),
    )

    # DataLoader nesnelerini hazırla
    BATCH_SIZE = 64
    NUM_ROUNDS = 15
    LOCAL_EPOCHS = 2
    LR = 1e-3

    iid_client_loaders = {
        cid: to_dataloader(x, y, batch_size=BATCH_SIZE, shuffle=True)
        for cid, (x, y) in clients_iid.items()
    }
    non_iid_client_loaders = {
        cid: to_dataloader(x, y, batch_size=BATCH_SIZE, shuffle=True)
        for cid, (x, y) in clients_non_iid.items()
    }
    global_test_loader = to_dataloader(global_test_x, global_test_y, batch_size=128, shuffle=False)
    day_test_loaders = {
        d: to_dataloader(tx, ty, batch_size=128, shuffle=False)
        for d, (tx, ty) in day_tests.items()
    }

    # Merkezi eğitim için tüm eğitim verisini birleştir
    central_x = np.concatenate([clients_iid[i][0] for i in range(NUM_CLIENTS)], axis=0)
    central_y = np.concatenate([clients_iid[i][1] for i in range(NUM_CLIENTS)], axis=0)
    central_train_loader = to_dataloader(central_x, central_y, batch_size=BATCH_SIZE, shuffle=True)

    # 4. Deney 1: Merkezi (Centralized) Eğitim
    print("\n--- 3. Deney 1: Geleneksel Merkezi (Centralized) Eğitim ---")
    central_model, hist_central = run_centralized_training(
        model_fn=lambda: RadarPositionCNN(num_classes=NUM_CLASSES),
        train_loader=central_train_loader,
        test_loader=global_test_loader,
        device=device,
        num_epochs=NUM_ROUNDS,
        lr=LR,
    )

    # 5. Deney 2: FedAvg (IID - 9 İstemci)
    print("\n--- 4. Deney 2: Federated Averaging (FedAvg - IID, 9 İstemci) ---")
    fedavg_iid_model, hist_fedavg_iid = run_federated_training(
        model_fn=lambda: RadarPositionCNN(num_classes=NUM_CLASSES),
        client_loaders=iid_client_loaders,
        test_loader=global_test_loader,
        extra_test_loaders=day_test_loaders,
        device=device,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        lr=LR,
        prox_mu=0.0,
        algo_name="FedAvg-IID",
    )

    # 6. Deney 3: FedAvg (Non-IID, alpha=0.3)
    print("\n--- 5. Deney 3: Federated Averaging (FedAvg - Non-IID, α=0.3) ---")
    fedavg_noniid_model, hist_fedavg_noniid = run_federated_training(
        model_fn=lambda: RadarPositionCNN(num_classes=NUM_CLASSES),
        client_loaders=non_iid_client_loaders,
        test_loader=global_test_loader,
        device=device,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        lr=LR,
        prox_mu=0.0,
        algo_name="FedAvg-NonIID",
    )

    # 7. Deney 4: FedProx (Non-IID, alpha=0.3, mu=0.05)
    print("\n--- 6. Deney 4: FedProx (Non-IID, α=0.3, μ=0.05) ---")
    fedprox_noniid_model, hist_fedprox_noniid = run_federated_training(
        model_fn=lambda: RadarPositionCNN(num_classes=NUM_CLASSES),
        client_loaders=non_iid_client_loaders,
        test_loader=global_test_loader,
        device=device,
        num_rounds=NUM_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
        lr=LR,
        prox_mu=0.05,
        algo_name="FedProx-NonIID",
    )

    # Eğitim Eğrilerini Kaydet
    plot_training_comparisons(
        hist_central,
        hist_fedavg_iid,
        hist_fedavg_noniid,
        hist_fedprox_noniid,
        os.path.join(results_dir, "training_curves_comparison.png"),
    )

    # 8. Detaylı Nihai Değerlendirme ve Karmaşıklık Matrisi
    print("\n--- 7. Nihai Performans Değerlendirmesi (FedAvg - IID Global Model) ---")
    final_loss, final_acc, y_true, y_pred = evaluate_model(
        fedavg_iid_model, global_test_loader, device
    )
    target_names = [f"Sınıf {i}" for i in range(NUM_CLASSES)]
    cls_report_str = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    cls_report_dict = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    print(cls_report_str)

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(
        cm,
        f"FedAvg Global Model Karmaşıklık Matrisi (Doğruluk: %{final_acc*100:.2f})",
        os.path.join(results_dir, "confusion_matrix_fedavg.png"),
    )

    # Gün Bazlı Test Doğrulukları
    day_accuracies = {}
    for d_name, d_loader in day_test_loaders.items():
        _, d_acc, _, _ = evaluate_model(fedavg_iid_model, d_loader, device)
        day_accuracies[d_name] = round(float(d_acc * 100), 2)
        print(f"  {d_name} Test Doğruluğu: %{d_acc*100:.2f}")

    # En iyi global modeli kaydet
    model_save_path = os.path.join(results_dir, "best_global_radar_cnn.pt")
    torch.save(fedavg_iid_model.state_dict(), model_save_path)

    # 9. AWR1842 Gerçek Zamanlı Köprü Testi (TLV -> (256, 63) -> Model Tahmini)
    print("\n--- 8. AWR1842 TLV Akışı & Gerçek Zamanlı Çıkarım (Inference) Doğrulaması ---")
    synthetic_tlv = np.random.randint(-500, 500, size=(2 * 8 * 256,), dtype=np.int16)
    reconstructed_heatmap = raw_tlv_azimuth_to_heatmap(synthetic_tlv)
    # Gerçek bir test çerçevesi üzerinde çıkarım süresi ve doğruluğu test et
    sample_real_frame = global_test_x[0, 0]  # (256, 63)
    true_label = int(global_test_y[0])
    t_inf0 = time.perf_counter()
    pred_cls, conf, _ = predict_radar_frame(fedavg_iid_model, sample_real_frame, device)
    inf_ms = (time.perf_counter() - t_inf0) * 1000.0
    print(
        f"  TLV -> Heatmap Boyutu: {reconstructed_heatmap.shape} | "
        f"Gerçek Çerçeve Tahmini: Sınıf {pred_cls} (Gerçek: Sınıf {true_label}, Güven: %{conf*100:.2f}) | "
        f"Çıkarım Süresi: {inf_ms:.2f} ms"
    )

    # Özet JSON Kaydı
    summary = {
        "hardware_config": radar_params,
        "dataset_stats": {
            "total_train_samples": total_train_samples,
            "total_test_samples": int(len(global_test_y)),
            "num_clients": NUM_CLIENTS,
            "num_classes": NUM_CLASSES,
            "days": ["day0", "day1", "day2"],
        },
        "final_test_accuracy_percent": {
            "Centralized_IID": round(float(hist_central["test_acc"][-1] * 100), 2),
            "FedAvg_IID": round(float(hist_fedavg_iid["test_acc"][-1] * 100), 2),
            "FedAvg_NonIID": round(float(hist_fedavg_noniid["test_acc"][-1] * 100), 2),
            "FedProx_NonIID": round(float(hist_fedprox_noniid["test_acc"][-1] * 100), 2),
        },
        "day_wise_test_accuracy_percent": day_accuracies,
        "classification_report": cls_report_dict,
        "realtime_inference_ms": round(inf_ms, 3),
        "total_execution_seconds": round(time.time() - start_time, 2),
    }

    with open(os.path.join(results_dir, "experiment_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Tüm deneyler {summary['total_execution_seconds']} saniyede tamamlandı! Sonuçlar '{results_dir}' klasörüne kaydedildi. ===")


if __name__ == "__main__":
    main()
