# -*- coding: utf-8 -*-
"""
mmWave MIMO Radar Veri Yükleme, Ön İşleme ve Dağıtık (IID / Non-IID) Bölümleme Modülü.

Önemli Düzeltmeler:
1. Orijinal koddaki `labels.flatten() - 1` işlemi uint8 veri tipinde 0 -> 255 (underflow) hatasına
   yol açıyordu. Etiketler zaten 0..9 aralığında olduğu için doğrudan int64'e dönüştürülür.
2. Sadece day0 değil, day0, day1 ve day2 eğitim/test kümelerinin tamamı yüklenebilir.
3. Örnek bazlı (per-sample) Z-score ve Min-Max normalizasyon seçenekleri eklendi.
4. Dirichlet dağılımı tabanlı Non-IID istemci veri bölümleme simülasyonu eklendi.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import TensorDataset, DataLoader

DEFAULT_RADAR_DATA_DIR = r"C:\Users\enyas\Desktop\Colab Notebooks\Radar"
DAYS = ["day0", "day1", "day2"]
NUM_CLIENTS = 9
NUM_CLASSES = 10


def normalize_radar_samples(data: np.ndarray, method: str = "zscore") -> np.ndarray:
    """
    Radar Range-Azimuth haritalarını (N, 256, 63) örnek bazlı normalize eder.
    method: 'zscore' (önerilen) veya 'minmax'
    """
    data = data.astype(np.float32)
    if method == "zscore":
        mean = data.mean(axis=(1, 2), keepdims=True)
        std = data.std(axis=(1, 2), keepdims=True) + 1e-8
        return (data - mean) / std
    elif method == "minmax":
        d_min = data.min(axis=(1, 2), keepdims=True)
        d_max = data.max(axis=(1, 2), keepdims=True)
        return (data - d_min) / (d_max - d_min + 1e-8)
    else:
        raise ValueError(f"Bilinmeyen normalizasyon yöntemi: {method}")


def load_day_data(
    base_dir: str = DEFAULT_RADAR_DATA_DIR,
    day: str = "day0",
    norm_method: str = "zscore",
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]]:
    """
    Belirtilen güne (day0, day1, day2) ait 9 istemcinin eğitim verisini ve test verisini yükler.
    Dönüş:
      client_data: {client_id (0..8): (X_train, y_train)} -> X shape: (N, 1, 256, 63), y shape: (N,)
      test_data: (X_test, y_test)
    """
    train_folder = os.path.join(base_dir, f"training_data_range_azimuth_{day}")
    test_folder = os.path.join(base_dir, f"test_data_range_azimuth_{day}")

    # Google Drive veya düz klasör yapısı için fallback
    if not os.path.exists(train_folder):
        train_folder = base_dir
    if not os.path.exists(test_folder):
        test_folder = base_dir

    client_data: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    for client_idx in range(NUM_CLIENTS):
        file_idx = client_idx + 1
        train_path = os.path.join(train_folder, f"mmwave_data_train_{file_idx}.mat")
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Eğitim dosyası bulunamadı: {train_path}")

        mat = sio.loadmat(train_path)
        data_key = f"mmwave_data_train_{file_idx}"
        label_key = f"label_train_{file_idx}"

        raw_x = mat[data_key]
        # DÜZELTME: Etiketler zaten 0..9 aralığında uint8 olarak kayıtlı. (-1 çıkarılmaz!)
        raw_y = mat[label_key].flatten().astype(np.int64)
        if raw_y.min() == 1 and raw_y.max() == 10:
            raw_y = raw_y - 1

        norm_x = normalize_radar_samples(raw_x, method=norm_method)
        norm_x = np.expand_dims(norm_x, axis=1)  # (N, 1, 256, 63) PyTorch NCHW formatı
        client_data[client_idx] = (norm_x, raw_y)

    test_path = os.path.join(test_folder, "mmwave_data_test.mat")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dosyası bulunamadı: {test_path}")

    test_mat = sio.loadmat(test_path)
    raw_test_x = test_mat["mmwave_data_test"]
    raw_test_y = test_mat["label_test"].flatten().astype(np.int64)
    if raw_test_y.min() == 1 and raw_test_y.max() == 10:
        raw_test_y = raw_test_y - 1

    norm_test_x = normalize_radar_samples(raw_test_x, method=norm_method)
    norm_test_x = np.expand_dims(norm_test_x, axis=1)

    return client_data, (norm_test_x, raw_test_y)


def load_all_days_dataset(
    base_dir: str = DEFAULT_RADAR_DATA_DIR,
    days: List[str] = DAYS,
    norm_method: str = "zscore",
) -> Tuple[
    Dict[int, Tuple[np.ndarray, np.ndarray]],
    Dict[str, Tuple[np.ndarray, np.ndarray]],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Tüm günlerin (day0, day1, day2) verilerini yükler ve her istemci (0..8) için birleştirir.
    Ayrıca gün bazlı test kümelerini ve birleştirilmiş global test kümesini döndürür.
    """
    merged_clients: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {
        i: [] for i in range(NUM_CLIENTS)
    }
    day_tests: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    all_test_x, all_test_y = [], []

    for day in days:
        c_data, (tx, ty) = load_day_data(base_dir=base_dir, day=day, norm_method=norm_method)
        for cid, (cx, cy) in c_data.items():
            merged_clients[cid].append((cx, cy))
        day_tests[day] = (tx, ty)
        all_test_x.append(tx)
        all_test_y.append(ty)

    final_clients: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for cid in range(NUM_CLIENTS):
        xs = np.concatenate([item[0] for item in merged_clients[cid]], axis=0)
        ys = np.concatenate([item[1] for item in merged_clients[cid]], axis=0)
        final_clients[cid] = (xs, ys)

    global_test = (
        np.concatenate(all_test_x, axis=0),
        np.concatenate(all_test_y, axis=0),
    )

    return final_clients, day_tests, global_test


def create_non_iid_partition(
    client_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    alpha: float = 0.5,
    seed: int = 42,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Mevcut istemci verilerini havuzda birleştirip Dirichlet(alpha) dağılımı ile
    9 istemciye Non-IID (dengesiz sınıf dağılımlı) olarak yeniden dağıtır.
    Küçük alpha (örn. 0.3 - 0.5) yüksek derecede Non-IID heterojenlik üretir.
    """
    rng = np.random.default_rng(seed)
    num_clients = len(client_data)

    all_x = np.concatenate([client_data[i][0] for i in range(num_clients)], axis=0)
    all_y = np.concatenate([client_data[i][1] for i in range(num_clients)], axis=0)

    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for k in range(NUM_CLASSES):
        idx_k = np.where(all_y == k)[0]
        rng.shuffle(idx_k)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        splits = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_batch = np.split(idx_k, splits)
        for cid in range(num_clients):
            client_indices[cid].extend(idx_batch[cid].tolist())

    non_iid_clients: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for cid in range(num_clients):
        idxs = np.array(client_indices[cid], dtype=np.int64)
        if len(idxs) == 0:
            # Güvenlik: hiçbir istemci tamamen boş kalmasın
            idxs = rng.choice(len(all_y), size=10, replace=False)
        rng.shuffle(idxs)
        non_iid_clients[cid] = (all_x[idxs], all_y[idxs])

    return non_iid_clients


def get_client_class_distribution(
    client_data: Dict[int, Tuple[np.ndarray, np.ndarray]]
) -> np.ndarray:
    """
    (num_clients, num_classes) boyutunda örnek sayısı matrisi döndürür.
    """
    mat = np.zeros((len(client_data), NUM_CLASSES), dtype=np.int64)
    for cid, (_, y) in client_data.items():
        uniq, cnts = np.unique(y, return_counts=True)
        for u, c in zip(uniq, cnts):
            mat[cid, int(u)] = c
    return mat


def to_dataloader(
    x: np.ndarray, y: np.ndarray, batch_size: int = 32, shuffle: bool = True
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
