# -*- coding: utf-8 -*-
"""
Federated Learning (FedAvg, FedProx) ve Merkezi (Centralized) Eğitim Motoru.

Desteklenen Algoritmalar:
1. Centralized (Merkezi): Tüm istemcilerin verisi tek bir havuzda toplanır (gizlilik yok, üst sınır referansı).
2. FedAvg (Federated Averaging): Her istemci lokal olarak E epoch eğitilir, sunucu örnek sayısı
   ağırlıklı ortalama (weighted averaging) alır.
3. FedProx: Non-IID veri dağılımında istemci sapmasını (client drift) önlemek için lokal kayıp
   fonksiyonuna proksimal terim ekler: L_prox(w) = L(w) + (mu / 2) * ||w - w_global||^2
"""

import copy
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Modeli test kümesi üzerinde değerlendirir.
    Dönüş: (loss, accuracy, y_true, y_pred)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for bx, by in dataloader:
            bx, by = bx.to(device), by.to(device)
            logits = model(bx)
            loss = criterion(logits, by)
            total_loss += loss.item() * by.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == by).sum().item()
            total_samples += by.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(by.cpu().numpy())

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    return (
        avg_loss,
        avg_acc,
        np.concatenate(all_targets, axis=0),
        np.concatenate(all_preds, axis=0),
    )


def train_local_client(
    global_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    local_epochs: int = 2,
    lr: float = 1e-3,
    prox_mu: float = 0.0,
) -> Tuple[Dict[str, torch.Tensor], float, float, int]:
    """
    Bir istemciyi kendi lokal verisi üzerinde eğitir.
    prox_mu > 0 ise FedProx proksimal düzenlileştirme terimi uygulanır.
    """
    local_model = copy.deepcopy(global_model).to(device)
    local_model.train()

    global_params = [p.detach().clone() for p in global_model.parameters()]
    optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for _ in range(local_epochs):
        for bx, by in dataloader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = local_model(bx)
            loss = criterion(logits, by)

            if prox_mu > 0.0:
                prox_term = torch.tensor(0.0, device=device)
                for p_local, p_global in zip(local_model.parameters(), global_params):
                    prox_term = prox_term + torch.sum((p_local - p_global) ** 2)
                loss = loss + (prox_mu / 2.0) * prox_term

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * by.size(0)
            total_correct += (logits.argmax(dim=1) == by).sum().item()
            total_samples += by.size(0)

    avg_loss = total_loss / max(1, total_samples)
    avg_acc = total_correct / max(1, total_samples)
    num_client_examples = len(dataloader.dataset)
    return copy.deepcopy(local_model.state_dict()), avg_loss, avg_acc, num_client_examples


def aggregate_weights(
    client_states: List[Dict[str, torch.Tensor]],
    client_sizes: List[int],
) -> Dict[str, torch.Tensor]:
    """
    İstemci model ağırlıklarını örnek sayılarına göre ağırlıklı olarak birleştirir (FedAvg).
    """
    total_size = float(sum(client_sizes))
    aggregated_state: Dict[str, torch.Tensor] = {}

    for key in client_states[0].keys():
        if client_states[0][key].is_floating_point():
            acc = torch.zeros_like(client_states[0][key], dtype=torch.float32)
            for state, size in zip(client_states, client_sizes):
                weight = float(size) / total_size
                acc = acc + state[key].float() * weight
            aggregated_state[key] = acc.to(client_states[0][key].dtype)
        else:
            aggregated_state[key] = client_states[0][key].clone()

    return aggregated_state


def run_federated_training(
    model_fn,
    client_loaders: Dict[int, DataLoader],
    test_loader: DataLoader,
    device: torch.device,
    num_rounds: int = 20,
    local_epochs: int = 2,
    lr: float = 1e-3,
    prox_mu: float = 0.0,
    clients_per_round: Optional[int] = None,
    extra_test_loaders: Optional[Dict[str, DataLoader]] = None,
    seed: int = 42,
    algo_name: str = "FedAvg",
) -> Tuple[nn.Module, Dict]:
    """
    Tam kapsamlı Federated Learning (FedAvg veya FedProx) eğitim döngüsü.
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    global_model = model_fn().to(device)
    client_ids = sorted(list(client_loaders.keys()))
    if clients_per_round is None or clients_per_round >= len(client_ids):
        clients_per_round = len(client_ids)

    history = {
        "round": [],
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "day_test_acc": {k: [] for k in (extra_test_loaders or {})},
    }

    # Tur 0 (Başlangıç değerlendirmesi)
    init_loss, init_acc, _, _ = evaluate_model(global_model, test_loader, device)
    history["round"].append(0)
    history["train_loss"].append(np.nan)
    history["train_acc"].append(np.nan)
    history["test_loss"].append(init_loss)
    history["test_acc"].append(init_acc)
    if extra_test_loaders:
        for d_name, d_loader in extra_test_loaders.items():
            _, d_acc, _, _ = evaluate_model(global_model, d_loader, device)
            history["day_test_acc"][d_name].append(d_acc)

    print(f"[{algo_name}] Tur 0 (Başlangıç) | Global Test Acc: {init_acc*100:.2f}% | Test Loss: {init_loss:.4f}")

    for rnd in range(1, num_rounds + 1):
        if clients_per_round < len(client_ids):
            selected_ids = rng.choice(client_ids, size=clients_per_round, replace=False).tolist()
        else:
            selected_ids = client_ids

        client_states = []
        client_sizes = []
        round_losses = []
        round_accs = []

        for cid in selected_ids:
            c_state, c_loss, c_acc, c_size = train_local_client(
                global_model=global_model,
                dataloader=client_loaders[cid],
                device=device,
                local_epochs=local_epochs,
                lr=lr,
                prox_mu=prox_mu,
            )
            client_states.append(c_state)
            client_sizes.append(c_size)
            round_losses.append(c_loss * c_size)
            round_accs.append(c_acc * c_size)

        # Sunucu Agregasyonu
        new_global_state = aggregate_weights(client_states, client_sizes)
        global_model.load_state_dict(new_global_state)

        total_round_samples = float(sum(client_sizes))
        train_loss = sum(round_losses) / total_round_samples
        train_acc = sum(round_accs) / total_round_samples

        test_loss, test_acc, _, _ = evaluate_model(global_model, test_loader, device)

        history["round"].append(rnd)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if extra_test_loaders:
            for d_name, d_loader in extra_test_loaders.items():
                _, d_acc, _, _ = evaluate_model(global_model, d_loader, device)
                history["day_test_acc"][d_name].append(d_acc)

        if rnd % 2 == 0 or rnd == 1 or rnd == num_rounds:
            print(
                f"[{algo_name}] Tur {rnd:02d}/{num_rounds:02d} | "
                f"Train Acc: {train_acc*100:.2f}%, Loss: {train_loss:.4f} | "
                f"Test Acc: {test_acc*100:.2f}%, Loss: {test_loss:.4f}"
            )

    return global_model, history


def run_centralized_training(
    model_fn,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    num_epochs: int = 20,
    lr: float = 1e-3,
    seed: int = 42,
) -> Tuple[nn.Module, Dict]:
    """
    Referans karşılaştırması için Geleneksel Merkezi (Centralized) Eğitim.
    """
    torch.manual_seed(seed)
    model = model_fn().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {
        "round": [0],
        "train_loss": [np.nan],
        "train_acc": [np.nan],
        "test_loss": [],
        "test_acc": [],
    }
    init_loss, init_acc, _, _ = evaluate_model(model, test_loader, device)
    history["test_loss"].append(init_loss)
    history["test_acc"].append(init_acc)

    print(f"[Centralized] Epoch 0 (Başlangıç) | Test Acc: {init_acc*100:.2f}%")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * by.size(0)
            total_correct += (logits.argmax(dim=1) == by).sum().item()
            total_samples += by.size(0)

        tr_loss = total_loss / max(1, total_samples)
        tr_acc = total_correct / max(1, total_samples)
        te_loss, te_acc, _, _ = evaluate_model(model, test_loader, device)

        history["round"].append(epoch)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(te_loss)
        history["test_acc"].append(te_acc)

        if epoch % 2 == 0 or epoch == 1 or epoch == num_epochs:
            print(
                f"[Centralized] Epoch {epoch:02d}/{num_epochs:02d} | "
                f"Train Acc: {tr_acc*100:.2f}%, Loss: {tr_loss:.4f} | "
                f"Test Acc: {te_acc*100:.2f}%, Loss: {te_loss:.4f}"
            )

    return model, history
