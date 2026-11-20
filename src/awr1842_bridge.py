# -*- coding: utf-8 -*-
"""
Texas Instruments AWR1842 / AWR1642 mmWave MIMO Radar Gerçek Zamanlı İşleme ve FL Tahmin Köprüsü.

Bu modül, `Colab Notebooks/Radar/AWR1842-python_for_raspberryPI4` klasöründeki
gerçek zamanlı seri port / TLV paket ayrıştırma mantığını (`readAndParseData18xx`)
eğitilen Federated Learning modeli (`RadarPositionCNN`) ile birleştirir:
1. AWR1842 konfigürasyon dosyasını (`profile_usb2x4_working_fps4_doppler+heat.cfg`) ayrıştırır.
2. Ham TLV Azimuth Static Heatmap (2Tx x 4Rx = 8 sanal anten x 256 Range bin) karmaşık FFT
   verisini tam olarak (256, 63) boyutunda Range-Azimuth ısı haritasına dönüştürür.
3. Eğitilmiş PyTorch Global Modeli üzerinde gerçek zamanlı konum sınıfı (Sınıf 0..9) ve
   olasılık (confidence) tahmini yapar.
"""

import os
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from src.dataset import normalize_radar_samples

DEFAULT_CFG_PATH = r"C:\Users\enyas\Desktop\Colab Notebooks\Radar\AWR1842-python_for_raspberryPI4\profile_usb2x4_working_fps4_doppler+heat.cfg"


def parse_awr1842_config(config_file: str = DEFAULT_CFG_PATH) -> Dict[str, float]:
    """
    AWR1842 .cfg dosyasını okuyarak radar fiziksel parametrelerini hesaplar:
    - numRangeBins (256)
    - numDopplerBins (16)
    - rangeResolutionMeters (~0.044 m)
    - maxRange (~9.02 m)
    - maxVelocity (~1.0 m/s)
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Radar cfg dosyası bulunamadı: {config_file}")

    num_rx_ant = 4
    num_tx_ant = 2
    start_freq = 77
    idle_time = 429
    ramp_end_time = 57.14
    freq_slope_const = 70.0
    num_adc_samples = 256
    num_adc_round2 = 256
    dig_out_sample_rate = 5209
    chirp_start_idx = 0
    chirp_end_idx = 1
    num_loops = 16

    with open(config_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("%")]

    for line in lines:
        parts = line.split()
        if parts[0] == "profileCfg":
            start_freq = int(float(parts[2]))
            idle_time = int(parts[3])
            ramp_end_time = float(parts[5])
            freq_slope_const = float(parts[8])
            num_adc_samples = int(parts[10])
            num_adc_round2 = 1
            while num_adc_samples > num_adc_round2:
                num_adc_round2 *= 2
            dig_out_sample_rate = int(parts[11])
        elif parts[0] == "frameCfg":
            chirp_start_idx = int(parts[1])
            chirp_end_idx = int(parts[2])
            num_loops = int(parts[3])

    num_chirps_per_frame = (chirp_end_idx - chirp_start_idx + 1) * num_loops
    num_doppler_bins = int(num_chirps_per_frame / num_tx_ant)
    num_range_bins = num_adc_round2

    range_res_m = (3e8 * dig_out_sample_rate * 1e3) / (2 * freq_slope_const * 1e12 * num_adc_samples)
    range_idx_to_m = (3e8 * dig_out_sample_rate * 1e3) / (2 * freq_slope_const * 1e12 * num_range_bins)
    doppler_res_mps = 3e8 / (
        2 * start_freq * 1e9 * (idle_time + ramp_end_time) * 1e-6 * num_doppler_bins * num_tx_ant
    )
    max_range_m = (300 * 0.9 * dig_out_sample_rate) / (2 * freq_slope_const * 1e3)
    max_vel_mps = 3e8 / (4 * start_freq * 1e9 * (idle_time + ramp_end_time) * 1e-6 * num_tx_ant)

    return {
        "numTxAnt": num_tx_ant,
        "numRxAnt": num_rx_ant,
        "numVirtualAnt": num_tx_ant * num_rx_ant,
        "numRangeBins": num_range_bins,
        "numAngleBins": 63,  # 64-nokta açı FFT'sinden ilk sütun atılarak 63 bin elde edilir
        "numDopplerBins": num_doppler_bins,
        "rangeResolutionMeters": round(range_res_m, 4),
        "rangeIdxToMeters": round(range_idx_to_m, 4),
        "dopplerResolutionMps": round(doppler_res_mps, 4),
        "maxRangeMeters": round(max_range_m, 2),
        "maxVelocityMps": round(max_vel_mps, 2),
    }


def raw_tlv_azimuth_to_heatmap(
    payload_int16: np.ndarray,
    num_range_bins: int = 256,
    num_virtual_ant: int = 8,
    num_angle_bins: int = 64,
) -> np.ndarray:
    """
    AWR1842 TLV Tipi 4 (MMWDEMO_OUTPUT_MSG_AZIMUT_STATIC_HEAT_MAP) int16 karmaşık
    payload dizisini (256, 63) boyutlu Range-Azimuth matrisine dönüştürür.
    (`MIMOradar_raspberry_range_azimuth_doppler_v3_nodisplay.py` satır 269-280 ile birebir uyumlu).
    """
    complex_ant = payload_int16[::2].astype(np.float32) + 1j * payload_int16[1::2].astype(np.float32)
    range_azimuth = np.reshape(complex_ant, (num_virtual_ant, num_range_bins), order="F")
    q = np.fft.fft(range_azimuth, n=num_angle_bins, axis=0)
    q = np.fft.fftshift(np.abs(q), axes=(0,))
    q = q.T  # (256, 64)
    q = q[:, 1:]  # (256, 63)
    heatmap = np.fliplr(q).copy()
    return heatmap


def predict_radar_frame(
    model: nn.Module,
    range_azimuth_map: np.ndarray,
    device: torch.device,
    norm_method: str = "zscore",
) -> Tuple[int, float, np.ndarray]:
    """
    Tek bir (256, 63) Range-Azimuth radar çerçevesi için eğitilmiş FL modelinden
    konum sınıfı (0..9) ve güven skoru döndürür.
    """
    if range_azimuth_map.ndim == 2:
        sample = np.expand_dims(range_azimuth_map, axis=0)  # (1, 256, 63)
    elif range_azimuth_map.ndim == 3 and range_azimuth_map.shape[0] == 1:
        sample = range_azimuth_map
    else:
        raise ValueError(f"Beklenen giriş boyutu (256, 63), alınan: {range_azimuth_map.shape}")

    norm_sample = normalize_radar_samples(sample, method=norm_method)
    tensor_x = torch.from_numpy(norm_sample).unsqueeze(1).to(device)  # (1, 1, 256, 63)

    model.eval()
    with torch.no_grad():
        logits = model(tensor_x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
        confidence = float(probs[pred_class])

    return pred_class, confidence, probs
