# -*- coding: utf-8 -*-
"""
mmWave MIMO Radar Range-Azimuth (256x63) Konum Tanıma için 2D CNN Mimarisi.

Not: Federated Learning (özellikle Non-IID senaryolarda) sırasında BatchNormalization
katmanlarının hareketli ortalama/varyans (running_mean, running_var) istatistikleri
istemciler arasında sapmaya (client drift) yol açar. Bu nedenle TFF uyarısında da
belirtildiği üzere GroupNorm kullanılarak hem kararlılık hem de doğruluk artırılmıştır.
"""

import torch
import torch.nn as nn


class RadarPositionCNN(nn.Module):
    """
    Giriş boyutu: (Batch, 1, 256, 63) -> Range (256 bin) x Azimuth (63 bin)
    Çıkış boyutu: (Batch, num_classes=10)
    """

    def __init__(self, num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Blok 1: (1, 256, 63) -> (24, 64, 16)
            nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Blok 2: (24, 64, 16) -> (48, 16, 4)
            nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Blok 3: (48, 16, 4) -> (96, 4, 2)
            nn.Conv2d(48, 96, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=96),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 2, 160),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(160, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
