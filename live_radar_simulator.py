# -*- coding: utf-8 -*-
"""
Canlı Radar Konum Tanıma Simülasyon & İnteraktif Masaüstü Arayüzü (Tkinter Native GUI)
(Live mmWave MIMO Radar Position Recognition Simulator - Native Desktop Application)

Not: PyCharm'ın "Python Plots / Scientific Mode" özelliği standart `plt.show()` çağrılarını
yakalayıp sağ panelde statik (tıklanamayan) bir PNG olarak gösterdiği için, bu simülatör
doğrudan yerel **Tkinter Masaüstü Penceresi (`FigureCanvasTkAgg`)** ve gerçek **Tkinter Butonları**
ile çalışacak şekilde tasarlanmıştır. Böylece PyCharm içinden çalıştırıldığında bile her zaman
bağımsız, tam interaktif bir masaüstü uygulaması olarak açılır.
"""

import argparse
import os
import time
import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.dataset import DEFAULT_RADAR_DATA_DIR, NUM_CLASSES, load_all_days_dataset
from src.models import RadarPositionCNN
from src.awr1842_bridge import DEFAULT_CFG_PATH, parse_awr1842_config, predict_radar_frame


# İnsan-Robot Ortak Çalışma Alanındaki 10 Konum Bölgesinin (Sınıf 0..9)
# Yaklaşık 2D Kartezyen Koordinatları (X: Yatay Metre, Y: Menzil Metre)
WORKSPACE_POSITIONS = {
    0: (-1.5, 1.5, "Bölge 0 (Sol-Yakın)"),
    1: (-0.5, 1.5, "Bölge 1 (OrtaSol-Yakın)"),
    2: (0.5, 1.5, "Bölge 2 (OrtaSağ-Yakın)"),
    3: (1.5, 1.5, "Bölge 3 (Sağ-Yakın)"),
    4: (-1.5, 2.8, "Bölge 4 (Sol-Orta)"),
    5: (0.0, 2.8, "Bölge 5 (Merkez-Robot Yanı)"),
    6: (1.5, 2.8, "Bölge 6 (Sağ-Orta)"),
    7: (-1.2, 4.0, "Bölge 7 (Sol-Uzak)"),
    8: (0.0, 4.0, "Bölge 8 (Merkez-Uzak)"),
    9: (1.2, 4.0, "Bölge 9 (Sağ-Uzak)"),
}


class RadarSimulatorEngine:
    """Hem Tkinter GUI hem de headless PNG kaydı için ortak çizim ve tahmin motoru."""

    def __init__(
        self,
        model: torch.nn.Module,
        test_x: np.ndarray,
        test_y: np.ndarray,
        radar_cfg: dict,
        device: torch.device,
    ):
        self.model = model
        self.test_x = test_x
        self.test_y = test_y
        self.radar_cfg = radar_cfg
        self.device = device
        self.rng = np.random.default_rng(42)

        self.current_idx = 0
        self.total_tested = 0
        self.total_correct = 0
        self.filter_class = None  # None = Tüm sınıflar, 0..9 = Seçili sınıf

        self.fig = Figure(figsize=(15.5, 6.8), dpi=100)
        gs = self.fig.add_gridspec(
            1, 3, width_ratios=[1.15, 1.15, 1.0], bottom=0.12, top=0.84, left=0.05, right=0.97, wspace=0.28
        )
        self.ax_heatmap = self.fig.add_subplot(gs[0])
        self.ax_workspace = self.fig.add_subplot(gs[1])
        self.ax_probs = self.fig.add_subplot(gs[2])
        self.cbar = None

    def pick_next_index(self) -> int:
        if self.filter_class is None:
            return int(self.rng.integers(0, len(self.test_y)))
        matching = np.where(self.test_y == self.filter_class)[0]
        return int(self.rng.choice(matching))

    def step_frame(self, initial: bool = False) -> dict:
        if not initial:
            self.current_idx = self.pick_next_index()
        else:
            self.current_idx = 0

        frame_2d = self.test_x[self.current_idx, 0]  # (256, 63)
        true_cls = int(self.test_y[self.current_idx])

        t0 = time.perf_counter()
        pred_cls, conf, probs = predict_radar_frame(self.model, frame_2d, self.device, norm_method="zscore")
        inf_ms = (time.perf_counter() - t0) * 1000.0

        self.total_tested += 1
        is_correct = pred_cls == true_cls
        if is_correct:
            self.total_correct += 1
        running_acc = (self.total_correct / self.total_tested) * 100.0

        status_txt = "DOĞRU TAHMİN ✓" if is_correct else "HATALI TAHMİN ✗"
        status_color = "#1b5e20" if is_correct else "#b71c1c"
        self.fig.suptitle(
            f"TI AWR1842 (77 GHz) Canlı Radar Akışı | Çerçeve #{self.current_idx} | "
            f"Gerçek: Sınıf {true_cls} → Tahmin: Sınıf {pred_cls} ({status_txt})\n"
            f"Model Güveni: %{conf*100:.2f} | Çıkarım Süresi: {inf_ms:.2f} ms | "
            f"Canlı Oturum Doğruluğu: %{running_acc:.1f} ({self.total_correct}/{self.total_tested})",
            fontsize=12,
            fontweight="bold",
            color=status_color,
        )

        # 1. Sol Panel: Range-Azimuth Isı Haritası
        self.ax_heatmap.clear()
        max_r = self.radar_cfg["maxRangeMeters"]
        im = self.ax_heatmap.imshow(
            frame_2d,
            aspect="auto",
            cmap="viridis",
            origin="lower",
            extent=[-60, 60, 0, max_r],
        )
        self.ax_heatmap.set_title(
            f"Range-Azimuth Isı Haritası (256x63)\n[Çerçeve #{self.current_idx} - Sınıf {true_cls}]",
            fontsize=11,
            fontweight="bold",
        )
        self.ax_heatmap.set_xlabel("Azimuth Açısı (Derece °)")
        self.ax_heatmap.set_ylabel("Menzil / Range (Metre)")
        if self.cbar is None:
            self.cbar = self.fig.colorbar(im, ax=self.ax_heatmap, fraction=0.046, pad=0.04)

        # 2. Orta Panel: İnsan-Robot Ortak Çalışma Alanı (2D Kuşbakışı Harita)
        self.ax_workspace.clear()
        self.ax_workspace.set_xlim(-2.4, 2.4)
        self.ax_workspace.set_ylim(-0.3, 4.8)
        self.ax_workspace.grid(True, linestyle="--", alpha=0.5)
        self.ax_workspace.set_title(
            "İnsan-Robot Ortak Çalışma Alanı (Kuşbakışı Konum)",
            fontsize=11,
            fontweight="bold",
        )
        self.ax_workspace.set_xlabel("Yatay Konum X (Metre)")
        self.ax_workspace.set_ylabel("Radar Menzili Y (Metre)")

        # Radar Sensörü ve Robot Kol Konumu
        self.ax_workspace.scatter(0, 0, c="black", marker="^", s=180, zorder=5, label="AWR1842 Radar (0,0)")
        self.ax_workspace.scatter(0, 2.2, c="#455a64", marker="s", s=160, zorder=4, label="Endüstriyel Robot")

        for cid, (px, py, _) in WORKSPACE_POSITIONS.items():
            base_color = "#cfd8dc"
            edge_color = "#90a4ae"
            size = 550
            if cid == true_cls and cid == pred_cls:
                base_color = "#66bb6a"  # Yeşil (Doğru eşleşme)
                edge_color = "#1b5e20"
                size = 980
            elif cid == true_cls:
                base_color = "#42a5f5"  # Mavi (Gerçek konum)
                edge_color = "#0d47a1"
                size = 860
            elif cid == pred_cls:
                base_color = "#ef5350"  # Kırmızı (Hatalı tahmin)
                edge_color = "#b71c1c"
                size = 860

            self.ax_workspace.scatter(
                px, py, c=base_color, edgecolors=edge_color, linewidths=2.2, s=size, alpha=0.9, zorder=3
            )
            self.ax_workspace.text(
                px, py, f"S{cid}", ha="center", va="center", fontsize=9, fontweight="bold", color="black", zorder=6
            )

        self.ax_workspace.legend(loc="lower right", fontsize=8.5)

        # 3. Sağ Panel: Sınıf Olasılıkları (Confidence Bar Chart)
        self.ax_probs.clear()
        classes = np.arange(NUM_CLASSES)
        bar_colors = [
            "#2e7d32" if (i == pred_cls and is_correct)
            else "#c62828" if (i == pred_cls and not is_correct)
            else "#1565c0" if i == true_cls
            else "#90a4ae"
            for i in classes
        ]
        bars = self.ax_probs.barh(classes, probs * 100.0, color=bar_colors, edgecolor="black", height=0.65)
        self.ax_probs.set_yticks(classes)
        self.ax_probs.set_yticklabels([f"Sınıf {i}" for i in classes])
        self.ax_probs.set_xlim(0, 110)
        self.ax_probs.set_xlabel("Model Güven Olasılığı (%)")
        self.ax_probs.set_title("FedAvg Global Model Olasılık Dağılımı", fontsize=11, fontweight="bold")
        self.ax_probs.grid(True, axis="x", linestyle="--", alpha=0.5)

        for bar, p in zip(bars, probs):
            if p >= 0.01:
                self.ax_probs.text(
                    bar.get_width() + 1.2,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"%{p*100:.1f}",
                    va="center",
                    fontsize=8.5,
                    fontweight="bold",
                )

        return {
            "frame_idx": self.current_idx,
            "true_cls": true_cls,
            "pred_cls": pred_cls,
            "conf": conf,
            "inf_ms": inf_ms,
            "running_acc": running_acc,
            "is_correct": is_correct,
        }


class DesktopRadarSimulatorApp:
    """PyCharm Scientific Mode tarafından engellenmeyen tam interaktif Tkinter Masaüstü Uygulaması."""

    def __init__(self, engine: RadarSimulatorEngine, preview_save_path: str):
        self.engine = engine
        self.preview_save_path = preview_save_path
        self.is_streaming = False
        self.after_id = None
        self.stream_interval_ms = 800

        self.root = tk.Tk()
        self.root.title("TI AWR1842 mmWave MIMO Radar - Canlı Federated Learning Konum Tanıma Simülatörü")
        self.root.geometry("1520x780")
        self.root.configure(bg="#f5f7fa")

        # Pencereyi ekranın en önüne getir
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(500, lambda: self.root.attributes("-topmost", False))

        # Üst Bilgi ve Durum Çubuğu
        top_frame = tk.Frame(self.root, bg="#1e293b", pady=8, padx=14)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        self.lbl_status = tk.Label(
            top_frame,
            text="Canlı Radar Simülatörü Hazır",
            font=("Segoe UI", 11, "bold"),
            fg="#f8fafc",
            bg="#1e293b",
        )
        self.lbl_status.pack(side=tk.LEFT)

        self.lbl_hint = tk.Label(
            top_frame,
            text="Kısayollar: [Space / Sağ Ok] Sonraki Çerçeve  |  [S] Canlı Akış Başlat/Durdur  |  [R] Sıfırla",
            font=("Segoe UI", 9),
            fg="#94a3b8",
            bg="#1e293b",
        )
        self.lbl_hint.pack(side=tk.RIGHT)

        # Alt Kontrol Paneli (Gerçek Tkinter Butonları ve Kontrolleri)
        ctrl_frame = tk.Frame(self.root, bg="#e2e8f0", pady=10, padx=15, relief=tk.RAISED, bd=1)
        ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.btn_next = tk.Button(
            ctrl_frame,
            text="▶ Sonraki Radar Çerçevesi",
            font=("Segoe UI", 10, "bold"),
            bg="#2563eb",
            fg="white",
            activebackground="#1d4ed8",
            activeforeground="white",
            padx=14,
            pady=6,
            cursor="hand2",
            command=self.on_next_frame,
        )
        self.btn_next.pack(side=tk.LEFT, padx=6)

        self.btn_stream = tk.Button(
            ctrl_frame,
            text="⟳ Canlı Radar Akışını Başlat (Otomatik)",
            font=("Segoe UI", 10, "bold"),
            bg="#16a34a",
            fg="white",
            activebackground="#15803d",
            activeforeground="white",
            padx=14,
            pady=6,
            cursor="hand2",
            command=self.on_toggle_stream,
        )
        self.btn_stream.pack(side=tk.LEFT, padx=6)

        # Sınıf Filtresi Seçici (Combobox)
        tk.Label(
            ctrl_frame,
            text="Konum Sınıfı Filtresi:",
            font=("Segoe UI", 10, "bold"),
            bg="#e2e8f0",
            fg="#1e293b",
        ).pack(side=tk.LEFT, padx=(18, 5))

        filter_options = ["Tüm Sınıflar (Rastgele)"] + [
            f"Sınıf {i} - {WORKSPACE_POSITIONS[i][2]}" for i in range(NUM_CLASSES)
        ]
        self.combo_filter = ttk.Combobox(
            ctrl_frame,
            values=filter_options,
            state="readonly",
            width=30,
            font=("Segoe UI", 10),
        )
        self.combo_filter.current(0)
        self.combo_filter.bind("<<ComboboxSelected>>", self.on_filter_changed)
        self.combo_filter.pack(side=tk.LEFT, padx=5)

        # Akış Hızı Seçici
        tk.Label(
            ctrl_frame,
            text="Akış Hızı:",
            font=("Segoe UI", 10, "bold"),
            bg="#e2e8f0",
            fg="#1e293b",
        ).pack(side=tk.LEFT, padx=(18, 5))

        self.combo_speed = ttk.Combobox(
            ctrl_frame,
            values=["Hızlı (400 ms / 2.5 FPS)", "Normal (800 ms / 1.25 FPS)", "Yavaş (1500 ms)"],
            state="readonly",
            width=22,
            font=("Segoe UI", 10),
        )
        self.combo_speed.current(1)
        self.combo_speed.bind("<<ComboboxSelected>>", self.on_speed_changed)
        self.combo_speed.pack(side=tk.LEFT, padx=5)

        self.btn_reset = tk.Button(
            ctrl_frame,
            text="↺ İstatistikleri Sıfırla",
            font=("Segoe UI", 10),
            bg="#dc2626",
            fg="white",
            activebackground="#b91c1c",
            activeforeground="white",
            padx=12,
            pady=6,
            cursor="hand2",
            command=self.on_reset,
        )
        self.btn_reset.pack(side=tk.RIGHT, padx=6)

        # Matplotlib Canvas'ı Tkinter Penceresine Göm
        canvas_frame = tk.Frame(self.root, bg="white")
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = FigureCanvasTkAgg(self.engine.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Klavye Kısayolları
        self.root.bind("<space>", lambda e: self.on_next_frame())
        self.root.bind("<Right>", lambda e: self.on_next_frame())
        self.root.bind("s", lambda e: self.on_toggle_stream())
        self.root.bind("S", lambda e: self.on_toggle_stream())
        self.root.bind("r", lambda e: self.on_reset())
        self.root.bind("R", lambda e: self.on_reset())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # İlk Çerçeveyi Çiz ve Önizleme PNG'sini Güncelle
        info = self.engine.step_frame(initial=True)
        self._update_status_bar(info)
        self.canvas.draw()
        self.engine.fig.savefig(self.preview_save_path, dpi=150)

    def _update_status_bar(self, info: dict):
        zone_name = WORKSPACE_POSITIONS[info["pred_cls"]][2]
        match_str = "DOĞRU" if info["is_correct"] else "YANLIŞ"
        self.lbl_status.config(
            text=(
                f"Çerçeve #{info['frame_idx']}  |  Gerçek: Sınıf {info['true_cls']}  →  "
                f"Tahmin: {zone_name} [{match_str}]  |  "
                f"Güven: %{info['conf']*100:.2f}  |  Çıkarım: {info['inf_ms']:.2f} ms  |  "
                f"Canlı Doğruluk: %{info['running_acc']:.1f}"
            )
        )

    def on_next_frame(self):
        info = self.engine.step_frame(initial=False)
        self._update_status_bar(info)
        self.canvas.draw_idle()

    def _stream_loop(self):
        if self.is_streaming:
            self.on_next_frame()
            self.after_id = self.root.after(self.stream_interval_ms, self._stream_loop)

    def on_toggle_stream(self):
        self.is_streaming = not self.is_streaming
        if self.is_streaming:
            self.btn_stream.config(
                text="⏸ Canlı Radar Akışını Durdur",
                bg="#d97706",
                activebackground="#b45309",
            )
            self._stream_loop()
        else:
            self.btn_stream.config(
                text="⟳ Canlı Radar Akışını Başlat (Otomatik)",
                bg="#16a34a",
                activebackground="#15803d",
            )
            if self.after_id is not None:
                self.root.after_cancel(self.after_id)
                self.after_id = None

    def on_filter_changed(self, event=None):
        idx = self.combo_filter.current()
        if idx == 0:
            self.engine.filter_class = None
        else:
            self.engine.filter_class = idx - 1
        self.on_next_frame()

    def on_speed_changed(self, event=None):
        idx = self.combo_speed.current()
        if idx == 0:
            self.stream_interval_ms = 400
        elif idx == 1:
            self.stream_interval_ms = 800
        else:
            self.stream_interval_ms = 1500

    def on_reset(self):
        self.engine.total_tested = 0
        self.engine.total_correct = 0
        self.engine.filter_class = None
        self.combo_filter.current(0)
        info = self.engine.step_frame(initial=True)
        self._update_status_bar(info)
        self.canvas.draw_idle()

    def on_close(self):
        self.is_streaming = False
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
        self.root.destroy()

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Canlı Radar Konum Tanıma Simülatörü")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Pencere açmadan simülasyon önizleme görüntüsünü results/live_simulator_preview.png olarak kaydeder.",
    )
    args = parser.parse_args()

    if args.headless:
        matplotlib.use("Agg")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, "best_global_radar_cnn.pt")
    preview_path = os.path.join(results_dir, "live_simulator_preview.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Canlı Radar Simülatörü Başlatılıyor (Cihaz: {device}) ===")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Eğitilmiş model bulunamadı: {model_path}. Lütfen önce 'python train_and_evaluate.py' çalıştırın."
        )

    radar_cfg = parse_awr1842_config(DEFAULT_CFG_PATH)
    model = RadarPositionCNN(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"  Model yüklendi: {model_path}")

    print("  Test veri seti yükleniyor (day0, day1, day2)...")
    _, _, (global_test_x, global_test_y) = load_all_days_dataset(
        base_dir=DEFAULT_RADAR_DATA_DIR,
        days=["day0", "day1", "day2"],
        norm_method="zscore",
    )
    print(f"  Toplam {len(global_test_y)} radar test çerçevesi simülasyona hazır!")

    engine = RadarSimulatorEngine(
        model=model,
        test_x=global_test_x,
        test_y=global_test_y,
        radar_cfg=radar_cfg,
        device=device,
    )

    if args.headless:
        for _ in range(4):
            engine.step_frame(initial=False)
        engine.fig.savefig(preview_path, dpi=150)
        print(f"  Simülatör ekran görüntüsü kaydedildi: {preview_path}")
    else:
        print("  Canlı İnteraktif Masaüstü Penceresi (Tkinter GUI) açılıyor...")
        print("  -> Penceredeki 'Canlı Radar Akışını Başlat' veya 'Sonraki Radar Çerçevesi' butonlarını kullanabilirsiniz.")
        app = DesktopRadarSimulatorApp(engine=engine, preview_save_path=preview_path)
        app.run()


if __name__ == "__main__":
    main()
