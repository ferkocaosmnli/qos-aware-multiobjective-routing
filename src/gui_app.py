from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import List

# --- DÜZELTME 1: İçe Aktarmalar (Bizim yapımıza uygun hale getirildi) ---
# Eğer 'src' klasörü ana dizindeyse 'src.' ön eki eklenmelidir.
# Hata alırsan 'from network.io' şeklinde eski haline getirebilirsin.
try:
    from src.network.io import load_graph_teacher_csv
    from src.experiments.scenarios import load_scenarios_from_csv, Scenario
    from src.algorithms.base import RoutingAlgorithm
    from src.network.api import is_valid_path
    # Yeni algoritma yöneticimizi bağlıyoruz:
    from src.algorithms import list_available_algorithms, get_algorithm_class
except ImportError:
    # Alternatif: Dosya src içindeyse veya path ayarlıysa
    from network.io import load_graph_teacher_csv
    from experiments.scenarios import load_scenarios_from_csv, Scenario
    from algorithms.base import RoutingAlgorithm
    from network.api import is_valid_path
    from algorithms import list_available_algorithms, get_algorithm_class


class RoutingGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("QoS Routing Demo (GA & SARSA)")
        self.geometry("950x650")

        # Proje kök ve CSV klasörü
        # gui_app.py'nin proje kök dizininde olduğunu varsayıyoruz
        self.project_root = Path(__file__).resolve().parent
        self.csv_dir = self.project_root / "graph_csv"

        self._load_data()
        
        # Algoritmaları Yükle (Otomatik)
        self.algorithms: List[RoutingAlgorithm] = []
        self._init_algorithms()

        self._build_widgets()

    def _load_data(self):
        """Veri yükleme işlemi hata kontrolü ile yapılır"""
        try:
            if not self.csv_dir.exists():
                messagebox.showwarning("Uyarı", f"CSV klasörü bulunamadı:\n{self.csv_dir}")
                self.graph = None
                self.scenarios = []
                return

            self.graph = load_graph_teacher_csv(
                self.csv_dir / "NodeData.csv",
                self.csv_dir / "EdgeData.csv",
            )
            self.scenarios: List[Scenario] = load_scenarios_from_csv(
                self.csv_dir / "DemandData.csv"
            )
        except Exception as e:
            messagebox.showerror("Veri Yükleme Hatası", f"Dosyalar okunurken hata oluştu:\n{e}")
            self.graph = None
            self.scenarios = []

    def _init_algorithms(self):
        """__init__.py içindeki registry'den algoritmaları çeker ve başlatır."""
        available_names = list_available_algorithms()
        for name in available_names:
            try:
                algo_class = get_algorithm_class(name)
                # Algoritma sınıfından bir örnek (instance) oluşturuyoruz
                instance = algo_class() 
                self.algorithms.append(instance)
            except Exception as e:
                print(f"Hata: {name} başlatılamadı - {e}")

    def _build_widgets(self) -> None:
        # Üst çerçeve: seçimler
        top_frame = ttk.LabelFrame(self, text="Konfigürasyon", padding=10)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # 1. Algoritma Seçimi
        ttk.Label(top_frame, text="Algoritma:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        alg_names = [alg.name for alg in self.algorithms] if self.algorithms else ["Yüklü Algoritma Yok"]
        
        self.alg_var = tk.StringVar(value=alg_names[0] if alg_names else "")
        self.alg_combo = ttk.Combobox(
            top_frame,
            textvariable=self.alg_var,
            values=alg_names,
            state="readonly",
            width=25,
        )
        self.alg_combo.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # 2. Senaryo Seçimi
        ttk.Label(top_frame, text="Senaryo (Demand ID):").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.scen_var = tk.IntVar(value=0)
        scen_values = list(range(len(self.scenarios))) if self.scenarios else []
        
        self.scen_combo = ttk.Combobox(
            top_frame,
            textvariable=self.scen_var,
            values=scen_values,
            state="readonly",
            width=10,
        )
        self.scen_combo.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # 3. Ağırlık Slider'ları
        slider_frame = ttk.Frame(top_frame)
        slider_frame.grid(row=0, column=2, rowspan=3, padx=20, sticky=tk.NSEW)

        self.w_delay = tk.DoubleVar(value=0.33)
        self.w_reliability = tk.DoubleVar(value=0.33)
        self.w_resource = tk.DoubleVar(value=0.34)

        def create_slider(parent, label, var, row):
            lbl = ttk.Label(parent, text=label)
            lbl.grid(row=row, column=0, sticky=tk.W)
            scl = tk.Scale(parent, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, variable=var, length=120)
            scl.grid(row=row, column=1, padx=5)

        create_slider(slider_frame, "Gecikme (Delay):", self.w_delay, 0)
        create_slider(slider_frame, "Güvenilirlik (Rel):", self.w_reliability, 1)
        create_slider(slider_frame, "Kaynak (Res):", self.w_resource, 2)

        # Çalıştır Butonu
        run_button = ttk.Button(top_frame, text="🚀 SİMÜLASYONU BAŞLAT", command=self.on_run_clicked)
        run_button.grid(row=2, column=0, columnspan=2, pady=15, sticky=tk.EW)

        # Sonuç alanı
        result_frame = ttk.LabelFrame(self, text="Sonuçlar ve Metrikler", padding=10)
        result_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.result_text = tk.Text(result_frame, height=20, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(result_frame, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Alt bilgi
        status_bar = ttk.Label(self, text="Hazır.", relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def on_run_clicked(self) -> None:
        if not self.graph or not self.scenarios:
            messagebox.showerror("Veri Hatası", "Graph veya Senaryo verileri yüklenemedi.\nLütfen 'graph_csv' klasörünü kontrol edin.")
            return

        # Algoritma seç
        alg_name = self.alg_var.get()
        alg = next((a for a in self.algorithms if a.name == alg_name), None)
        
        if alg is None:
            messagebox.showerror("Hata", "Lütfen geçerli bir algoritma seçin.")
            return

        # Senaryo seç
        try:
            scen_index = int(self.scen_var.get())
            scenario = self.scenarios[scen_index]
        except (ValueError, IndexError):
            messagebox.showerror("Hata", "Geçersiz senaryo seçimi.")
            return

        # Ağırlıklar
        wd = self.w_delay.get()
        wr = self.w_reliability.get()
        wres = self.w_resource.get()
        total = wd + wr + wres
        
        if total <= 0:
            messagebox.showwarning("Uyarı", "Ağırlıkların toplamı 0 olamaz. Varsayılan (0.33) değerler kullanılacak.")
            wd = wr = wres = 0.33
            total = 1.0

        # Normalize et
        weights = {
            "delay": wd / total,
            "reliability": wr / total,
            "resource": wres / total
        }

        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"⏳ {alg.name} çalıştırılıyor...\n")
        self.result_text.configure(state=tk.DISABLED)
        self.update() # Arayüzü güncelle

        # --- ÇALIŞTIRMA ---
        try:
            import time
            start_t = time.time()
            
            result = alg.run(
                self.graph,
                source=scenario.s,
                dest=scenario.d,
                weights=weights,
                bandwidth_requirement=scenario.bandwidth_requirement,
                seed=42, # Tekrar edilebilirlik için
            )
            
            elapsed = time.time() - start_t
        except Exception as exc:
            messagebox.showerror("Algoritma Hatası", f"Çalışma sırasında hata oluştu:\n{exc}")
            import traceback
            traceback.print_exc()
            return

        # Sonuçları Yazdır
        self._display_results(alg, scenario, result, weights, elapsed)

    def _display_results(self, alg, scenario, result, weights, elapsed):
        lines = []
        lines.append("="*50)
        lines.append(f"ALGORİTMA: {alg.name}")
        lines.append(f"SENARYO  : #{self.scen_var.get()} (S: {scenario.s} -> D: {scenario.d})")
        lines.append(f"BANDWIDTH: {scenario.bandwidth_requirement} Mbps")
        lines.append("-" * 50)
        lines.append(f"AĞIRLIKLAR: D={weights['delay']:.2f}, R={weights['reliability']:.2f}, B={weights['resource']:.2f}")
        lines.append("-" * 50)
        
        if not result.path:
            lines.append("❌ SONUÇ: HEDEF DÜĞÜME ULAŞILAMADI!")
        else:
            lines.append("✅ SONUÇ: BAŞARILI")
            lines.append(f"📍 YOL: {result.path}")
            lines.append(f"📏 HOP SAYISI: {len(result.path) - 1}")
            lines.append(f"💰 TOPLAM MALİYET: {result.total_cost:.6f}")
            lines.append("")
            lines.append("DETAYLI METRİKLER:")
            for k, v in result.details.items():
                lines.append(f"   • {k}: {v}")
        
        lines.append("")
        lines.append(f"⏱️ HESAPLAMA SÜRESİ: {elapsed:.4f} saniye")
        lines.append("="*50)

        self._set_result_text("\n".join(lines))

    def _set_result_text(self, text: str) -> None:
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state=tk.DISABLED)


def main() -> None:
    app = RoutingGUI()
    app.mainloop()


if __name__ == "__main__":
    main()