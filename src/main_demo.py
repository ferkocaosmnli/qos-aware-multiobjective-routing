from __future__ import annotations

"""
Altyapı Test Senaryosu:
- NodeData ve EdgeData CSV'lerinden ağı yükler.
- DemandData CSV'sinden talepleri okur.
- İlk 5 talebi (veya rastgele 5 tanesini) test eder.
- Yol var mı kontrol eder, varsa maliyetleri hesaplar.
"""

import csv
import random
from pathlib import Path

from network.api import (
    find_any_path,
    evaluate_path_with_weights,
    is_valid_path,
    is_reachable,
)
from network.io import load_graph_teacher_csv


def load_demands(filepath: Path) -> list[dict]:
    """DemandData.csv dosyasını okur.
    Format: src;dst;demand_mbps
    """
    demands = []
    if not filepath.exists():
        print(f"UYARI: {filepath} bulunamadı!")
        return demands

    with filepath.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            demands.append({
                "src": int(row["src"]),
                "dst": int(row["dst"]),
                # Virgüllü sayıyı (12,5) float'a (12.5) çevir
                "demand_mbps": float(row["demand_mbps"].replace(",", "."))
            })
    return demands


def main() -> None:
    # 1. Klasör yollarını ayarla
    project_root = Path(__file__).resolve().parents[1]
    csv_dir = project_root / "graph_csv"

    print(f"--- 1. Ağ Yükleniyor ({csv_dir}) ---")
    try:
        graph = load_graph_teacher_csv(
            csv_dir / "NodeData.csv",
            csv_dir / "EdgeData.csv",
        )
        print(f" -> Başarılı! Toplam Düğüm: {len(graph)}")
    except Exception as e:
        print(f"HATA: Ağ yüklenirken sorun oluştu: {e}")
        return

    # 2. Talepleri Yükle
    print(f"--- 2. Talepler Yükleniyor (DemandData.csv) ---")
    demands = load_demands(csv_dir / "DemandData.csv")
    print(f" -> Toplam Talep Sayısı: {len(demands)}")

    if not demands:
        print("Talep listesi boş veya okunamadı. Test sonlandırılıyor.")
        return

    # 3. Test için örnek seç (İlk 5 talep veya Rastgele 5)
    # Algoritma geliştirirken hepsini döngüye sokacaksın, şimdilik test için 5 tane yeter.
    sample_demands = demands[:5] 
    # İstersen rastgele için: sample_demands = random.sample(demands, min(5, len(demands)))

    print("\n--- 3. Test Başlıyor (İlk 5 Talep) ---")
    
    # Ağırlıklar (Senaryoya göre değişebilir)
    w_delay = 0.33
    w_reliability = 0.33
    w_resource = 0.34

    for i, item in enumerate(sample_demands, 1):
        src = item["src"]
        dst = item["dst"]
        bw = item["demand_mbps"]

        print(f"\n[Test {i}] Kaynak: {src} -> Hedef: {dst} | İstenen Hız: {bw} Mbps")

        # A) Yol Var mı?
        if not is_reachable(graph, src, dst):
            print("  -> ! HATA: Bu iki düğüm arasında fiziksel yol YOK!")
            continue

        # B) Herhangi bir yol bul (BFS) - İleride buraya kendi Algoritmanı koyacaksın!
        path = find_any_path(graph, src, dst)
        
        if path:
            print(f"  -> Yol Bulundu (Hop: {len(path)-1}): {path}")
            
            # C) Maliyeti Hesapla
            total, details = evaluate_path_with_weights(
                graph, path, w_delay, w_reliability, w_resource
            )
            
            print(f"  -> Total Cost: {total:.4f}")
            print(f"     (Gecikme: {details['delay_ms']:.2f}ms, "
                  f"Güvenilirlik Cost: {details['reliability_cost']:.2f}, "
                  f"Kaynak Cost: {details['resource_cost']:.2f})")
        else:
            print("  -> Yol bulunamadı.")

if __name__ == "__main__":
    main()