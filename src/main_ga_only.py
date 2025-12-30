from __future__ import annotations

import random
from pathlib import Path
import sys
import os

# Import sorunu yaşamamak için proje ana dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network.io import load_graph_teacher_csv
from src.network.api import is_valid_path, evaluate_path_with_weights

# --- DÜZELTME 1: GAConfig IMPORT EDİLDİ ---
# Yeni kodda ayarlar bu sınıf üzerinden yapılıyor.
from src.algorithms.metaheuristics.ga import GeneticAlgorithm, GAConfig

def main() -> None:
    print("\n" + "="*50)
    print("      GENETİK ALGORİTMA - OPTİMİZE EDİLMİŞ VERSİYON")
    print("="*50 + "\n")
    
    # Seed Ayarı (Testlerin tutarlı olması için)
    seed = 0  
    random.seed(seed)
    
    # Senaryo Ağırlıkları
    weights = {
        "delay": 0.5,       
        "reliability": 0.3, 
        "resource": 0.2     
    }

    # Veri Setini Yükle
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent 
    csv_dir = project_root / "graph_csv"

    try:
        graph = load_graph_teacher_csv(
            csv_dir / "NodeData.csv",
            csv_dir / "EdgeData.csv",
        )
    except FileNotFoundError:
        print("HATA: CSV dosyaları bulunamadı.")
        return

    # Kaynak ve Hedef (Web GUI ile Eşitlendi: 8 -> 44)
    source = 8
    dest = 44

    print("-" * 40)
    print(f"Kaynak ID: {source}")
    print(f"Hedef  ID: {dest}")
    print("-" * 40)

    print("Optimize edilmiş GA çalıştırılıyor...")
    
    # --- DÜZELTME 2: AYARLAR CONFIG İLE YAPILIYOR ---
    # Eski kod: GeneticAlgorithm(pop_size=50, ...) şeklindeydi ve hata verirdi.
    # Yeni kod: Önce ayar kutusunu (GAConfig) hazırlıyoruz.
    
    config = GAConfig(
        population_size=50,   # Eski kodda pop_size idi
        generations=30,
        mutation_rate=0.2,
        elitism=3,            # Eski kodda elite_size idi
        # Diğer ayarlar default kalabilir veya buraya eklenebilir
    )
    
    # Config nesnesini algoritmaya veriyoruz
    solver = GeneticAlgorithm(config=config)
    
    result = solver.run(
        graph=graph,
        source=source,
        dest=dest,
        weights=weights,
        seed=seed
    )
    
    path = result.path

    if not path:
        print("Sonuç: Rota Bulunamadı.")
        return

    print(f"\nBULUNAN ROTA: {path}")
    print(f"Adım Sayısı (Hop): {len(path) - 1}")

    if not is_valid_path(graph, path):
        print("HATA: Rota geçersiz!")
        return

    # Sonuçları Hesapla
    total, details = evaluate_path_with_weights(
        graph,
        path,
        w_delay=weights["delay"],
        w_reliability=weights["reliability"],
        w_resource=weights["resource"],
    )

    print("\n" + "-"*30)
    print("   SONUÇ METRİKLERİ")
    print("-"*30)
    print(f"  Gecikme (ms)         : {details['delay_ms']:.3f}")
    print(f"  Güvenilirlik Maliyeti: {details['reliability_cost']:.6f}")
    print(f"  Kaynak Maliyeti      : {details['resource_cost']:.6f}")
    print("-" * 30)
    print(f"  TOTAL COST           : {total:.6f}")
    print("-" * 30)

if __name__ == "__main__":
    main()