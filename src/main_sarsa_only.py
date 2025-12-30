from __future__ import annotations

import random
from pathlib import Path
import sys
import os
import time

# Proje ana dizinini path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.network.io import load_graph_teacher_csv
from src.network.api import is_valid_path, evaluate_path_with_weights

# SARSA Modülünü Çağırıyoruz
try:
    from src.algorithms.rl.sarsa import SARSARouting, SARSAConfig
except ImportError:
    # Eğer rl klasörü yoksa veya path sorunu varsa uyar
    print("HATA: 'src/algorithms/rl/sarsa.py' bulunamadı.")
    print("Lütfen 'rl' klasörü oluşturup sarsa.py'yi oraya attığınızdan emin olun.")
    sys.exit(1)

def main() -> None:
    print("\n" + "="*60)
    print("      SARSA (Reinforcement Learning) - TEST SENARYOSU")
    print("="*60 + "\n")
    
    # 1. Hazırlık
    seed = 42 
    random.seed(seed)
    
    # PDF'deki varsayılan ağırlıklar
    weights = {
        "delay": 0.5,       
        "reliability": 0.3, 
        "resource": 0.2     
    }

    # Dosya yolları
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent 
    csv_dir = project_root / "graph_csv"

    # Grafı Yükle
    try:
        graph = load_graph_teacher_csv(
            csv_dir / "NodeData.csv",
            csv_dir / "EdgeData.csv",
        )
        print(f"✅ Graf Yüklendi: {len(graph.nodes)} düğüm.")
    except FileNotFoundError:
        print("❌ HATA: CSV dosyaları bulunamadı.")
        return

    # Kaynak ve Hedef (Web GUI Scenario 0 ile aynı: 8 -> 44)
    source = 8
    dest = 44

    print("-" * 40)
    print(f"Kaynak ID: {source}")
    print(f"Hedef  ID: {dest}")
    print("-" * 40)

    print("🚀 SARSA Eğitimi Başlıyor (Biraz sürebilir)...")
    start_time = time.time()

    # --- KONFİGÜRASYON ---
    # Test için episode sayısını biraz düşürebiliriz ama 
    # iyi sonuç için genelde yüksek olması gerekir.
    config = SARSAConfig(
        episodes=1000,      # Test için 1000 (Normalde 2000+)
        alpha=0.1,          # Learning rate
        gamma=0.95,         # Discount factor
        epsilon_start=0.9,  # Başta çok keşfet
        epsilon_end=0.05,
        verbose_every=100   # Her 100 turda bilgi ver
    )
    
    solver = SARSARouting(config=config, verbose=True)
    
    # Çalıştır
    result = solver.run(
        graph=graph,
        source=source,
        dest=dest,
        weights=weights,
        seed=seed
    )
    
    elapsed = time.time() - start_time
    print(f"\n✅ Eğitim Tamamlandı ({elapsed:.2f} saniye)")

    # Sonuçları Göster
    path = result.path

    if not path:
        print("\n❌ Sonuç: Rota BULUNAMADI.")
        return

    print(f"\n📍 BULUNAN ROTA: {path}")
    print(f"👣 Adım Sayısı (Hop): {len(path) - 1}")

    if not is_valid_path(graph, path):
        print("❌ HATA: Rota kopuk veya geçersiz!")
        return

    # Metrikleri Hesapla
    total, details = evaluate_path_with_weights(
        graph,
        path,
        w_delay=weights["delay"],
        w_reliability=weights["reliability"],
        w_resource=weights["resource"],
    )

    print("\n" + "-"*35)
    print("   📊 SONUÇ METRİKLERİ")
    print("-"*35)
    print(f"  ⏱️  Gecikme (ms)         : {details['delay_ms']:.3f}")
    print(f"  🛡️  Güvenilirlik Maliyeti: {details['reliability_cost']:.6f}")
    print(f"  💾  Kaynak Maliyeti      : {details['resource_cost']:.6f}")
    print("-" * 35)
    print(f"  🏆  TOTAL COST           : {total:.6f}")
    print("-" * 35)

if __name__ == "__main__":
    main()