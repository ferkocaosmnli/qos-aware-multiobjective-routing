# src/experiments/run_experiments.py
from __future__ import annotations

import sys
import csv
from pathlib import Path
from typing import List, Dict, Optional

# --- 1. PYTHON YOL AYARI ---
# Bu ayar, Python'un 'src' klasörünü ve alt modülleri görmesini garanti eder.
# VS Code'daki sarı çizgilerin asıl ilacı budur.
current_file = Path(__file__).resolve()
src_path = current_file.parents[1]  # 'src' klasörüne gider
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# --- 2. TEMEL MODÜLLER ---
try:
    from network.io import load_graph_teacher_csv
    from algorithms.base import RoutingAlgorithm, PathResult
except ImportError as e:
    print(f"\n❌ KRİTİK HATA: Temel modüller yüklenemedi. ({e})")
    sys.exit(1)

# --- 3. ALGORİTMA IMPORTLARI ---
# Artık try-except karmaşası yok, doğrudan senin klasör yapını kullanıyoruz.

# A) Genetik Algoritma
try:
    from algorithms.metaheuristics.ga import GeneticAlgorithm
except ImportError:
    GeneticAlgorithm = None
    print("⚠️ UYARI: GeneticAlgorithm (ga.py) bulunamadı.")

# B) SARSA (Klasör: algorithms/rl/sarsa.py)
try:
    from algorithms.rl.sarsa import SarsaRouting
except ImportError:
    SarsaRouting = None
    print("⚠️ UYARI: SARSA (rl/sarsa.py) bulunamadı.")

# C) Q-Learning (Klasör: algorithms/rl/q_learning.py)
try:
    from algorithms.rl.q_learning import QLearningRouting
except ImportError:
    QLearningRouting = None
    print("⚠️ UYARI: Q-Learning (rl/q_learning.py) bulunamadı.")

# D) Simulated Annealing (Klasör: algorithms/metaheuristics/sa.py)
try:
    from algorithms.metaheuristics.sa import SimulatedAnnealing
except ImportError:
    SimulatedAnnealing = None
    # SA henüz oluşturulmadıysa uyarı vermesine gerek yok, sessizce geçsin.
    pass 


# --- 4. SENARYOLAR ---
try:
    from experiments.scenarios import SCENARIOS, Scenario, load_scenarios_from_csv
except ImportError:
    SCENARIOS = []
    load_scenarios_from_csv = None
    print("⚠️ UYARI: Senaryo dosyaları yüklenemedi.")


def default_algorithms() -> List[RoutingAlgorithm]:
    """Çalıştırılacak algoritmaları listeler."""
    algos = []
    
    if GeneticAlgorithm:
        algos.append(GeneticAlgorithm())
    
    if SarsaRouting:
        algos.append(SarsaRouting())
        
    if QLearningRouting:
        algos.append(QLearningRouting())

    if SimulatedAnnealing:
        algos.append(SimulatedAnnealing())
    
    if not algos:
        print("\n❌ HATA: Hiçbir algoritma listeye eklenemedi!")
    
    return algos


def run_all_experiments(
    algorithms: Optional[List[RoutingAlgorithm]] = None,
    num_repeats: int = 5,
    output_csv: str | Path = "experiment_results.csv",
) -> None:
    """Tüm senaryolar ve algoritmalar için deneyleri çalıştırır."""
    
    if algorithms is None:
        algorithms = default_algorithms()

    if not algorithms:
        print("❌ HATA: Çalıştırılacak algoritma yok.")
        return

    # Proje kök dizinini bul
    project_root = current_file.parents[2]
    csv_dir = project_root / "graph_csv"

    # Grafı yükle
    try:
        graph = load_graph_teacher_csv(
            csv_dir / "NodeData.csv",
            csv_dir / "EdgeData.csv",
        )
    except FileNotFoundError:
        print(f"\n❌ HATA: Grafik CSV dosyaları bulunamadı: {csv_dir}")
        return

    # Senaryoları yükle
    scenarios = []
    if load_scenarios_from_csv:
        try:
            scenarios = load_scenarios_from_csv(csv_dir / "DemandData.csv")
        except FileNotFoundError:
            print("⚠️ UYARI: DemandData.csv yok, varsayılan senaryolar kullanılıyor.")
            scenarios = SCENARIOS
    else:
        scenarios = SCENARIOS

    # Sabit Ağırlıklar
    weights: Dict[str, float] = {
        "delay": 0.5,
        "reliability": 0.3,
        "resource": 0.2,
    }

    output_path = Path(output_csv)
    print(f"\n🚀 Deneyler Başlatılıyor...")
    print(f"   Algoritmalar: {[a.name for a in algorithms]}")
    print(f"   Senaryo Sayısı: {len(scenarios)}")
    print(f"   Tekrar Sayısı: {num_repeats}")

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "algorithm", "scenario_index", "run_id", "source", "dest",
                "bandwidth_requirement", "path", "cost", 
                "delay_ms", "reliability_cost", "resource_cost",
            ]
        )

        for alg in algorithms:
            print(f"\n>> {alg.name} çalışıyor...")
            for idx, sc in enumerate(scenarios):
                # İlerleme durumu
                print(f"   Senaryo {idx+1}/{len(scenarios)}", end="\r")
                
                for run_id in range(num_repeats):
                    try:
                        result: PathResult = alg.run(
                            graph,
                            source=sc.s,
                            dest=sc.d,
                            weights=weights,
                            bandwidth_requirement=sc.bandwidth_requirement,
                            seed=run_id, 
                        )
                        
                        # Güvenli cost okuma
                        final_cost = getattr(result, "cost", 0.0)

                        writer.writerow(
                            [
                                alg.name,
                                idx,
                                run_id,
                                sc.s,
                                sc.d,
                                sc.bandwidth_requirement,
                                result.path,
                                final_cost,
                                result.details.get("total_delay", 0.0),
                                result.details.get("reliability_cost", 0.0),
                                result.details.get("resource_cost", 0.0),
                            ]
                        )
                    except Exception as e:
                        print(f"\n   ❌ HATA: {alg.name} (Senaryo {idx}): {e}")

    print(f"\n\n✅ İşlem Tamam! Sonuçlar kaydedildi: {output_path.resolve()}")


if __name__ == "__main__":
    run_all_experiments()