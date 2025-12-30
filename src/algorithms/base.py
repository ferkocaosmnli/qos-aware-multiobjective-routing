# src/algorithms/base.py

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Dict, Protocol, Optional

# Eğer network.api modülü projenizde yoksa bu import hata verebilir.
# Ancak eski dosyanızda olduğu için koruyoruz.
from network.api import (
    Graph,
    evaluate_path_with_weights,
    is_valid_path,
    find_any_path,
)

@dataclass
class PathResult:
    """Algoritma ekibinin döndüreceği standart sonuç yapısı.

    - path: bulunan yol (düğüm ID'leri listesi)
    - cost: seçilen ağırlıklara göre hesaplanmış nihai skor (Eski adı: total_cost)
    - details: delay_ms, reliability_cost, resource_cost gibi ek bilgiler
    """
    path: List[int]
    cost: float          # DİKKAT: GUI ile uyum için ismini 'cost' yaptık.
    details: Dict[str, float]


class RoutingAlgorithm(Protocol):
    """Tüm yönlendirme algoritmaları için ortak arayüz.

    GA, ACO, Q-Learning, vb. algoritma sınıfları bu interface'i
    kendilerine uyarlayarak kullanmalıdır.
    """

    name: str  # Algoritma ismi (ör: "GeneticAlgorithm")

    def run(
        self,
        graph: Graph,
        source: int,
        dest: int,
        weights: Dict[str, float],
        bandwidth_requirement: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> PathResult:
        """Verilen graph ve S,D için bir yol üretip değerlendirmelidir.
        
        Dönüş:
            PathResult (içinde path, cost ve details dolu olmalı)
        """
        ...

# ---------------------------------------------------------------------------
# YENİ EKLENEN HESAPLAMA FONKSİYONU (PDF'TEKİ FORMÜL)
# ---------------------------------------------------------------------------
def calculate_weighted_cost(details: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Hocanın istediği ağırlıklı toplam formülünü uygular.
    
    Formül:
      TotalCost = (W_delay * Delay) + (W_reliability * ReliabilityCost) + (W_resource * ResourceCost)
      
    [cite_start]Not: ReliabilityCost, PDF'te belirtildiği gibi -log(Reliability) olmalıdır[cite: 52].
    """
    
    # 1. Değerleri çek (Eğer algoritma hesaplamadıysa 0 al)
    total_delay = details.get("total_delay", 0.0)
    res_cost = details.get("resource_cost", 0.0)
    
    # 2. Güvenilirlik Maliyeti Kontrolü
    # Eğer algoritma zaten 'reliability_cost' hesapladıysa onu kullan.
    if "reliability_cost" in details:
        rel_cost = details["reliability_cost"]
    else:
        # Hesaplamadıysa ham güvenilirlikten (raw_reliability) dönüştür
        raw_rel = details.get("raw_reliability", 0.0)
        if raw_rel > 0:
            rel_cost = -math.log(raw_rel) 
        else:
            rel_cost = 1000.0 # Güvenilirlik 0 ise ceza puanı

    # 3. Ağırlıkları çek
    w_d = weights.get("delay", 0.33)
    w_r = weights.get("reliability", 0.33)
    w_c = weights.get("resource", 0.33)
    
    # 4. Sonuç Hesapla
    final_score = (w_d * total_delay) + (w_r * rel_cost) + (w_c * res_cost)
    
    return final_score