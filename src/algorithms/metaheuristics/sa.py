# src/algorithms/metaheuristics/sa.py
from __future__ import annotations

import logging
import math
import random
from typing import Dict, Optional, List, Tuple, Any

# Proje modülleri
from algorithms.base import RoutingAlgorithm, PathResult, calculate_weighted_cost
from network.api import Graph

logger = logging.getLogger(__name__)

class SimulatedAnnealing(RoutingAlgorithm):
    """
    Benzetimli Tavlama (Simulated Annealing - SA) Algoritması.
    """

    name: str = "Simulated Annealing"

    def run(
        self,
        graph: Graph,
        source: int,
        dest: int,
        weights: Dict[str, float],
        bandwidth_requirement: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> PathResult:
        
        if seed is not None:
            random.seed(seed)

        # 1. Ağırlıkları Normalize Et
        w_d = float(weights.get("delay", 1.0))
        w_r = float(weights.get("reliability", 1.0))
        w_res = float(weights.get("resource", 1.0))
        total_w = w_d + w_r + w_res
        if total_w > 0:
            w_d /= total_w
            w_r /= total_w
            w_res /= total_w

        req_bw = float(bandwidth_requirement or 0.0)

        # 2. Geçerli Komşulukları Belirle
        actions = self._build_valid_actions(graph, req_bw)

        # 3. SA Parametreleri
        TEMPERATURE = 1000.0
        COOLING_RATE = 0.95
        MIN_TEMPERATURE = 0.01
        ITERATIONS = 20

        # 4. Başlangıç Çözümü
        current_path = self._find_initial_solution(graph, source, dest, actions)
        
        if not current_path:
            return PathResult(path=[], cost=0.0, details={"error": "Yol bulunamadı."})

        # İlk maliyeti hesapla
        current_cost, _ = self._evaluate_path_robust(graph, current_path, w_d, w_r, w_res)

        best_path = current_path[:]
        best_cost = current_cost
        best_details = {}

        # 5. Tavlama Döngüsü
        while TEMPERATURE > MIN_TEMPERATURE:
            for _ in range(ITERATIONS):
                neighbor_path = self._get_neighbor(current_path, actions, dest)
                
                if not neighbor_path:
                    continue

                neighbor_cost, _ = self._evaluate_path_robust(graph, neighbor_path, w_d, w_r, w_res)
                delta = neighbor_cost - current_cost

                accept = False
                if delta < 0:
                    accept = True
                else:
                    p = math.exp(-delta / TEMPERATURE)
                    if random.random() < p:
                        accept = True

                if accept:
                    current_path = neighbor_path
                    current_cost = neighbor_cost

                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_path = current_path[:]
            
            TEMPERATURE *= COOLING_RATE

        # 6. Sonuç Paketleme
        final_cost_raw, details = self._evaluate_path_robust(graph, best_path, w_d, w_r, w_res)
        final_score = calculate_weighted_cost(details, weights)

        return PathResult(
            path=best_path,
            cost=final_score,
            details=details
        )

    # --- YARDIMCI FONKSİYONLAR ---
    def _find_initial_solution(self, graph, s, d, actions) -> Optional[List[int]]:
        queue = [(s, [s])]
        visited = {s}
        while queue:
            curr, path = queue.pop(0)
            if curr == d:
                return path
            for nxt in actions.get(curr, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [nxt]))
        return None

    def _get_neighbor(self, path: List[int], actions: Dict[int, List[int]], dest: int) -> Optional[List[int]]:
        if len(path) < 2: return None
        cut_idx = random.randint(0, len(path) - 2)
        prefix = path[:cut_idx+1]
        curr = prefix[-1]
        visited = set(prefix)
        new_suffix = []
        
        for _ in range(50):
            if curr == dest: return prefix + new_suffix
            candidates = [n for n in actions.get(curr, []) if n not in visited]
            if not candidates: return None
            nxt = random.choice(candidates)
            new_suffix.append(nxt)
            visited.add(nxt)
            curr = nxt
        return None

    def _build_valid_actions(self, graph: Graph, req_bw: float) -> Dict[int, List[int]]:
        actions = {}
        for u, nbrs in graph.adj.items():
            valid = []
            for v, link in nbrs.items():
                bw = self._get_safe_attr(link, ["bandwidth_mbps", "bandwidth", "bw"], 0.0)
                if bw >= req_bw:
                    valid.append(v)
            actions[u] = valid
        return actions

    def _evaluate_path_robust(self, graph, path, w_d, w_r, w_res) -> Tuple[float, Dict[str, float]]:
        if not path: return float('inf'), {}
        total_delay = 0.0
        reliability_cost = 0.0
        resource_cost = 0.0
        raw_rel = 1.0

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            link = graph.get_link(u, v)
            if not link: return float('inf'), {}

            total_delay += self._get_safe_attr(link, ["delay_ms", "delay"], 0.0)
            r = self._get_safe_attr(link, ["reliability", "rel"], 0.99)
            r = max(r, 1e-6)
            reliability_cost += -math.log(r)
            raw_rel *= r
            
            bw = self._get_safe_attr(link, ["bandwidth_mbps", "bandwidth", "bw"], 0.1)
            if bw <= 0.001: bw = 0.1
            resource_cost += (1000.0 / bw)

        if len(path) > 2:
            for node_id in path[1:-1]:
                node = graph.nodes.get(node_id)
                if node:
                    total_delay += self._get_safe_attr(node, ["processing_delay_ms", "proc_delay"], 0.0)
                    nr = self._get_safe_attr(node, ["reliability", "rel"], 0.99)
                    nr = max(nr, 1e-6)
                    reliability_cost += -math.log(nr)
                    raw_rel *= nr

        cost = (w_d * total_delay) + (w_r * reliability_cost) + (w_res * resource_cost)
        return cost, {
            "total_delay": total_delay,
            "reliability_cost": reliability_cost,
            "resource_cost": resource_cost,
            "raw_reliability": raw_rel
        }

    def _get_safe_attr(self, obj: Any, names: List[str], default: float) -> float:
        for name in names:
            if hasattr(obj, name):
                val = getattr(obj, name)
                if val is not None:
                    try: return float(val)
                    except: pass
        return default