# src/algorithms/metaheuristics/ga.py
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

# calculate_weighted_cost fonksiyonunu import ettik
from algorithms.base import RoutingAlgorithm, PathResult, calculate_weighted_cost
from network.api import Graph

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class GAConfig:
    # GUI'de donmayı engellemek için sayıları optimize ettim
    population_size: int = 50   # 100 -> 50 (Hız için yeterli)
    generations: int = 50       # 200 -> 50 (Arayüz bekletmemek için)
    tournament_k: int = 5
    elitism: int = 2
    crossover_rate: float = 0.85
    mutation_rate: float = 0.25
    
    # Init ve Onarım Ayarları
    max_init_attempts: int = 1000 # 3000 çok fazla, azalttım
    max_path_len: int = 250
    
    # Güvenlik
    min_prob: float = 1e-6
    # Maksimum işlem süresi (saniye) - Arayüzün çökmemesi için
    max_execution_time: float = 15.0 

class GeneticAlgorithm(RoutingAlgorithm):
    """
    Optimize Edilmiş Genetik Algoritma.
    
    Düzeltmeler:
    1. Sonsuz döngü koruması (Safety Counter) eklendi.
    2. Zaman aşımı (Timeout) kontrolü eklendi.
    3. PDF Madde 3.1'e tam uyumlu Düğüm Gecikmesi hesabı.
    """

    name: str = "GA (Optimized)"

    def __init__(self, config: GAConfig | None = None, verbose: bool = False) -> None:
        self.config = config or GAConfig()
        self.verbose = verbose
        self.fitness_cache: Dict[Tuple[int, ...], Tuple[float, Dict[str, Any]]] = {}

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
        
        start_time = time.time()
        self.fitness_cache.clear()

        req_bw = float(bandwidth_requirement or 0.0)
        w_delay, w_rel, w_res = self._normalize_weights(weights)

        # 1. Adım: Geçerli Komşulukları Belirle
        actions = self._build_valid_actions(graph, req_bw)

        # 2. Adım: Başlangıç Popülasyonunu Oluştur
        population = self._init_population_iterative(graph, source, dest, actions)

        if not population:
            logger.warning("GA: Başlangıç popülasyonu bulunamadı, Fallback BFS devreye giriyor.")
            fallback = self._bfs_path_iterative(graph, source, dest, actions)
            if fallback:
                population = [fallback] * self.config.population_size
            else:
                # Boş sonuç dön, hata fırlatma ki arayüz çökmesin
                # Cost=0.0 ve boş detay ile dönüyoruz
                return PathResult(path=[], cost=0.0, details={"error": "Yol bulunamadı."})

        best_path = population[0]
        best_cost = float('inf')

        # --- EVRİM DÖNGÜSÜ ---
        for gen in range(self.config.generations):
            # Süre kontrolü (GUI donmasın diye)
            if time.time() - start_time > self.config.max_execution_time:
                logger.warning("GA: Zaman aşımı (Timeout). Mevcut en iyi sonuç döndürülüyor.")
                break

            # Fitness Hesapla (Memoization)
            fitness_scores = []
            for p in population:
                cost, _ = self._calculate_fitness_with_cache(graph, p, w_delay, w_rel, w_res)
                fitness_scores.append(cost)
            
            # En iyiyi güncelle
            gen_best_idx = min(range(len(fitness_scores)), key=lambda i: fitness_scores[i])
            if fitness_scores[gen_best_idx] < best_cost:
                best_cost = fitness_scores[gen_best_idx]
                best_path = population[gen_best_idx]

            # Elitizm
            sorted_indices = sorted(range(len(population)), key=lambda i: fitness_scores[i])
            new_population = [population[i] for i in sorted_indices[:self.config.elitism]]

            # Yeni bireyler üret
            safety_counter = 0
            max_safety = self.config.population_size * 5 

            while len(new_population) < self.config.population_size:
                safety_counter += 1
                if safety_counter > max_safety:
                    remaining = self.config.population_size - len(new_population)
                    for _ in range(remaining):
                        new_population.append(best_path[:])
                    break

                p1 = self._tournament_select(population, fitness_scores)
                p2 = self._tournament_select(population, fitness_scores)
                
                c1, c2 = p1[:], p2[:]

                # Çaprazlama (Crossover)
                if random.random() < self.config.crossover_rate:
                    c1, c2 = self._crossover(p1, p2)
                
                # Mutasyon
                if random.random() < self.config.mutation_rate:
                    c1 = self._mutate(c1, actions)
                if random.random() < self.config.mutation_rate:
                    c2 = self._mutate(c2, actions)

                c1 = self._trim_cycles(c1)
                c2 = self._trim_cycles(c2)

                if self._is_valid_path(c1, source, dest):
                    new_population.append(c1)
                if len(new_population) < self.config.population_size and self._is_valid_path(c2, source, dest):
                    new_population.append(c2)
            
            population = new_population

        # Final Sonuç
        # Burada GA'nın kendi iç fitness hesabını alıyoruz
        internal_cost, details = self._calculate_fitness_with_cache(graph, best_path, w_delay, w_rel, w_res)
        
        # --- KRİTİK DÜZELTME ---
        # GUI'de doğru görünmesi için base.py'deki standart fonksiyonla nihai skoru hesaplıyoruz.
        final_score = calculate_weighted_cost(details, weights)

        if self.verbose:
            logger.info(f"GA Finished. Cost: {final_score:.4f}")

        # total_cost yerine 'cost' kullanıyoruz
        return PathResult(path=best_path, cost=final_score, details=details)

    # ----------------------------------------------------------------
    # 1. INITIALIZATION (ITERATIVE)
    # ----------------------------------------------------------------
    def _init_population_iterative(
        self, 
        graph: Graph, 
        source: int, 
        dest: int, 
        actions: Dict[int, List[int]]
    ) -> List[List[int]]:
        population = []
        seen_hashes = set()
        
        attempts = 0
        while len(population) < self.config.population_size and attempts < self.config.max_init_attempts:
            attempts += 1
            path = self._generate_random_path_iterative(graph, source, dest, actions)
            if path:
                h = tuple(path)
                if h not in seen_hashes:
                    seen_hashes.add(h)
                    population.append(path)
        return population

    def _generate_random_path_iterative(
        self, 
        graph: Graph, 
        source: int, 
        dest: int, 
        actions: Dict[int, List[int]]
    ) -> Optional[List[int]]:
        """Rekürsif olmayan, Stack tabanlı Rastgele DFS."""
        stack = [(source, [source])]
        iter_count = 0
        max_iters = 2000 

        while stack:
            iter_count += 1
            if iter_count > max_iters:
                return None
                
            curr, path = stack.pop()
            
            if curr == dest:
                return path
            
            if len(path) >= self.config.max_path_len:
                continue

            neighbors = actions.get(curr, [])[:]
            if not neighbors:
                continue

            random.shuffle(neighbors)
            
            for n in neighbors:
                if n not in path:
                    stack.append((n, path + [n]))
                    break # Sadece bir tane ekle, derinlemesine git
            
            if len(stack) > 50:
                stack = stack[-50:]

        return None

    # ----------------------------------------------------------------
    # 2. FITNESS & MEMOIZATION
    # ----------------------------------------------------------------
    def _calculate_fitness_with_cache(
        self, graph: Graph, path: List[int], w_d: float, w_r: float, w_res: float
    ) -> Tuple[float, Dict[str, Any]]:
        path_tuple = tuple(path)
        if path_tuple in self.fitness_cache:
            return self.fitness_cache[path_tuple]

        cost, details = self._evaluate_path_strict(graph, path, w_d, w_r, w_res)
        self.fitness_cache[path_tuple] = (cost, details)
        return cost, details

    def _evaluate_path_strict(
        self, graph: Graph, path: List[int], w_d: float, w_r: float, w_res: float
    ) -> Tuple[float, Dict[str, Any]]:
        """PDF Madde 3.1, 3.2, 3.3 Uyumlu Hesaplama."""
        if not path or len(path) < 2:
            return float('inf'), {}

        total_delay = 0.0
        reliability_cost = 0.0
        resource_cost = 0.0
        raw_rel = 1.0

        # A. Kenar (Link) Döngüsü
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            link = graph.get_link(u, v)
            if not link: 
                return float('inf'), {}

            # 1. Link Delay
            d = self._get_safe_attr(link, ["delay_ms", "delay"], 0.0)
            total_delay += d

            # 2. Link Reliability
            r = self._get_safe_attr(link, ["reliability", "rel"], 0.99)
            r = max(r, self.config.min_prob)
            reliability_cost += -math.log(r)
            raw_rel *= r

            # 3. Resource Cost (1 Gbps / BW)
            bw = self._get_safe_attr(link, ["bandwidth_mbps", "bandwidth", "bw"], 0.1)
            if bw <= 1e-3: 
                bw = 0.1
            resource_cost += (1000.0 / bw)

        # B. Düğüm (Node) Döngüsü - SADECE ARA DÜĞÜMLER 
        if len(path) > 2:
            for node_id in path[1:-1]:
                node = graph.nodes.get(node_id)
                if node:
                    # Node Delay
                    nd = self._get_safe_attr(node, ["processing_delay_ms", "proc_delay"], 0.0)
                    total_delay += nd
                    
                    # Node Reliability
                    nr = self._get_safe_attr(node, ["reliability", "rel"], 0.99)
                    nr = max(nr, self.config.min_prob)
                    reliability_cost += -math.log(nr)
                    raw_rel *= nr

        weighted_cost = (w_d * total_delay) + (w_r * reliability_cost) + (w_res * resource_cost)
        
        return weighted_cost, {
            "total_delay": total_delay,
            "reliability_cost": reliability_cost,
            "resource_cost": resource_cost,
            "raw_reliability": raw_rel
        }

    # ----------------------------------------------------------------
    # 3. GENETİK OPERATÖRLER
    # ----------------------------------------------------------------
    def _crossover(self, p1: List[int], p2: List[int]) -> Tuple[List[int], List[int]]:
        common_nodes = set(p1[1:-1]) & set(p2[1:-1])
        
        if common_nodes:
            cut_node = random.choice(list(common_nodes))
            idx1 = p1.index(cut_node)
            idx2 = p2.index(cut_node)
            
            child1 = p1[:idx1] + p2[idx2:]
            child2 = p2[:idx2] + p1[idx1:]
            return child1, child2
        else:
            return p1[:], p2[:]

    def _mutate(self, path: List[int], actions: Dict[int, List[int]]) -> List[int]:
        if len(path) < 3:
            return path
        
        mutated = path[:]
        mutation_type = random.choice(["swap", "remove_add"]) 
        
        if mutation_type == "swap":
            i = random.randint(1, len(mutated)-2)
            j = random.randint(1, len(mutated)-2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
            
        elif mutation_type == "remove_add":
            idx = random.randint(1, len(mutated)-2)
            node = mutated[idx]
            neighbors = actions.get(node, [])
            if neighbors:
                new_node = random.choice(neighbors)
                if new_node not in mutated:
                    mutated[idx] = new_node
        
        return mutated

    # ----------------------------------------------------------------
    # 4. YARDIMCI FONKSİYONLAR
    # ----------------------------------------------------------------
    def _tournament_select(self, pop: List[List[int]], scores: List[float]) -> List[int]:
        k = min(self.config.tournament_k, len(pop))
        indices = random.sample(range(len(pop)), k)
        best_idx = min(indices, key=lambda i: scores[i])
        return pop[best_idx]

    def _trim_cycles(self, path: List[int]) -> List[int]:
        if not path: 
            return []
        
        seen = {}
        new_path = []
        
        for n in path:
            if n in seen:
                new_path = new_path[:seen[n]+1]
                seen = {node: i for i, node in enumerate(new_path)}
            else:
                seen[n] = len(new_path)
                new_path.append(n)
        
        return new_path

    def _is_valid_path(self, path: List[int], s: int, d: int) -> bool:
        return bool(path and path[0] == s and path[-1] == d)

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

    def _bfs_path_iterative(self, graph: Graph, s: int, d: int, actions: Dict[int, List[int]]) -> Optional[List[int]]:
        queue = [(s, [s])]
        visited = {s}
        
        while queue:
            curr, path = queue.pop(0)
            if curr == d: 
                return path
            
            for n in actions.get(curr, []):
                if n not in visited:
                    visited.add(n)
                    queue.append((n, path + [n]))
        
        return None

    def _get_safe_attr(self, obj: Any, names: List[str], default: float) -> float:
        for name in names:
            if hasattr(obj, name):
                val = getattr(obj, name)
                if val is not None:
                    try:
                        return float(val)
                    except:
                        pass
        return default

    @staticmethod
    def _normalize_weights(weights: Dict[str, float]) -> Tuple[float, float, float]:
        d = float(weights.get("delay", 1.0))
        r = float(weights.get("reliability", 1.0))
        res = float(weights.get("resource", 1.0))
        total = d + r + res
        if total <= 1e-9: 
            return (0.33, 0.33, 0.33)
        return (d/total, r/total, res/total)