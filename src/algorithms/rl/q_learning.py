# src/algorithms/rl/q_learning.py
from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Set, Any

# Proje modülleri
from algorithms.base import RoutingAlgorithm, PathResult, calculate_weighted_cost
from network.api import Graph

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 2000
    alpha: float = 0.1  # Learning rate
    gamma: float = 0.9  # Discount factor
    
    # Epsilon (Exploration)
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.998

    # Limitler
    max_steps_factor: float = 3.0
    
    # Ödül/Ceza
    goal_reward: float = 100.0
    step_penalty_factor: float = 1.0 
    deadend_penalty: float = -50.0
    loop_penalty: float = -10.0
    
    initial_q: float = 0.0

class QLearningRouting(RoutingAlgorithm):
    """
    Q-Learning (Off-Policy) Algoritması.
    Dosya Adı: q_learning.py
    """

    name: str = "Q-Learning"

    def __init__(self, config: QLearningConfig | None = None) -> None:
        self.config = config or QLearningConfig()
        self.q_table: Dict[Tuple[int, int], float] = {}

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

        # 1. Hazırlık
        w_d, w_r, w_res = self._normalize_weights(weights)
        req_bw = float(bandwidth_requirement or 0.0)
        
        # Geçerli komşular (BW kısıtına göre)
        actions = self._build_valid_actions(graph, req_bw)
        
        # Başlangıçta yol var mı kontrol et (Hız için)
        if not self._is_reachable_bfs(actions, source, dest):
             return PathResult(path=[], cost=0.0, details={"error": "Yol yok (BW kısıtı)."})

        self.q_table.clear()
        
        # 2. Eğitim Döngüsü
        max_steps = int(len(graph.nodes) * self.config.max_steps_factor)

        for ep in range(self.config.episodes):
            state = source
            epsilon = self._get_epsilon(ep)
            path_trace = [state]
            visited = {state}
            
            for _ in range(max_steps):
                if state == dest:
                    break
                
                # A) Aksiyon Seç
                available_actions = actions.get(state, [])
                if not available_actions:
                    break # Çıkmaz sokak
                
                if random.random() < epsilon:
                    action = random.choice(available_actions)
                else:
                    action = self._get_best_action(state, available_actions)

                # B) Adım At ve Ödül Hesapla
                step_cost = self._calculate_step_cost(graph, state, action, w_d, w_r, w_res)
                reward = -(step_cost * self.config.step_penalty_factor)
                
                next_state = action
                done = (next_state == dest)
                
                if done:
                    reward += self.config.goal_reward
                elif next_state in visited:
                    reward += self.config.loop_penalty
                
                # C) Q-Learning Update
                # Q(s,a) = Q(s,a) + alpha * [R + gamma * max(Q(s', a')) - Q(s,a)]
                current_q = self._get_q(state, action)
                
                max_next_q = 0.0
                if not done:
                    next_possible = actions.get(next_state, [])
                    if next_possible:
                        max_next_q = max([self._get_q(next_state, a) for a in next_possible])
                
                target = reward + (self.config.gamma * max_next_q)
                new_q = current_q + self.config.alpha * (target - current_q)
                self._set_q(state, action, new_q)
                
                # İlerle
                visited.add(next_state)
                path_trace.append(next_state)
                state = next_state
                
                if done:
                    break

        # 3. En İyi Yolu Çıkar (Greedy)
        best_path = self._extract_policy_path(source, dest, actions)
        
        if not best_path:
            best_path = self._bfs_path(actions, source, dest) # Fallback

        if not best_path:
             return PathResult(path=[], cost=0.0, details={"error": "Rota bulunamadı."})

        # 4. Sonuç Hesapla
        final_cost_raw, details = self._evaluate_path_robust(graph, best_path, w_d, w_r, w_res)
        final_score = calculate_weighted_cost(details, weights)

        return PathResult(
            path=best_path,
            cost=final_score,
            details=details
        )

    # --- YARDIMCI METOTLAR ---
    def _get_q(self, s: int, a: int) -> float:
        return self.q_table.get((s, a), self.config.initial_q)

    def _set_q(self, s: int, a: int, val: float):
        self.q_table[(s, a)] = val

    def _get_best_action(self, state: int, actions: List[int]) -> int:
        if not actions: return -1
        q_vals = [self._get_q(state, a) for a in actions]
        max_q = max(q_vals)
        # Eşitlik durumunda rastgele seç (çeşitlilik için)
        candidates = [a for a, q in zip(actions, q_vals) if q == max_q]
        return random.choice(candidates)

    def _get_epsilon(self, episode: int) -> float:
        d = (self.config.epsilon_end / self.config.epsilon_start) ** (episode / self.config.episodes)
        return self.config.epsilon_start * d

    def _extract_policy_path(self, s: int, d: int, actions: Dict[int, List[int]]) -> Optional[List[int]]:
        path = [s]; curr = s; visited = {s}; limit = 500
        while curr != d and limit > 0:
            limit -= 1
            available = actions.get(curr, [])
            if not available: return None
            nxt = self._get_best_action(curr, available)
            if nxt in visited: return None
            visited.add(nxt); path.append(nxt); curr = nxt
        return path if curr == d else None

    def _calculate_step_cost(self, graph, u, v, w_d, w_r, w_res) -> float:
        link = graph.get_link(u, v)
        if not link: return 1000.0
        
        delay = self._get_safe_attr(link, ["delay_ms", "delay"], 0.0)
        bw = self._get_safe_attr(link, ["bandwidth_mbps", "bandwidth", "bw"], 0.1)
        if bw <= 0: bw = 0.1
        res_cost = 1000.0 / bw
        
        rel = self._get_safe_attr(link, ["reliability", "rel"], 0.99)
        rel_cost = -math.log(max(rel, 1e-6))
        
        return (w_d * delay) + (w_r * rel_cost) + (w_res * res_cost)

    def _build_valid_actions(self, graph, req_bw):
        actions = {}
        for u, nbrs in graph.adj.items():
            valid = []
            for v, link in nbrs.items():
                bw = self._get_safe_attr(link, ["bandwidth_mbps", "bandwidth", "bw"], 0.0)
                if bw >= req_bw: valid.append(v)
            actions[u] = valid
        return actions

    def _get_safe_attr(self, obj, names, default):
        for name in names:
            if hasattr(obj, name):
                return float(getattr(obj, name) or default)
        return default

    def _normalize_weights(self, w):
        d = w.get("delay", 1.0); r = w.get("reliability", 1.0); res = w.get("resource", 1.0)
        t = d + r + res
        return (d/t, r/t, res/t) if t > 0 else (0.33, 0.33, 0.33)

    def _is_reachable_bfs(self, actions, s, d):
        q = [s]; vis = {s}
        while q:
            curr = q.pop(0)
            if curr == d: return True
            for n in actions.get(curr, []):
                if n not in vis: vis.add(n); q.append(n)
        return False

    def _bfs_path(self, actions, s, d):
        q = [(s, [s])]; vis = {s}
        while q:
            curr, p = q.pop(0)
            if curr == d: return p
            for n in actions.get(curr, []):
                if n not in vis: vis.add(n); q.append((n, p+[n]))
        return None

    def _evaluate_path_robust(self, graph, path, w_d, w_r, w_res):
        if not path: return 0.0, {}
        td, rc, resc, raw_r = 0.0, 0.0, 0.0, 1.0
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            link = graph.get_link(u, v)
            if link:
                td += self._get_safe_attr(link, ["delay_ms", "delay"], 0)
                bw = self._get_safe_attr(link, ["bandwidth_mbps", "bw"], 0.1)
                resc += 1000.0/max(bw, 0.1)
                r = self._get_safe_attr(link, ["reliability", "rel"], 0.99)
                rc += -math.log(max(r, 1e-6))
                raw_r *= r
        if len(path) > 2:
            for n in path[1:-1]:
                node = graph.nodes.get(n)
                if node:
                    td += self._get_safe_attr(node, ["processing_delay_ms", "proc_delay"], 0)
                    r = self._get_safe_attr(node, ["reliability", "rel"], 0.99)
                    rc += -math.log(max(r, 1e-6))
                    raw_r *= r
        cost = (w_d * td) + (w_r * rc) + (w_res * resc)
        return cost, {"total_delay": td, "reliability_cost": rc, "resource_cost": resc, "raw_reliability": raw_r}