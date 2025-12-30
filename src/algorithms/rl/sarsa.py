# src/algorithms/rl/sarsa.py
from __future__ import annotations

import logging
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Set, Any

# Proje modülleri
from algorithms.base import RoutingAlgorithm, PathResult, calculate_weighted_cost
from network.api import Graph

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SARSAConfig:
    episodes: int = 2000
    alpha: float = 0.14
    gamma: float = 0.95

    # Exponential epsilon decay
    epsilon_start: float = 0.60
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.999

    # Episode limits
    max_steps_factor: float = 2.5
    max_hops_extract: int = 450

    # Reward shaping
    goal_bonus_multiplier: float = 2.0
    revisit_penalty: float = 8.0
    deadend_penalty: float = 80.0

    # Initial Q-values
    initial_q_value: float = -1000.0

    # REMOVED: Experience Replay (not compatible with on-policy SARSA)
    # replay_buffer_size: int = 5000
    # batch_size: int = 32
    # use_replay: bool = True

    # Numerics
    min_prob: float = 1e-6
    one_gbps_mbps: float = 1000.0

    # Debug
    verbose_every: int = 200


class SarsaRouting(RoutingAlgorithm):
    """
    Düzeltilmiş SARSA (On-policy) QoS routing.
    100% PDF uyumlu - Experience Replay kaldırıldı ve kaynak düğüm gecikmesi hariç tutuldu.
    """

    name: str = "SARSA (PDF Compliant)"

    def __init__(self, config: SARSAConfig | None = None, verbose: bool = False) -> None:
        self.config = config or SARSAConfig()
        self.verbose = verbose

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

        if source not in graph.nodes or dest not in graph.nodes:
            raise ValueError(f"SARSA: source/dest graph içinde yok. source={source}, dest={dest}")

        w_delay, w_rel, w_res = self._normalize_weights(weights)
        req_bw = float(bandwidth_requirement or 0.0)

        actions = self._build_actions(graph, req_bw)

        if not self._is_reachable_with_constraint(graph, source, dest, req_bw):
            raise RuntimeError(
                f"SARSA: S({source}) ile D({dest}) arasında bandwidth kısıtı={req_bw} ile yol yok."
            )

        N = len(graph.nodes)
        max_steps = max(10, int(N * self.config.max_steps_factor))

        Q: Dict[Tuple[int, int], float] = {}

        def get_q(s: int, a: int) -> float:
            return Q.get((s, a), self.config.initial_q_value)

        def set_q(s: int, a: int, v: float) -> None:
            Q[(s, a)] = v

        sample_bonus = self._estimate_typical_step_cost(graph, actions, dest, w_delay, w_rel, w_res)
        goal_bonus = max(50.0, self.config.goal_bonus_multiplier * sample_bonus * max(2.0, N / 10.0))

        best_path: List[int] | None = None
        best_total: float = float("inf")

        for ep in range(self.config.episodes):
            epsilon = self._epsilon_exponential(ep)

            state = source
            visited: Set[int] = {state}
            trace: List[int] = [state]

            action = self._choose_action(state, actions, get_q, epsilon)
            if action is None:
                continue

            steps = 0
            is_first_step = True

            while steps < max_steps and state != dest and action is not None:
                next_state = action

                # Step Cost Hesapla - PDF'ye uygun: kaynak düğüm gecikmesi HARİÇ
                step_cost = self._step_weighted_cost_corrected(
                    graph=graph,
                    u=state,
                    v=next_state,
                    dest=dest,
                    is_first_step=is_first_step,
                    w_delay=w_delay,
                    w_rel=w_rel,
                    w_res=w_res,
                )

                reward = -step_cost

                if next_state in visited:
                    reward -= self.config.revisit_penalty

                done = (next_state == dest)
                next_action = None if done else self._choose_action(next_state, actions, get_q, epsilon)

                if done:
                    reward += goal_bonus
                elif next_action is None:
                    reward -= self.config.deadend_penalty

                # Pure SARSA on-policy update - NO experience replay
                current_q = get_q(state, action)
                if done or next_action is None:
                    target_q = reward
                else:
                    next_q = get_q(next_state, next_action)
                    target_q = reward + self.config.gamma * next_q

                new_q = current_q + self.config.alpha * (target_q - current_q)
                set_q(state, action, new_q)

                state = next_state
                action = next_action
                trace.append(state)
                visited.add(state)
                steps += 1
                is_first_step = False

                if done:
                    break

            # En iyi yolu güncelle
            if trace and trace[-1] == dest:
                # PDF uyumlu değerlendirme: kaynak ve hedef düğüm gecikmeleri hariç
                cost, _ = self._evaluate_path_robust(graph, trace, w_delay, w_rel, w_res)
                if cost < best_total:
                    best_total = cost
                    best_path = trace[:]

            if self.verbose and self.config.verbose_every and (ep % self.config.verbose_every == 0):
                logger.info(
                    "SARSA ep=%d/%d eps=%.3f best=%s",
                    ep, self.config.episodes, epsilon,
                    ("∞" if best_path is None else f"{best_total:.4f}")
                )

        # Greedy extraction
        greedy = self._extract_greedy_path(source, dest, actions, get_q)

        candidates = []
        if greedy: candidates.append(greedy)
        if best_path: candidates.append(best_path)

        # Fallback BFS
        if not candidates:
            bfs = self._find_any_path_with_constraint(graph, source, dest, req_bw)
            if bfs: candidates.append(bfs)

        if not candidates:
            raise RuntimeError("SARSA: Yol bulunamadı.")

        # Final Seçim (PDF uyumlu değerlendirme ile)
        final_path, final_total, final_details = self._pick_best_path(
            graph, candidates, w_delay, w_rel, w_res
        )
        
        # Base.py'den gelen fonksiyonla nihai skoru hesapla
        final_score = calculate_weighted_cost(final_details, weights)
        return PathResult(path=final_path, cost=final_score, details=final_details)

    # ----------------------------------------------------------------
    # PDF UYUMLU FONKSİYONLAR
    # ----------------------------------------------------------------
    def _evaluate_path_robust(
        self, graph: Graph, path: List[int], w_d: float, w_r: float, w_res: float
    ) -> Tuple[float, Dict[str, float]]:
        """PDF'ye uygun: kaynak (S) ve hedef (D) düğüm gecikmeleri hariç."""
        if not path or len(path) < 2:
            return float('inf'), {}

        total_delay = 0.0
        reliability_cost = 0.0
        resource_cost = 0.0
        raw_rel = 1.0

        # Link gecikmeleri (tüm linkler)
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            link = graph.get_link(u, v)
            if not link: return float('inf'), {}

            # Güvenli okuma
            d = self._get_safe_attr(link, ["delay_ms", "delay"], 0.0)
            total_delay += d

            r = self._get_safe_attr(link, ["reliability", "rel"], 0.99)
            r = max(r, self.config.min_prob)
            reliability_cost += -math.log(r)
            raw_rel *= r

            bw = self._get_safe_attr(link, ["bandwidth_mbps", "bandwidth", "bw"], 0.1)
            if bw <= 1e-3: bw = 0.1
            resource_cost += (1000.0 / bw)

        # Node Processing Delay - PDF: S ve D hariç ara düğümler
        # path[0] = source, path[-1] = dest, bu yüzden path[1:-1] ara düğümler
        if len(path) > 2:
            for node_id in path[1:-1]:  # S ve D hariç
                node = graph.nodes.get(node_id)
                if node:
                    nd = self._get_safe_attr(node, ["processing_delay_ms", "proc_delay"], 0.0)
                    total_delay += nd
                    
                    nr = self._get_safe_attr(node, ["reliability", "rel"], 0.99)
                    nr = max(nr, self.config.min_prob)
                    reliability_cost += -math.log(nr)
                    raw_rel *= nr

        weighted = (w_d * total_delay) + (w_r * reliability_cost) + (w_res * resource_cost)
        return weighted, {
            "total_delay": total_delay,
            "reliability_cost": reliability_cost,
            "resource_cost": resource_cost,
            "raw_reliability": raw_rel
        }

    def _get_safe_attr(self, obj: Any, names: List[str], default: float) -> float:
        """Nesneden güvenli veri okur (delay mi delay_ms mi diye dert etmez)."""
        for name in names:
            if hasattr(obj, name):
                val = getattr(obj, name)
                if val is not None:
                    try:
                        return float(val)
                    except:
                        pass
        return default

    def _pick_best_path(
            self,
            graph: Graph,
            paths: List[List[int]],
            w_delay: float,
            w_rel: float,
            w_res: float,
    ) -> Tuple[List[int], float, Dict[str, float]]:
        best_p = None
        best_total = float("inf")
        best_details = {}

        for p in paths:
            # PDF uyumlu değerlendirme fonksiyonunu kullanıyoruz
            total, details = self._evaluate_path_robust(graph, p, w_delay, w_rel, w_res)
            if total < best_total:
                best_total = total
                best_p = p
                best_details = details

        if best_p is None:
            raise RuntimeError("SARSA: Yollar değerlendirilemedi.")
        return best_p, best_total, best_details

    def _step_weighted_cost_corrected(
            self, graph: Graph, u: int, v: int, dest: int, is_first_step: bool,
            w_delay: float, w_rel: float, w_res: float,
    ) -> float:
        """PDF uyumlu: kaynak düğüm (S) gecikmesi hariç, sadece ara düğümlerin gecikmesi eklenir."""
        link = graph.get_link(u, v)
        if link is None: return 1e9

        # Link gecikmesi (her zaman eklenir)
        delay = self._get_safe_attr(link, ["delay_ms", "delay"], 0.0)
        
        # PDF: Kaynak düğüm (S) işlem gecikmesi EKLENMEZ
        # if is_first_step:  # REMOVED - PDF'ye göre kaynak düğüm hariç
        #     node_u = graph.nodes.get(u)
        #     if node_u:
        #         delay += self._get_safe_attr(node_u, ["processing_delay_ms", "proc_delay"], 0.0)

        # Hedef olmayan ara düğümlerin işlem gecikmesi eklenir
        if v != dest:
            node_v = graph.nodes.get(v)
            if node_v:
                delay += self._get_safe_attr(node_v, ["processing_delay_ms", "proc_delay"], 0.0)

        # Reliability - PDF uyumlu: kaynak ve hedef düğümler hariç
        rel_cost = 0.0
        link_rel = self._get_safe_attr(link, ["reliability", "rel"], 1.0)
        link_rel = max(link_rel, self.config.min_prob)
        rel_cost += -math.log(link_rel)

        # Kaynak düğüm güvenilirliği EKLENMEZ (PDF'ye göre)
        # if is_first_step:  # REMOVED
        #     node_u = graph.nodes.get(u)
        #     if node_u:
        #         nu_rel = self._get_safe_attr(node_u, ["reliability", "rel"], 1.0)
        #         rel_cost += -math.log(max(nu_rel, self.config.min_prob))

        # Hedef olmayan ara düğümlerin güvenilirliği eklenir
        if v != dest:
            node_v = graph.nodes.get(v)
            if node_v:
                nv_rel = self._get_safe_attr(node_v, ["reliability", "rel"], 1.0)
                rel_cost += -math.log(max(nv_rel, self.config.min_prob))

        # Resource cost (bandwidth)
        bw = self._get_safe_attr(link, ["bandwidth_mbps", "bandwidth", "bw"], 0.1)
        if bw <= 0: bw = 0.1
        res_cost = self.config.one_gbps_mbps / bw

        return (w_delay * delay) + (w_rel * rel_cost) + (w_res * res_cost)

    # ----------------------------------------------------------------
    # HELPER FUNCTIONS
    # ----------------------------------------------------------------
    @staticmethod
    def _normalize_weights(weights):
        d = float(weights.get("delay", 0.33))
        r = float(weights.get("reliability", 0.33))
        res = float(weights.get("resource", 0.33))
        s = d + r + res
        if s <= 0: return (0.33, 0.33, 0.33)
        return (d/s, r/s, res/s)

    def _epsilon_exponential(self, ep):
        if self.config.episodes <= 1: return self.config.epsilon_end
        decay = (self.config.epsilon_end / self.config.epsilon_start) ** (ep / self.config.episodes)
        return self.config.epsilon_start * decay

    @staticmethod
    def _build_actions(graph, req_bw):
        actions = {}
        for u, nbrs in graph.adj.items():
            valid = []
            for v, link in nbrs.items():
                bw = getattr(link, "bandwidth_mbps", getattr(link, "bandwidth", 0.0))
                if float(bw or 0.0) >= req_bw:
                    valid.append(v)
            actions[u] = valid
        return actions

    def _choose_action(self, state, actions, get_q, eps):
        nbrs = actions.get(state, [])
        if not nbrs: return None
        if random.random() < eps: return random.choice(nbrs)
        qs = [get_q(state, a) for a in nbrs]
        max_q = max(qs)
        best = [a for a, q in zip(nbrs, qs) if q == max_q]
        return random.choice(best)

    def _extract_greedy_path(self, s, d, actions, get_q):
        path = [s]; visited = {s}; cur = s
        for _ in range(self.config.max_hops_extract):
            if cur == d: return path
            nbrs = actions.get(cur, [])
            if not nbrs: return None
            qs = [get_q(cur, a) for a in nbrs]
            max_q = max(qs)
            best = [a for a, q in zip(nbrs, qs) if q == max_q]
            nxt = random.choice(best)
            if nxt in visited: return None
            path.append(nxt); visited.add(nxt); cur = nxt
        return None

    def _estimate_typical_step_cost(self, graph, actions, dest, wd, wr, wres):
        samples = []
        nodes = list(actions.keys())
        random.shuffle(nodes)
        for u in nodes[:20]:
            nbrs = actions.get(u, [])
            if not nbrs: continue
            v = random.choice(nbrs)
            cost = self._step_weighted_cost_corrected(graph, u, v, dest, True, wd, wr, wres)
            samples.append(cost)
        if not samples: return 50.0
        samples.sort()
        return samples[len(samples)//2]

    @staticmethod
    def _is_reachable_with_constraint(graph, s, d, req):
        if s == d: return True
        q = deque([s]); seen = {s}
        while q:
            cur = q.popleft()
            for nxt, link in graph.neighbors(cur).items():
                bw = getattr(link, "bandwidth_mbps", getattr(link, "bandwidth", 0.0))
                if float(bw or 0.0) < req: continue
                if nxt == d: return True
                if nxt not in seen: seen.add(nxt); q.append(nxt)
        return False

    @staticmethod
    def _find_any_path_with_constraint(graph, s, d, req):
        if s == d: return [s]
        parents = {s: None}; q = deque([s])
        while q:
            cur = q.popleft()
            for nxt, link in graph.neighbors(cur).items():
                bw = getattr(link, "bandwidth_mbps", getattr(link, "bandwidth", 0.0))
                if float(bw or 0.0) < req or nxt in parents: continue
                parents[nxt] = cur
                if nxt == d:
                    path = []; node = d
                    while node is not None: path.append(node); node = parents[node]
                    return path[::-1]
                q.append(nxt)
        return None