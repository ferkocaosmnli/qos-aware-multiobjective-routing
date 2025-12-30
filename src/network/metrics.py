from __future__ import annotations

import math
from typing import List, Tuple, Dict

from .graph_model import Graph


def total_delay_ms(graph: Graph, path: List[int]) -> float:
    """Verilen yol için toplam gecikmeyi (ms cinsinden) hesaplar.

    Tanım (öğretim üyesinin notuna göre):
    - Tüm link gecikmeleri
    - S ve D *hariç* aradaki düğümlerin işlem gecikmeleri
    """
    if len(path) < 2:
        raise ValueError("Path must contain at least two nodes (S and D)")

    delay = 0.0

    # Link gecikmeleri
    for u, v in zip(path[:-1], path[1:]):
        link = graph.get_link(u, v)
        if link is None:
            raise ValueError(f"No link between {u} and {v}")
        delay += link.delay_ms

    # Ara düğümlerin işlem gecikmeleri (S ve D hariç)
    for node_id in path[1:-1]:
        node = graph.nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in graph")
        delay += node.processing_delay_ms

    return delay


def reliability_cost(graph: Graph, path: List[int]) -> float:
    """Güvenilirlik için minimize edilen cost'i hesaplar.

    Gerçek güvenilirlik çarpımsal: P_success = Π (node_reliability * link_reliability)
    Bunu log-domain'de toplamsal cost'e çeviriyoruz:
        cost = -log(P_success) = Σ(-log(p_i))

    Cost ne kadar küçükse o kadar güvenilir yol demektir.
    """
    if len(path) < 2:
        raise ValueError("Path must contain at least two nodes (S and D)")

    cost = 0.0

    # Linkler
    for u, v in zip(path[:-1], path[1:]):
        link = graph.get_link(u, v)
        if link is None:
            raise ValueError(f"No link between {u} and {v}")
        cost += -math.log(link.reliability)

    # Ara düğümler (S ve D hariç)
    for node_id in path:
        node = graph.nodes.get(node_id)
        if node is None:
            raise ValueError(f"Node {node_id} not found in graph")
        cost += -math.log(node.reliability)

    return cost


def resource_cost(graph: Graph, path: List[int]) -> float:
    """Kaynak kullanımı cost'i.

    Tanım:
        her link için (1 Gbps / bandwidth_ij) toplanır.
    Bant genişliği düşük olan link daha "pahalı" olur.
    """
    if len(path) < 2:
        raise ValueError("Path must contain at least two nodes (S and D)")

    cost = 0.0
    ONE_GBPS = 1000.0  # Mbps

    for u, v in zip(path[:-1], path[1:]):
        link = graph.get_link(u, v)
        if link is None:
            raise ValueError(f"No link between {u} and {v}")
        cost += ONE_GBPS / link.bandwidth_mbps

    return cost


def total_cost(
    graph: Graph,
    path: List[int],
    w_delay: float,
    w_reliability: float,
    w_resource: float,
) -> Tuple[float, Dict[str, float]]:
    """Üç metriği ağırlıklandırıp tek bir TotalCost değerine çevirir.

    w_delay + w_reliability + w_resource = 1 olmalıdır.
    """
    total_w = w_delay + w_reliability + w_resource
    if not math.isclose(total_w, 1.0, rel_tol=1e-6):
        raise ValueError("Weights must sum to 1")

    d = total_delay_ms(graph, path)
    r = reliability_cost(graph, path)
    c = resource_cost(graph, path)

    total = w_delay * d + w_reliability * r + w_resource * c

    details = {
        "delay_ms": d,
        "reliability_cost": r,
        "resource_cost": c,
        "w_delay": w_delay,
        "w_reliability": w_reliability,
        "w_resource": w_resource,
    }
    return total, details
