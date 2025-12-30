from __future__ import annotations

from typing import List, Tuple, Dict, Optional

from .graph_model import Graph
from .topology_generator import generate_random_graph
from .metrics import total_delay_ms, reliability_cost, resource_cost, total_cost
from .validators import is_valid_path, is_reachable, find_any_path
from .io import (
    save_graph_json,
    load_graph_json,
    save_graph_csv,
    load_graph_csv,
    load_graph_teacher_csv,  # <--- YENİ EKLENDİ
)


__all__ = [
    "Graph",
    "generate_random_graph",
    "total_delay_ms",
    "reliability_cost",
    "resource_cost",
    "total_cost",
    "is_valid_path",
    "is_reachable",
    "find_any_path",
    "save_graph_json",
    "load_graph_json",
    "save_graph_csv",
    "load_graph_csv",
    "load_graph_teacher_csv", # <--- YENİ EKLENDİ
]


def evaluate_path_with_weights(
    graph: Graph,
    path: List[int],
    w_delay: float,
    w_reliability: float,
    w_resource: float,
) -> Tuple[float, Dict[str, float]]:
    """Algoritma ekiplerinin direkt çağırabileceği top-level fonksiyon.

    Dönüş:
        (total_cost, detay_dict)
    """
    return total_cost(graph, path, w_delay, w_reliability, w_resource)