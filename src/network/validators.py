from __future__ import annotations

from collections import deque
from typing import Iterable, List, Optional

from .graph_model import Graph


def is_valid_path(graph: Graph, path: Iterable[int]) -> bool:
    """Yolun graf üzerinde geçerli olup olmadığını kontrol eder.

    - Tüm düğümler graf içinde olmalı
    - Ardışık her çift arasında kenar olmalı
    """
    path = list(path)
    if len(path) < 2:
        return False

    # Düğümler mevcut mu?
    for node_id in path:
        if node_id not in graph.nodes:
            return False

    # Kenarlar mevcut mu?
    for u, v in zip(path[:-1], path[1:]):
        if graph.get_link(u, v) is None:
            return False

    return True


def is_reachable(graph: Graph, source: int, dest: int) -> bool:
    """BFS ile S'den D'ye en az bir yol var mı kontrol eder."""
    return find_any_path(graph, source, dest) is not None


def find_any_path(graph: Graph, source: int, dest: int) -> Optional[List[int]]:
    """Basit BFS ile S'den D'ye herhangi bir yol bulur (en az hop'lu)."""
    if source not in graph.nodes or dest not in graph.nodes:
        return None

    queue = deque([source])
    parents = {source: None}

    while queue:
        current = queue.popleft()
        if current == dest:
            break
        for neighbor in graph.neighbors(current).keys():
            if neighbor not in parents:
                parents[neighbor] = current
                queue.append(neighbor)

    if dest not in parents:
        return None

    # Yol rekonstrüksiyonu
    path: List[int] = []
    cur = dest
    while cur is not None:
        path.append(cur)
        cur = parents[cur]
    path.reverse()
    return path
