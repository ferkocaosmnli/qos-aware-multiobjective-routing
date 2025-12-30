from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class Node:
    """Ağdaki bir düğümü temsil eder."""

    id: int
    processing_delay_ms: float
    reliability: float  # 0.0–1.0 arası


@dataclass
class Link:
    """İki düğüm arasındaki bağlantıyı temsil eder."""

    u: int
    v: int
    delay_ms: float
    bandwidth_mbps: float
    reliability: float  # 0.0–1.0 arası


class Graph:
    """Basit, yönsüz bir ağı adjacency-list yapısıyla tutar."""

    def __init__(self) -> None:
        # Düğüm ID -> Node nesnesi
        self.nodes: Dict[int, Node] = {}
        # Düğüm ID -> (komşu ID -> Link)
        self.adj: Dict[int, Dict[int, Link]] = {}

    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            raise ValueError(f"Node with id {node.id} already exists")
        self.nodes[node.id] = node
        self.adj.setdefault(node.id, {})

    def add_undirected_edge(self, link: Link) -> None:
        """Yönsüz kenar ekler (u-v ve v-u)."""
        if link.u not in self.nodes or link.v not in self.nodes:
            raise ValueError("Both endpoints must be added as nodes before adding an edge")
        self.adj.setdefault(link.u, {})[link.v] = link
        self.adj.setdefault(link.v, {})[link.u] = link

    def neighbors(self, node_id: int) -> Dict[int, Link]:
        return self.adj.get(node_id, {})

    def get_link(self, u: int, v: int) -> Optional[Link]:
        return self.adj.get(u, {}).get(v)

    def __len__(self) -> int:  # len(graph) -> node sayısı
        return len(self.nodes)
