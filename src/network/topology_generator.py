from __future__ import annotations

import random
from typing import Optional

from .graph_model import Graph, Node, Link

# Varsayılan parametre aralıkları (isterseniz README'de güncelleyebilirsiniz)
NODE_PROCESSING_DELAY_RANGE = (0.5, 2.0)  # ms
NODE_RELIABILITY_RANGE = (0.95, 0.999)

LINK_DELAY_RANGE = (3.0, 15.0)  # ms
LINK_BANDWIDTH_RANGE = (100.0, 1000.0)  # Mbps
LINK_RELIABILITY_RANGE = (0.95, 0.999)


def _random_in_range(lo: float, hi: float) -> float:
    return random.uniform(lo, hi)


def generate_random_graph(
    num_nodes: int = 250,
    p: float = 0.4,
    seed: Optional[int] = None,
) -> Graph:
    """Bağlı olmasına dikkat ederek G(n, p) benzeri rastgele bir ağ üretir.

    Strateji:
    1. Tüm düğümleri oluştur.
    2. Önce rastgele bir "ağaç" oluşturarak grafı kesinlikle bağlı yap.
    3. Ardından her düğüm çifti için ek olarak p olasılıkla kenar ekle.

    Bu topoloji, algoritma ekibinin metrik motoru üzerinde çalışacağı
    temel altyapıdır.
    """
    if seed is not None:
        random.seed(seed)

    g = Graph()

    # 1) Düğümler
    for i in range(num_nodes):
        node = Node(
            id=i,
            processing_delay_ms=_random_in_range(*NODE_PROCESSING_DELAY_RANGE),
            reliability=_random_in_range(*NODE_RELIABILITY_RANGE),
        )
        g.add_node(node)

    # 2) Bağlılığı garanti eden basit rastgele ağaç:
    for i in range(1, num_nodes):
        # Her düğümü kendinden önceki rastgele bir düğüme bağla
        j = random.randint(0, i - 1)
        link = _create_random_link(i, j)
        g.add_undirected_edge(link)

    # 3) Ek rastgele kenarlar (G(n,p) benzeri)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if g.get_link(i, j) is not None:
                continue  # Zaten kenar varsa geç
            if random.random() < p:
                link = _create_random_link(i, j)
                g.add_undirected_edge(link)

    return g


def _create_random_link(u: int, v: int) -> Link:
    return Link(
        u=u,
        v=v,
        delay_ms=_random_in_range(*LINK_DELAY_RANGE),
        bandwidth_mbps=_random_in_range(*LINK_BANDWIDTH_RANGE),
        reliability=_random_in_range(*LINK_RELIABILITY_RANGE),
    )
