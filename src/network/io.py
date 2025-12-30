from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Any

from .graph_model import Graph, Node, Link


# ---------- JSON ----------
def graph_to_dict(graph: Graph) -> Dict[str, Any]:
    """Graph nesnesini JSON'a uygun bir sözlüğe çevirir."""
    nodes = []
    for node in graph.nodes.values():
        nodes.append(
            {
                "id": node.id,
                "processing_delay_ms": node.processing_delay_ms,
                "reliability": node.reliability,
            }
        )

    edges = []
    seen = set()
    for u, nbrs in graph.adj.items():
        for v, link in nbrs.items():
            key = tuple(sorted((u, v)))
            if key in seen:
                continue
            seen.add(key)
            edges.append(
                {
                    "u": link.u,
                    "v": link.v,
                    "delay_ms": link.delay_ms,
                    "bandwidth_mbps": link.bandwidth_mbps,
                    "reliability": link.reliability,
                }
            )

    return {"nodes": nodes, "edges": edges}


def save_graph_json(graph: Graph, filename: str | Path) -> None:
    """Grafı tek bir JSON dosyasına kaydeder."""
    path = Path(filename)
    data = graph_to_dict(graph)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_graph_json(filename: str | Path) -> Graph:
    """JSON dosyasından Graph nesnesi yükler."""
    path = Path(filename)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    g = Graph()

    # Nodes
    for n in data.get("nodes", []):
        node = Node(
            id=int(n["id"]),
            processing_delay_ms=float(n["processing_delay_ms"]),
            reliability=float(n["reliability"]),
        )
        g.add_node(node)

    # Edges (undirected)
    for e in data.get("edges", []):
        link = Link(
            u=int(e["u"]),
            v=int(e["v"]),
            delay_ms=float(e["delay_ms"]),
            bandwidth_mbps=float(e["bandwidth_mbps"]),
            reliability=float(e["reliability"]),
        )
        g.add_undirected_edge(link)

    return g


# ---------- CSV ----------
def save_graph_csv(graph: Graph, folder: str | Path) -> None:
    """Grafı iki CSV dosyasına kaydeder: nodes.csv ve edges.csv.

    - nodes.csv: id,processing_delay_ms,reliability
    - edges.csv: u,v,delay_ms,bandwidth_mbps,reliability
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    nodes_path = folder / "nodes.csv"
    edges_path = folder / "edges.csv"

    with nodes_path.open("w", newline="", encoding="utf-8") as f_nodes:
        writer = csv.writer(f_nodes)
        writer.writerow(["id", "processing_delay_ms", "reliability"])
        for node in graph.nodes.values():
            writer.writerow(
                [node.id, node.processing_delay_ms, node.reliability]
            )

    seen = set()
    with edges_path.open("w", newline="", encoding="utf-8") as f_edges:
        writer = csv.writer(f_edges)
        writer.writerow(["u", "v", "delay_ms", "bandwidth_mbps", "reliability"])
        for u, nbrs in graph.adj.items():
            for v, link in nbrs.items():
                key = tuple(sorted((u, v)))
                if key in seen:
                    continue
                seen.add(key)
                writer.writerow(
                    [
                        link.u,
                        link.v,
                        link.delay_ms,
                        link.bandwidth_mbps,
                        link.reliability,
                    ]
                )


def load_graph_csv(folder: str | Path) -> Graph:
    """nodes.csv ve edges.csv'den Graph nesnesi yükler."""
    folder = Path(folder)
    nodes_path = folder / "nodes.csv"
    edges_path = folder / "edges.csv"

    g = Graph()

    # Nodes
    with nodes_path.open("r", newline="", encoding="utf-8") as f_nodes:
        reader = csv.DictReader(f_nodes)
        for row in reader:
            node = Node(
                id=int(row["id"]),
                processing_delay_ms=float(row["processing_delay_ms"]),
                reliability=float(row["reliability"]),
            )
            g.add_node(node)

    # Edges
    with edges_path.open("r", newline="", encoding="utf-8") as f_edges:
        reader = csv.DictReader(f_edges)
        for row in reader:
            link = Link(
                u=int(row["u"]),
                v=int(row["v"]),
                delay_ms=float(row["delay_ms"]),
                bandwidth_mbps=float(row["bandwidth_mbps"]),
                reliability=float(row["reliability"]),
            )
            g.add_undirected_edge(link)

    return g

# ---------- ÖĞRETMEN CSV FORMATINDAN YÜKLEME ----------
def _parse_float_eu(value: str) -> float:
    """Virgüllü ondalık (örn: '0,85') içeren string'i float'a çevirir."""
    if value is None:
        return 0.0
    value = value.strip()
    if not value:
        return 0.0
    return float(value.replace(",", "."))


def load_graph_teacher_csv(node_csv_path: str | Path, edge_csv_path: str | Path) -> Graph:
    """NodeData.csv + EdgeData.csv dosyalarından Graph nesnesi oluşturur.

    Beklenen kolonlar:
        NodeData.csv : node_id ; s_ms ; r_node
        EdgeData.csv : src ; dst ; capacity_mbps ; delay_ms ; r_link
    """
    node_csv_path = Path(node_csv_path)
    edge_csv_path = Path(edge_csv_path)

    g = Graph()

    # Nodes
    with node_csv_path.open("r", newline="", encoding="utf-8-sig") as f_nodes:
        reader = csv.DictReader(f_nodes, delimiter=";")
        for row in reader:
            node = Node(
                id=int(row["node_id"]),
                processing_delay_ms=_parse_float_eu(row["s_ms"]),
                reliability=_parse_float_eu(row["r_node"]),
            )
            g.add_node(node)

    # Edges (undirected)
    with edge_csv_path.open("r", newline="", encoding="utf-8-sig") as f_edges:
        reader = csv.DictReader(f_edges, delimiter=";")
        for row in reader:
            link = Link(
                u=int(row["src"]),
                v=int(row["dst"]),
                delay_ms=_parse_float_eu(row["delay_ms"]),
                bandwidth_mbps=_parse_float_eu(row["capacity_mbps"]),
                reliability=_parse_float_eu(row["r_link"]),
            )
            g.add_undirected_edge(link)

    return g
