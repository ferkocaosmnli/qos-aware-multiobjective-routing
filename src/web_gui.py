# src/web_gui.py
from __future__ import annotations

import time
import base64
import os
import math
import random
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import streamlit as st
import networkx as nx
import plotly.graph_objects as go

# Proje modülleri
try:
    from network.io import load_graph_teacher_csv
    from experiments.scenarios import load_scenarios_from_csv, Scenario
    from experiments.run_experiments import default_algorithms
    from algorithms.base import RoutingAlgorithm, PathResult
except ImportError as e:
    st.error(f"Modül import hatası: {e}. Lütfen terminalde 'src' klasörü içinde olduğunuzdan emin olun.")
    st.stop()

# -----------------------------------------------------------------------------
# 1. YARDIMCI FONKSİYONLAR & VERİ YÜKLEME
# -----------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


def get_base64(file_path):
    if not os.path.isfile(file_path):
        return None
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def load_tooltip_data_direct(csv_path: Path) -> Dict[int, Dict[str, float]]:
    data = {}
    try:
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            for line in lines[1:]:
                parts = line.strip().split(';')
                if len(parts) >= 3:
                    try:
                        nid = int(parts[0])
                        delay = float(parts[1].replace(',', '.'))
                        rel = float(parts[2].replace(',', '.'))
                        data[nid] = {'delay': delay, 'rel': rel}
                    except:
                        continue
    except:
        pass
    return data


@st.cache_resource
def load_data():
    project_root = Path(__file__).resolve().parents[1]
    csv_dir = project_root / "graph_csv"
    try:
        graph = load_graph_teacher_csv(csv_dir / "NodeData.csv", csv_dir / "EdgeData.csv")
        scenarios = load_scenarios_from_csv(csv_dir / "DemandData.csv")
        tooltip_data = load_tooltip_data_direct(csv_dir / "NodeData.csv")
        return graph, scenarios, tooltip_data
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return None, None, {}


# -----------------------------------------------------------------------------
# 2. ESTETİK TASARIM (CSS)
# -----------------------------------------------------------------------------

def inject_custom_design():
    arkaplan_yolu = BASE_DIR / "arkaplan.webp"
    bin_str = get_base64(arkaplan_yolu)

    bg_style = f"""
        .stApp {{
            background-image: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)),
                              url("data:image/webp;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
    """ if bin_str else ".stApp { background-color: #050a14; }"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Rajdhani:wght@300;500;700&display=swap');
    {bg_style}

    /* ==================================================
       GENEL DARK TEMA
       ================================================== */
    .stApp {{
        color: #E6F1FF !important;
        font-family: 'Rajdhani', sans-serif;
    }}

    .stMarkdown p,
    .stMarkdown span,
    label,
    h1, h2, h3, h4, h5, h6 {{
        color: #E6F1FF !important;
        opacity: 1 !important;
    }}

    h2, h3 {{
        color: #00d4ff !important;
        font-family: 'Orbitron', sans-serif;
    }}

    /* ==================================================
       HEADER / MENU GİZLE
       ================================================== */
    [data-testid="stHeader"] {{
        background: transparent !important;
    }}

    .stAppDeployButton {{ display: none !important; }}
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    /* ==================================================
       ANA CONTAINER
       ================================================== */
    .main .block-container {{
        background: rgba(10,15,25,0.88);
        backdrop-filter: blur(15px);
        border-radius: 30px;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
        margin: 0 auto !important;
        border: 1px solid rgba(0,212,255,0.25);
        margin-top: -30px !important;
    }}

    /* ==================================================
       SIDEBAR
       ================================================== */
    [data-testid="stSidebar"] {{
        background-color: rgba(5,10,20,0.98) !important;
        border-right: 2px solid #00d4ff !important;
    }}

    /* ==================================================
       TAB BAŞLIKLARI
       ================================================== */
    .stTabs [data-baseweb="tab"] p {{
        color: #00d4ff !important;
        font-family: 'Orbitron', sans-serif;
        font-size: 14px;
    }}

    /* ==================================================
       BUTONLAR
       ================================================== */
    div.stButton > button {{
        background: linear-gradient(45deg, #00CCFF, #0077ff) !important;
        color: #ffffff !important;
        border-radius: 50px !important;
        font-family: 'Orbitron', sans-serif;
        box-shadow: 0 0 20px rgba(0,212,255,0.6);
        border: none !important;
        width: 100%;
        height: 45px;
    }}

    /* ==================================================
       METRIC KUTULARI (NET DEĞERLER)
       ================================================== */
    [data-testid="stMetric"] {{
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(0,212,255,0.35) !important;
        border-radius: 15px !important;
    }}

    div[data-testid="stMetricLabel"] {{
        color: #00d4ff !important;
        opacity: 1 !important;
        font-weight: 600;
    }}

    div[data-testid="stMetricValue"] {{
        color: #E6F1FF !important;
        opacity: 1 !important;
        font-weight: 700 !important;
    }}

    /* ==================================================
       SELECTBOX (STABLE)
       ================================================== */
    div[data-baseweb="select"] > div {{
        background-color: rgba(5,10,20,0.95) !important;
        border: 1px solid rgba(0,212,255,0.35) !important;
        color: #E6F1FF !important;
    }}

    div[data-baseweb="popover"],
    div[data-baseweb="menu"] {{
        background-color: #0b1220 !important;
    }}

    div[data-baseweb="option"] {{
        background-color: #0b1220 !important;
        color: #E6F1FF !important;
        opacity: 1 !important;
    }}

    /* ==================================================
       TABLE / STRATEJİ KARŞILAŞTIRMA FIX (SON KALAN YER)
       ================================================== */
    [data-testid="stTable"],
    [data-testid="stDataFrame"] {{
        background: rgba(5,10,20,0.95) !important;
        border-radius: 15px !important;
    }}

    [data-testid="stTable"] *,
    [data-testid="stDataFrame"] * {{
        color: #E6F1FF !important;
        opacity: 1 !important;
    }}

    /* ==================================================
       ROTA KUTUSU (st.code) – BEYAZ + SİYAH YAZI
       ================================================== */
    div[data-testid="stCodeBlock"] {{
        background: #ffffff !important;
        border-radius: 18px !important;
    }}

    div[data-testid="stCodeBlock"] pre {{
        background: transparent !important;
        margin: 0 !important;
    }}

    div[data-testid="stCodeBlock"] code {{
        color: #000000 !important;
        font-weight: 500;
        font-size: 15px !important;
    }}

    div[data-testid="stCodeBlock"] button {{
        color: #000000 !important;
    }}

    </style>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 3. GRAFİK ÇİZİM FONKSİYONU
# -----------------------------------------------------------------------------

def draw_network_interactive(graph_obj, path_nodes, tooltip_data):
    """
    Gelişmiş topoloji çizimi: Spring layout, yol vurgulama ve performans optimizasyonu içerir.
    """
    G_temp = nx.Graph()

    # 1. Node Yükleme
    nodes_iter = []
    if hasattr(graph_obj, "nodes"):
        val = graph_obj.nodes
        nodes_iter = val.values() if isinstance(val, dict) else val

    for node in nodes_iter:
        nid = getattr(node, 'id', node)
        G_temp.add_node(nid)

    # 2. Edge Yükleme
    if hasattr(graph_obj, "adj"):
        adj_data = graph_obj.adj
        for u, neighbors in adj_data.items():
            if isinstance(neighbors, dict):
                for v in neighbors:
                    G_temp.add_edge(u, v)
            elif isinstance(neighbors, list):
                for v in neighbors:
                    G_temp.add_edge(u, v)

    # 3. Yerleşim Algoritması
    N = len(G_temp.nodes)
    k_val = 0.9 / math.sqrt(N) if N > 0 else 0.2

    if len(G_temp.edges) > 0:
        pos = nx.spring_layout(G_temp, seed=42, k=k_val, iterations=50)
    else:
        pos = nx.circular_layout(G_temp)

    # 4. Yol Kenarlarını Belirleme
    path_edges = set()
    if path_nodes and len(path_nodes) > 1:
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            path_edges.add((u, v))
            path_edges.add((v, u))

    # Renk Tanımları
    COLOR_SOURCE = "#00FF66"  # Neon Yeşil
    COLOR_DEST = "#FFA500"  # Neon Turuncu
    COLOR_PATH = "#FF0055"  # Neon Kırmızı/Pembe
    COLOR_DEFAULT = "#FFF8F8"  # Koyu Gri

    node_x, node_y, node_text = [], [], []
    node_color, node_size, node_line = [], [], []

    for nid, (x, y) in pos.items():
        node_x.append(x)
        node_y.append(y)

        info = tooltip_data.get(nid, {'delay': 0.0, 'rel': 0.0})
        node_text.append(
            f"<b>Node {nid}</b><br>"
            f"⏱ {info['delay']:.2f} ms<br>"
            f"🛡 {info['rel']:.4f}"
        )

        if path_nodes and nid in path_nodes:
            if nid == path_nodes[0]:
                node_color.append(COLOR_SOURCE)
                node_size.append(20)
                node_line.append("white")
            elif nid == path_nodes[-1]:
                node_color.append(COLOR_DEST)
                node_size.append(20)
                node_line.append("white")
            else:
                node_color.append(COLOR_PATH)
                node_size.append(12)
                node_line.append("white")
        else:
            node_color.append(COLOR_DEFAULT)
            node_size.append(6)
            node_line.append("rgba(0,0,0,0)")

    # Kenarları Çizme (Optimizasyonlu)
    edge_x, edge_y = [], []
    random.seed(42)

    for u, v in G_temp.edges():
        is_in_path = (u, v) in path_edges
        # Sadece rotadakileri ve diğerlerinin %3'ünü çiz
        if is_in_path or random.random() < 0.03:
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

    trace_edges = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=1, color="rgba(200, 200, 200, 0.6)"),
        hoverinfo="none"
    )

    # Rota Çizgisi
    path_x, path_y = [], []
    if path_nodes and len(path_nodes) > 1:
        for i in range(len(path_nodes) - 1):
            u, v = path_nodes[i], path_nodes[i + 1]
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                path_x.extend([x0, x1, None])
                path_y.extend([y0, y1, None])

    trace_path = go.Scatter(
        x=path_x, y=path_y,
        mode="lines",
        line=dict(width=3, color=COLOR_PATH),
        opacity=1,
        hoverinfo="none"
    )

    # Node'lar
    trace_nodes = go.Scatter(
        x=node_x, y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            color=node_color,
            size=node_size,
            line=dict(width=1, color=node_line)
        )
    )

    fig = go.Figure(
        data=[trace_edges, trace_path, trace_nodes],
        layout=go.Layout(
            title="",
            showlegend=False,
            hovermode="closest",
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(227, 237, 252, 0.2)", 
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            dragmode="pan",
            height=550
        )
    )

    return fig


# -----------------------------------------------------------------------------
# 4. ANA UYGULAMA
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="BSM307 Network Projesi", layout="wide")
    inject_custom_design()

    banner_str = get_base64(BASE_DIR / "baslik.webp")
    if banner_str:
        st.markdown(f"""
            <div style="text-align:center; margin-top: -260px; margin-bottom: -100px;">
                <img src="data:image/png;base64,{banner_str}" 
                     style="width:100%; max-width:850px; position: relative; z-index: 100;">
            </div>
            """, unsafe_allow_html=True)

    data = load_data()
    if not data or not data[0]: return
    graph, scenarios, tooltip_data = data

    algorithms = default_algorithms()
    alg_names = [a.name for a in algorithms]

    with st.sidebar:
        st.markdown("<h2 style='color:#00d4ff; text-align:center; font-family:Orbitron; margin-top:50px;'>AYARLAR</h2>",
                    unsafe_allow_html=True)
        sel_idx = st.selectbox("🎯 Senaryo Seçimi", range(len(scenarios)), format_func=lambda x: f"Talep #{x + 1}")
        scen = scenarios[sel_idx]
        st.success(f"**Kaynak:** {scen.s} ➔ **Hedef:** {scen.d}")
        st.info(f"**Gereksinim:** {scen.bandwidth_requirement} Mbps")

    tab1, tab2, tab3 = st.tabs(["🔍 ANALİZ", "📊 PARETO", "⚔️ KIYASLAMA"])

    with tab1:
        st.subheader("Rota Hesaplama")
        c_alg, c_void = st.columns([1, 1])
        sel_alg_name = c_alg.selectbox("Algoritma", alg_names)

        c1, c2, c3 = st.columns(3)
        w = {"delay": c1.slider("Gecikme (D)", 0.0, 1.0, 0.5),
             "reliability": c2.slider("Güvenlik (R)", 0.0, 1.0, 0.3),
             "resource": c3.slider("Maliyet (C)", 0.0, 1.0, 0.2)}

        if st.button("🚀 ROTAYI HESAPLA"):
            alg_obj = next(a for a in algorithms if a.name == sel_alg_name)
            res = alg_obj.run(graph, scen.s, scen.d, w, scen.bandwidth_requirement, seed=42)

            if res and res.path:
                fig = draw_network_interactive(graph, res.path, tooltip_data)
                if fig:
                    # GÜNCELLEME BURADA: 'select2d' ve 'lasso2d' butonları kaldırıldı.
                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                        config={
                            'scrollZoom': True,
                            'displayModeBar': True,
                            'modeBarButtonsToRemove': ['select2d', 'lasso2d']
                        }
                    )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("🏆 Skor", f"{res.cost:.4f}")
                m2.metric("⏱️ Gecikme", f"{res.details.get('total_delay', 0):.2f} ms")
                m3.metric("🛡️ Güven", f"%{res.details.get('raw_reliability', 0) * 100:.1f}")
                m4.metric("🔗 Hop", len(res.path) - 1)
                st.code(f"Rota: {' -> '.join(map(str, res.path))}")
            else:
                st.warning("Belirtilen kriterlerde yol bulunamadı.")

    with tab2:
        st.subheader("Strateji Karşılaştırması")
        p_alg = st.selectbox("Algoritma Seçin", alg_names, key="p")
        if st.button("📊 STRATEJİLERİ ANALİZ ET"):
            obj = next(a for a in algorithms if a.name == p_alg)
            cfgs = [("Hız Odaklı", {"delay": 1.0, "reliability": 0, "resource": 0}),
                    ("Güvenlik Odaklı", {"delay": 0, "reliability": 1.0, "resource": 0}),
                    ("Dengeli", {"delay": 0.33, "reliability": 0.33, "resource": 0.33})]
            p_data = []
            for lbl, wg in cfgs:
                r = obj.run(graph, scen.s, scen.d, wg, scen.bandwidth_requirement)
                if r.path:
                    p_data.append(
                        {"Strateji": lbl, "Skor": round(r.cost, 4), "Gecikme": r.details.get("total_delay", 0)})
            st.table(pd.DataFrame(p_data))

    with tab3:
        st.subheader("Performans Arenası")
        col1, col2 = st.columns(2)
        a1 = col1.selectbox("1. Algoritma", alg_names, index=0)
        a2 = col2.selectbox("2. Algoritma", alg_names, index=min(1, len(alg_names) - 1))

        if st.button("⚔️ ALGORİTMALARI YARIŞTIR"):
            cols = st.columns(2)
            for i, name in enumerate([a1, a2]):
                obj = next(a for a in algorithms if a.name == name)
                t0 = time.time()
                r = obj.run(graph, scen.s, scen.d, {"delay": 0.5, "reliability": 0.3, "resource": 0.2},
                            scen.bandwidth_requirement)
                dt = time.time() - t0
                with cols[i]:
                    st.info(f"**{name}**")
                    if r.path:
                        st.metric("Süre", f"{dt:.4f}s")
                        st.metric("Skor", f"{r.cost:.4f}")
                        st.code("->".join(map(str, r.path)))
                    else:
                        st.error("Yol bulunamadı.")



if __name__ == "__main__":
    main()