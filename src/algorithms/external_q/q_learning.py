import random
import math
class QLearning:
    def __init__(self, graph_model, source, destination, weights, params=None):
        """
        Q-Learning Algoritması (Off-Policy).
        Farkı: Bir sonraki durumun en iyi Q değerini (Max Q) kullanarak günceller.
        """
        self.graph_model = graph_model  # Model ağ grafiği
        self.source = source  # Kaynak düğümü
        self.destination = destination  # Hedef düğümü
        self.weights = weights  # Ağırlıklar (delay, reliability, resource)

        # Hiperparametreler
        self.params = params if params else {}
        self.alpha = self.params.get("alpha", 0.1)  # Öğrenme hızı
        self.gamma = self.params.get("gamma", 0.9)  # Gelecek ödül katsayısı
        self.epsilon = self.params.get("epsilon", 0.1)  # Keşif oranı
        self.episodes = self.params.get("episodes", 1000)  # Eğitim iterasyon sayısı

        # Q-Tablosu: {state: {action: q_value}}
        self.q_table = {}

    def get_q_value(self, state, action):
        """Q-Değerini getir, yoksa 0 döndür."""
        if state not in self.q_table:
            self.q_table[state] = {}
        return self.q_table[state].get(action, 0.0)

    def get_neighbors(self, node):
        """Güvenli komşu alma fonksiyonu"""
        try:
            return list(self.graph_model.graph.neighbors(node))
        except AttributeError:
            return list(self.graph_model.neighbors(node))

    def get_max_q(self, state):
        """Bir durumdaki (State) mümkün olan EN YÜKSEK Q değerini döndürür."""
        neighbors = self.get_neighbors(state)
        if not neighbors:
            return 0.0

        if state not in self.q_table:
            return 0.0

        # O durumdaki tüm eylemlerin Q değerlerini kontrol et
        q_values = [self.get_q_value(state, action) for action in neighbors]
        return max(q_values) if q_values else 0.0

    def choose_action(self, state, visited=None):
        """Epsilon-Greedy ile eylem seçimi (Hareket etmek için)"""
        if visited is None:
            visited = set()

        neighbors = self.get_neighbors(state)
        if not neighbors:
            return None

        # Sadece ziyaret edilmemiş komşuları düşün (Loop Prevention)
        available_neighbors = [n for n in neighbors if n not in visited]
        if not available_neighbors:
            return None

        # Keşif (Exploration)
        if random.random() < self.epsilon:
            return random.choice(available_neighbors)

        # Sömürü (Exploitation)
        best_action = None
        max_q = -float("inf")

        for action in available_neighbors:
            q_val = self.get_q_value(state, action)
            if q_val > max_q:
                max_q = q_val
                best_action = action

        return best_action if best_action is not None else random.choice(available_neighbors)

    def calculate_total_path_cost(self, path):
        """Bir yolun toplam maliyetini hesapla (Node ve Link metrikleri dahil)"""
        if len(path) < 2:
            return float('inf')

        total_cost = 0.0

        for i in range(len(path) - 1):
            current_node = path[i]
            next_node = path[i + 1]

            try:
                # Link verileri
                edge_data = self.graph_model.graph[current_node][next_node]

                delay = edge_data.get("delay", 5)
                reliability = edge_data.get("reliability", 0.99)
                bandwidth = edge_data.get("bandwidth", 100)

                # --- LINK COST HESABI ---
                # 1. Gecikme
                c_delay = delay
                # 2. Güvenilirlik (-log)
                r_val = reliability if reliability > 0.0001 else 0.0001
                c_rel = -math.log(r_val)
                # 3. Kaynak (1000/BW)
                c_res = 1000.0 / bandwidth if bandwidth > 0 else 100.0

                # Link maliyeti
                link_cost = (
                        self.weights["w_delay"] * c_delay
                        + self.weights["w_reliability"] * c_rel
                        + self.weights["w_resource"] * c_res
                )

                # --- NODE COST HESABI (PDF Metric Compliance) ---
                # Sadece hedef düğüm için node maliyeti ekle (kaynak hariç)
                if i > 0:  # İlk düğüm (kaynak) hariç
                    node_data = self.graph_model.graph.nodes[next_node]
                    node_delay = node_data.get("processing_delay", 1)
                    node_reliability = node_data.get("reliability", 0.999)

                    # Node processing delay
                    c_node_delay = node_delay
                    # Node reliability (-log)
                    r_node_val = node_reliability if node_reliability > 0.0001 else 0.0001
                    c_node_rel = -math.log(r_node_val)

                    node_cost = (
                            self.weights["w_delay"] * c_node_delay
                            + self.weights["w_reliability"] * c_node_rel
                    )

                    total_cost += link_cost + node_cost
                else:
                    total_cost += link_cost

            except Exception:
                # Varsayılan maliyet
                total_cost += 10.0

        return total_cost

    def calculate_reward(self, current_node, next_node, path_so_far=None):
        """Güncellenmiş ödül hesaplama (PDF uyumlu)"""
        if next_node == self.destination:
            if path_so_far is not None:
                # Hedefe ulaşıldı: yolun tamamını al
                full_path = path_so_far + [next_node]
                total_cost = self.calculate_total_path_cost(full_path)

                # PDF önerisi: Reward = 1000 / TotalPathCost
                if total_cost > 0:
                    return 1000.0 / total_cost
                else:
                    return 1000.0  # Fallback
            else:
                # Path tracking yoksa eski davranış
                return 1000.0

        try:
            # Link verileri
            edge_data = self.graph_model.graph[current_node][next_node]

            delay = edge_data.get("delay", 5)
            reliability = edge_data.get("reliability", 0.99)
            bandwidth = edge_data.get("bandwidth", 100)

            # --- LINK COST HESABI ---
            # 1. Gecikme
            c_delay = delay
            # 2. Güvenilirlik (-log)
            r_val = reliability if reliability > 0.0001 else 0.0001
            c_rel = -math.log(r_val)
            # 3. Kaynak (1000/BW)
            c_res = 1000.0 / bandwidth if bandwidth > 0 else 100.0

            # Link maliyeti
            link_cost = (
                    self.weights["w_delay"] * c_delay
                    + self.weights["w_reliability"] * c_rel
                    + self.weights["w_resource"] * c_res
            )

            # --- NODE COST HESABI (PDF Metric Compliance) ---
            # Hedef düğümün node maliyeti
            node_data = self.graph_model.graph.nodes[next_node]
            node_delay = node_data.get("processing_delay", 1)
            node_reliability = node_data.get("reliability", 0.999)

            # Node processing delay
            c_node_delay = node_delay
            # Node reliability (-log)
            r_node_val = node_reliability if node_reliability > 0.0001 else 0.0001
            c_node_rel = -math.log(r_node_val)

            node_cost = (
                    self.weights["w_delay"] * c_node_delay
                    + self.weights["w_reliability"] * c_node_rel
            )

            # Toplam adım maliyeti
            total_step_cost = link_cost + node_cost

            return -total_step_cost  # Maliyet ne kadar azsa ödül o kadar büyük
        except Exception:
            return -10.0

    def train(self):
        """Q-Learning Eğitim Döngüsü (Loop prevention ile)"""
        for episode in range(self.episodes):
            if episode % 100 == 0:
                print(f"Episode {episode}/{self.episodes}")

            state = self.source
            visited = {state}  # Loop prevention için ziyaret edilen düğümler
            episode_path = [state]  # Bu bölümdeki yolu takip et

            step_count = 0
            while state != self.destination:
                # 1. Eylemi Seç (Loop prevention ile)
                action = self.choose_action(state, visited)
                if action is None:
                    # Hiç geçerli komşu yok, bölümü sonlandır
                    break

                next_state = action

                # 2. Ödülü Al (yol takibi ile)
                reward = self.calculate_reward(state, next_state, episode_path)

                # 3. Q-Learning Güncellemesi (BELLMAN DENKLEMİ)
                current_q = self.get_q_value(state, action)

                if next_state == self.destination:
                    max_next_q = 0.0
                else:
                    # Bir sonraki durumda alabileceğimiz MAKSİMUM puan
                    max_next_q = self.get_max_q(next_state)

                # Formül: Q(s,a) = Q(s,a) + alpha * [R + gamma * MaxQ(s') - Q(s,a)]
                new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

                # Kaydet
                if state not in self.q_table:
                    self.q_table[state] = {}
                self.q_table[state][action] = new_q

                # İlerle ve ziyaret edilenleri güncelle
                state = next_state
                visited.add(state)
                episode_path.append(state)

                step_count += 1
                if step_count > 250:
                    break

    def find_path(self):
        self.train()  # Önce eğit

        path = [self.source]
        current = self.source
        visited = {current}  # Loop prevention

        # Test için rastgeleliği kapat
        old_eps = self.epsilon
        self.epsilon = 0

        while current != self.destination:
            best = self.choose_action(current, visited)
            if best is None or best in visited:
                break
            path.append(best)
            current = best
            visited.add(current)
            if len(path) > 250:
                break

        self.epsilon = old_eps
        return path

if __name__ == "__main__":
    import csv
    from topology_generator import generate_random_graph
    import networkx as nx

    # 1. Generate Network
    print("Generating network topology...")
    graph_model = generate_random_graph(num_nodes=250, edge_prob=0.08)

    # Ensure source and destination are connected
    nodes = list(graph_model.graph.nodes())
    source = nodes[0]
    destination = nodes[-1]

    # Check connectivity
    if not nx.has_path(graph_model.graph, source, destination):
        print("No path exists between source and destination. Regenerating...")
        # Simple retry or just pick connected nodes
        while not nx.has_path(graph_model.graph, source, destination):
            graph_model = generate_random_graph(num_nodes=250, edge_prob=0.08)
            nodes = list(graph_model.graph.nodes())
            source = nodes[0]
            destination = nodes[-1]

    print(f"Source: {source}, Destination: {destination}")

    # 2. Define Weights
    weights = {
        "w_delay": 0.4,
        "w_reliability": 0.4,
        "w_resource": 0.2
    }

    # 3. Initialize Q-Learning
    ql = QLearning(graph_model, source, destination, weights,
                   params={"episodes": 1000, "alpha": 0.1, "gamma": 0.9, "epsilon": 0.1})

    # 4. Train
    print("Training Q-Learning agent...")
    ql.train()

    # 5. Find Path
    print("Finding optimal path...")
    optimal_path = ql.find_path()
    print(f"Optimal Path: {optimal_path}")

    # 6. Calculate Total Cost and Metrics (Node metrics dahil)
    total_cost = 0
    total_delay = 0
    total_node_delay = 0
    total_reliability_log = 0
    total_node_reliability_log = 0
    min_bandwidth = float('inf')

    if len(optimal_path) > 1:
        for i in range(len(optimal_path) - 1):
            u, v = optimal_path[i], optimal_path[i + 1]
            edge_data = graph_model.graph[u][v]

            # Edge metrics
            delay = edge_data.get("delay", 5)
            reliability = edge_data.get("reliability", 0.99)
            bandwidth = edge_data.get("bandwidth", 100)

            # Node metrics (target node)
            node_data = graph_model.graph.nodes[v]
            node_delay = node_data.get("processing_delay", 1)
            node_reliability = node_data.get("reliability", 0.999)

            # Cost calculation
            c_delay = delay
            r_val = reliability if reliability > 0.0001 else 0.0001
            c_rel = -math.log(r_val)
            c_res = 1000.0 / bandwidth if bandwidth > 0 else 100.0

            # Node cost components
            c_node_delay = node_delay
            r_node_val = node_reliability if node_reliability > 0.0001 else 0.0001
            c_node_rel = -math.log(r_node_val)

            step_cost = (
                    weights["w_delay"] * (c_delay + c_node_delay)
                    + weights["w_reliability"] * (c_rel + c_node_rel)
                    + weights["w_resource"] * c_res
            )
            total_cost += step_cost

            # Raw metrics for reporting
            total_delay += delay
            total_node_delay += node_delay
            total_reliability_log += math.log(r_val)
            total_node_reliability_log += math.log(r_node_val)
            if bandwidth < min_bandwidth:
                min_bandwidth = bandwidth

    # Calculate final reliability metrics
    total_reliability = math.exp(total_reliability_log) if len(optimal_path) > 1 else 0
    total_node_reliability = math.exp(total_node_reliability_log) if len(optimal_path) > 1 else 0

    print(f"Total Cost: {total_cost:.4f}")
    print(f"Total Link Delay: {total_delay:.2f} ms")
    print(f"Total Node Processing Delay: {total_node_delay:.2f} ms")
    print(f"Total Link Reliability: {total_reliability:.4f}")
    print(f"Total Node Reliability: {total_node_reliability:.4f}")
    print(f"Min Bandwidth: {min_bandwidth:.2f} Mbps")

    # 7. Save to CSV
    csv_filename = "q_learning_results.csv"
    print(f"Saving results to {csv_filename}...")
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Source", "Destination", "Path", "Total Cost", "Total Link Delay", "Total Node Delay",
                         "Total Link Reliability", "Total Node Reliability", "Min Bandwidth"])
        writer.writerow([source, destination, str(optimal_path), f"{total_cost:.4f}",
                         f"{total_delay:.4f}", f"{total_node_delay:.4f}",
                         f"{total_reliability:.4f}", f"{total_node_reliability:.4f}",
                         f"{min_bandwidth:.4f}"])

    print("Done.")