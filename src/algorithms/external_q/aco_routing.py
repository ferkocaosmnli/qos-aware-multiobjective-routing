import random
import math
import networkx as nx

class AntColonyOptimization:
    def __init__(self, graph_model, source, destination, weights, params=None):
        self.graph_model = graph_model
        self.source = source
        self.destination = destination
        self.weights = weights
        self.params = params if params else {}
        
        self.num_ants = self.params.get("num_ants", 20)
        self.iterations = self.params.get("iterations", 50)
        self.alpha = self.params.get("alpha", 1.0) # Pheromone importance
        self.beta = self.params.get("beta", 2.0) # Heuristic importance
        self.evaporation_rate = self.params.get("evaporation_rate", 0.5)
        self.pheromone_deposit = self.params.get("pheromone_deposit", 100.0)
        
        self.pheromones = {}
        self.initialize_pheromones()

    def initialize_pheromones(self):
        for u, v in self.graph_model.graph.edges():
            self.pheromones[(u, v)] = 1.0
            self.pheromones[(v, u)] = 1.0

    def get_heuristic(self, u, v):
        # Heuristic is inverse of cost (1/cost)
        edge_data = self.graph_model.graph[u][v]
        delay = edge_data.get("delay", 5)
        reliability = edge_data.get("reliability", 0.99)
        bandwidth = edge_data.get("bandwidth", 100)

        c_delay = delay
        r_val = reliability if reliability > 0.0001 else 0.0001
        c_rel = -math.log(r_val)
        c_res = 1000.0 / bandwidth if bandwidth > 0 else 100.0

        cost = (
            self.weights["w_delay"] * c_delay
            + self.weights["w_reliability"] * c_rel
            + self.weights["w_resource"] * c_res
        )
        return 1.0 / cost if cost > 0 else 1.0

    def construct_solution(self):
        path = [self.source]
        current = self.source
        visited = {self.source}
        
        while current != self.destination:
            neighbors = list(self.graph_model.graph.neighbors(current))
            valid_neighbors = [n for n in neighbors if n not in visited]
            
            if not valid_neighbors:
                return None # Dead end
            
            probabilities = []
            total_prob = 0
            
            for neighbor in valid_neighbors:
                pheromone = self.pheromones.get((current, neighbor), 1.0)
                heuristic = self.get_heuristic(current, neighbor)
                prob = (pheromone ** self.alpha) * (heuristic ** self.beta)
                probabilities.append(prob)
                total_prob += prob
            
            if total_prob == 0:
                next_node = random.choice(valid_neighbors)
            else:
                probabilities = [p / total_prob for p in probabilities]
                next_node = random.choices(valid_neighbors, weights=probabilities, k=1)[0]
                
            path.append(next_node)
            visited.add(next_node)
            current = next_node
            
            if len(path) > 250:
                return None
                
        return path

    def calculate_path_cost(self, path):
        total_cost = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            heuristic = self.get_heuristic(u, v)
            cost = 1.0 / heuristic if heuristic > 0 else float('inf')
            total_cost += cost
        return total_cost

    def update_pheromones(self, all_paths):
        # Evaporation
        for key in self.pheromones:
            self.pheromones[key] *= (1.0 - self.evaporation_rate)
            
        # Deposit
        for path in all_paths:
            if path is None: continue
            cost = self.calculate_path_cost(path)
            deposit = self.pheromone_deposit / cost if cost > 0 else 0
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i+1]
                if (u, v) in self.pheromones:
                    self.pheromones[(u, v)] += deposit
                if (v, u) in self.pheromones: # Undirected graph assumption usually
                    self.pheromones[(v, u)] += deposit

    def run(self):
        best_path = None
        best_cost = float('inf')
        
        for _ in range(self.iterations):
            all_paths = []
            for _ in range(self.num_ants):
                path = self.construct_solution()
                if path:
                    all_paths.append(path)
                    cost = self.calculate_path_cost(path)
                    if cost < best_cost:
                        best_cost = cost
                        best_path = path
            
            self.update_pheromones(all_paths)
            
        return best_path
