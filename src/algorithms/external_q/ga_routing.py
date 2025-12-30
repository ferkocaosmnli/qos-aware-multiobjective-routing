import random
import math
import networkx as nx

class GeneticAlgorithm:
    def __init__(self, graph_model, source, destination, weights, params=None):
        self.graph_model = graph_model
        self.source = source
        self.destination = destination
        self.weights = weights
        self.params = params if params else {}
        
        self.pop_size = self.params.get("pop_size", 50)
        self.generations = self.params.get("generations", 100)
        self.mutation_rate = self.params.get("mutation_rate", 0.1)
        self.elite_size = self.params.get("elite_size", 2)

    def generate_random_path(self):
        """Generates a random loop-free path from source to destination."""
        try:
            # Simple random walk with loop avoidance
            path = [self.source]
            current = self.source
            visited = {self.source}
            
            while current != self.destination:
                neighbors = list(self.graph_model.graph.neighbors(current))
                valid_neighbors = [n for n in neighbors if n not in visited]
                
                if not valid_neighbors:
                    return None # Dead end
                
                next_node = random.choice(valid_neighbors)
                path.append(next_node)
                visited.add(next_node)
                current = next_node
                
                if len(path) > 250: # Safety break
                    return None
            return path
        except Exception:
            return None

    def initialize_population(self):
        population = []
        attempts = 0
        while len(population) < self.pop_size and attempts < self.pop_size * 10:
            path = self.generate_random_path()
            if path:
                population.append(path)
            attempts += 1
        return population

    def calculate_cost(self, path):
        total_cost = 0
        if not path:
            return float('inf')
            
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if not self.graph_model.graph.has_edge(u, v):
                return float('inf')
                
            edge_data = self.graph_model.graph[u][v]
            delay = edge_data.get("delay", 5)
            reliability = edge_data.get("reliability", 0.99)
            bandwidth = edge_data.get("bandwidth", 100)

            c_delay = delay
            r_val = reliability if reliability > 0.0001 else 0.0001
            c_rel = -math.log(r_val)
            c_res = 1000.0 / bandwidth if bandwidth > 0 else 100.0

            step_cost = (
                self.weights["w_delay"] * c_delay
                + self.weights["w_reliability"] * c_rel
                + self.weights["w_resource"] * c_res
            )
            total_cost += step_cost
        return total_cost

    def selection(self, population, fitnesses):
        # Tournament selection
        tournament_size = 3
        selected = []
        for _ in range(len(population)):
            candidates = random.sample(list(zip(population, fitnesses)), min(tournament_size, len(population)))
            winner = min(candidates, key=lambda x: x[1]) # Min cost is better
            selected.append(winner[0])
        return selected

    def crossover(self, parent1, parent2):
        # Find common nodes to swap segments
        common_nodes = [node for node in parent1 if node in parent2 and node != self.source and node != self.destination]
        
        if not common_nodes:
            return parent1 # No crossover possible
            
        crossover_point = random.choice(common_nodes)
        
        idx1 = parent1.index(crossover_point)
        idx2 = parent2.index(crossover_point)
        
        # Create new paths
        child1 = parent1[:idx1] + parent2[idx2:]
        child2 = parent2[:idx2] + parent1[idx1:]
        
        # Remove loops
        child1 = self.remove_loops(child1)
        child2 = self.remove_loops(child2)
        
        return child1 if random.random() < 0.5 else child2

    def remove_loops(self, path):
        if len(path) != len(set(path)):
            new_path = []
            visited = set()
            for node in path:
                if node in visited:
                    # Loop detected, cut back to first occurrence
                    try:
                        first_idx = new_path.index(node)
                        new_path = new_path[:first_idx+1]
                        # Rebuild visited set
                        visited = set(new_path)
                    except ValueError:
                        pass
                else:
                    new_path.append(node)
                    visited.add(node)
            return new_path
        return path

    def mutation(self, path):
        if len(path) < 3:
            return path
            
        # Pick two points and try to find a new subpath between them
        idx1, idx2 = sorted(random.sample(range(len(path)), 2))
        node1 = path[idx1]
        node2 = path[idx2]
        
        # Try to find a short random path between node1 and node2
        try:
            subpath = nx.shortest_path(self.graph_model.graph, node1, node2) # Using shortest path as mutation heuristic
            # Or purely random walk for more diversity?
            # Let's stick to shortest path to guide towards better solutions, 
            # or just a random neighbor if we want pure mutation.
            # Let's try a small random deviation.
            pass
        except nx.NetworkXNoPath:
            return path
            
        new_path = path[:idx1] + subpath + path[idx2+1:]
        return self.remove_loops(new_path)

    def run(self):
        population = self.initialize_population()
        if not population:
            return None # Failed to init
            
        best_path = None
        best_cost = float('inf')
        
        for gen in range(self.generations):
            fitnesses = [self.calculate_cost(p) for p in population]
            
            # Track best
            min_cost = min(fitnesses)
            if min_cost < best_cost:
                best_cost = min_cost
                best_path = population[fitnesses.index(min_cost)]
            
            # Elitism
            sorted_pop = sorted(zip(population, fitnesses), key=lambda x: x[1])
            new_population = [x[0] for x in sorted_pop[:self.elite_size]]
            
            # Selection
            parents = self.selection(population, fitnesses)
            
            # Crossover & Mutation
            while len(new_population) < self.pop_size:
                p1 = random.choice(parents)
                p2 = random.choice(parents)
                child = self.crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    child = self.mutation(child)
                new_population.append(child)
                
            population = new_population
            
        return best_path
