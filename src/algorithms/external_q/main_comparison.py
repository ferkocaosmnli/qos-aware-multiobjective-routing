import time
import csv
import math
import networkx as nx
from topology_generator import generate_random_graph
from q_learning import QLearning
from ga_routing import GeneticAlgorithm
from aco_routing import AntColonyOptimization

def calculate_path_metrics(graph_model, path, weights):
    if not path or len(path) < 2:
        return float('inf'), 0, 0, 0
        
    total_cost = 0
    total_delay = 0
    total_reliability_log = 0
    min_bandwidth = float('inf')
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i+1]
        edge_data = graph_model.graph[u][v]
        
        delay = edge_data.get("delay", 5)
        reliability = edge_data.get("reliability", 0.99)
        bandwidth = edge_data.get("bandwidth", 100)
        
        c_delay = delay
        r_val = reliability if reliability > 0.0001 else 0.0001
        c_rel = -math.log(r_val)
        c_res = 1000.0 / bandwidth if bandwidth > 0 else 100.0
        
        step_cost = (
            weights["w_delay"] * c_delay
            + weights["w_reliability"] * c_rel
            + weights["w_resource"] * c_res
        )
        total_cost += step_cost
        
        total_delay += delay
        total_reliability_log += math.log(r_val)
        if bandwidth < min_bandwidth:
            min_bandwidth = bandwidth
            
    total_reliability = math.exp(total_reliability_log)
    return total_cost, total_delay, total_reliability, min_bandwidth

def main():
    # 1. Generate Network
    print("Generating network topology...")
    graph_model = generate_random_graph(num_nodes=250, edge_prob=0.08)
    
    nodes = list(graph_model.graph.nodes())
    source = nodes[0]
    destination = nodes[-1]
    
    # Ensure connectivity
    if not nx.has_path(graph_model.graph, source, destination):
        print("No path exists between source and destination. Regenerating...")
        while not nx.has_path(graph_model.graph, source, destination):
             graph_model = generate_random_graph(num_nodes=250, edge_prob=0.08)
             nodes = list(graph_model.graph.nodes())
             source = nodes[0]
             destination = nodes[-1]

    print(f"Source: {source}, Destination: {destination}")
    
    weights = {
        "w_delay": 0.4,
        "w_reliability": 0.4,
        "w_resource": 0.2
    }
    
    results = []
    
    # --- Q-Learning ---
    print("\nRunning Q-Learning...")
    start_time = time.time()
    ql = QLearning(graph_model, source, destination, weights, params={"episodes": 1000, "alpha": 0.1, "gamma": 0.9, "epsilon": 0.1})
    ql_path = ql.find_path()
    ql_time = time.time() - start_time
    ql_cost, ql_delay, ql_rel, ql_bw = calculate_path_metrics(graph_model, ql_path, weights)
    results.append(["Q-Learning", ql_time, ql_cost, ql_delay, ql_rel, ql_bw, len(ql_path)])
    print(f"Q-Learning: Cost={ql_cost:.4f}, Time={ql_time:.4f}s")

    # --- Genetic Algorithm ---
    print("\nRunning Genetic Algorithm...")
    start_time = time.time()
    ga = GeneticAlgorithm(graph_model, source, destination, weights, params={"pop_size": 50, "generations": 100})
    ga_path = ga.run()
    ga_time = time.time() - start_time
    ga_cost, ga_delay, ga_rel, ga_bw = calculate_path_metrics(graph_model, ga_path, weights)
    results.append(["Genetic Algorithm", ga_time, ga_cost, ga_delay, ga_rel, ga_bw, len(ga_path) if ga_path else 0])
    print(f"GA: Cost={ga_cost:.4f}, Time={ga_time:.4f}s")

    # --- Ant Colony Optimization ---
    print("\nRunning Ant Colony Optimization...")
    start_time = time.time()
    aco = AntColonyOptimization(graph_model, source, destination, weights, params={"num_ants": 20, "iterations": 50})
    aco_path = aco.run()
    aco_time = time.time() - start_time
    aco_cost, aco_delay, aco_rel, aco_bw = calculate_path_metrics(graph_model, aco_path, weights)
    results.append(["ACO", aco_time, aco_cost, aco_delay, aco_rel, aco_bw, len(aco_path) if aco_path else 0])
    print(f"ACO: Cost={aco_cost:.4f}, Time={aco_time:.4f}s")

    # Save to CSV
    csv_filename = "comparison_results.csv"
    print(f"\nSaving results to {csv_filename}...")
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "Execution Time (s)", "Total Cost", "Total Delay", "Total Reliability", "Min Bandwidth", "Path Length"])
        writer.writerows(results)
        
    print("Comparison complete.")

if __name__ == "__main__":
    main()
