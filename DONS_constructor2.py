# dons_ba_er_generator.py

import json
import random
import argparse
import sys
import os
import heapq
import networkx as nx

def generate_base_topology(graph_type: str, num_nodes: int, **kwargs) -> dict:
    """
    Generates a base network topology using a specified graph model.

    Args:
        graph_type (str): Type of graph to generate ('ER' or 'BA').
        num_nodes (int): The number of nodes in the network.
        **kwargs: Additional keyword arguments for the graph generation.

    Returns:
        dict: The adjacency list of the generated graph.
    """
    if graph_type.upper() == 'ER':
        # p is the probability of an edge existing
        p = kwargs.get('p', 0.2)
        print(f"Generating an ER graph with num_nodes={num_nodes}, p={p}...")
        graph = nx.gnp_random_graph(num_nodes, p)
    elif graph_type.upper() == 'BA':
        # m is the number of edges to attach from a new node to existing nodes
        m = kwargs.get('m', 2)
        print(f"Generating a BA graph with num_nodes={num_nodes}, m={m}...")
        graph = nx.barabasi_albert_graph(num_nodes, m)
    else:
        print("Error: Invalid graph type. Please choose 'ER' or 'BA'.")
        sys.exit(1)
        
    # Assign random RTT weights to edges
    min_rtt = kwargs.get('min_rtt', 10)
    max_rtt = kwargs.get('max_rtt', 100)
    for u, v in graph.edges():
        rtt = random.randint(min_rtt, max_rtt)
        graph[u][v]['weight'] = rtt
        
    # Convert to a format that our functions can use
    adj_list = {i: {} for i in range(num_nodes)}
    for u, v, data in graph.edges(data=True):
        weight = data['weight']
        adj_list[u][v] = weight
        adj_list[v][u] = weight
        
    # Check if the graph is connected
    if num_nodes > 0 and not nx.is_connected(graph):
        print("Warning: The generated graph is not connected. "
              "Consider increasing parameters (p for ER, m for BA).")
    
    return adj_list

def prims_mst_edges(num_nodes: int, adj_list: dict) -> list:
    """
    Computes the edges of the Minimum Spanning Tree (MST) using Prim's algorithm.

    Args:
        num_nodes (int): The number of nodes in the graph.
        adj_list (dict): The adjacency list of the graph.

    Returns:
        list: A list of edges representing the MST (source, target, weight).
    """
    if num_nodes == 0:
        return []

    mst = []
    visited = [False] * num_nodes
    min_heap = [(0, 0, -1)]  # (weight, node, parent)

    while min_heap and len(mst) < num_nodes - 1:
        weight, u, parent = heapq.heappop(min_heap)

        if visited[u]:
            continue

        visited[u] = True
        if parent != -1:
            mst.append((parent, u, weight))

        for v, edge_weight in adj_list[u].items():
            if not visited[v]:
                heapq.heappush(min_heap, (edge_weight, v, u))

    if len(mst) != num_nodes - 1 and num_nodes > 1:
        print("Warning: The graph is not fully connected, MST is incomplete.")
    
    return mst

def find_ons(num_nodes: int, mst_edges: list) -> dict:
    """
    Finds the Optimized Neighbor Set (ONS) for each node from the MST.

    Args:
        num_nodes (int): The number of nodes.
        mst_edges (list): The edges of the MST.

    Returns:
        dict: A dictionary where keys are node IDs and values are their ONS (neighbor: weight).
    """
    ons_sets = {i: {} for i in range(num_nodes)}
    for u, v, weight in mst_edges:
        ons_sets[u][v] = weight
        ons_sets[v][u] = weight
    return ons_sets

def create_topology_json(file_path: str, full_topology: dict, nodes_ons: dict, graph_type: str, num_nodes: int):
    """
    Saves the full topology and the ONS data to a JSON file.
    """
    nodes = [{"id": i} for i in range(num_nodes)]
    edges = []
    
    processed_edges = set()
    for u, neighbors in full_topology.items():
        for v, weight in neighbors.items():
            if (v, u) not in processed_edges:
                edges.append({"source": u, "target": v, "weight": weight})
                processed_edges.add((u, v))

    data = {
        "directed": False,
        "multigraph": False,
        "graph_info": {
            "type": graph_type.upper(),
            "num_nodes": num_nodes,
        },
        "nodes": nodes,
        "edges": edges,
        "ons_sets": nodes_ons
    }

    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Successfully created DONS-optimized {graph_type.upper()} topology file at: {file_path}")
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a DONS-optimized network from an ER or BA graph.")
    parser.add_argument("--graph_type", type=str, choices=['ER', 'BA'], required=True, help="Type of base graph to generate ('ER' or 'BA').")
    parser.add_argument("--num_nodes", type=int, default=100, help="Number of nodes in the network.")
    parser.add_argument("--p", type=float, default=0.05, help="Probability 'p' for ER graph.")
    parser.add_argument("--m", type=int, default=3, help="Number of edges 'm' for BA graph.")
    parser.add_argument("--min_rtt", type=int, default=10, help="Minimum RTT in milliseconds.")
    parser.add_argument("--max_rtt", type=int, default=100, help="Maximum RTT in milliseconds.")
    parser.add_argument("--output_file", type=str, default="topology/dons_optimized_topology.json", help="Output file path.")
    args = parser.parse_args()

    # Ensure the output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Generating a base {args.graph_type.upper()} network with {args.num_nodes} nodes...")
    if args.graph_type.upper() == 'ER':
        base_topology = generate_base_topology(args.graph_type, args.num_nodes, p=args.p, min_rtt=args.min_rtt, max_rtt=args.max_rtt)
    else: # BA
        base_topology = generate_base_topology(args.graph_type, args.num_nodes, m=args.m, min_rtt=args.min_rtt, max_rtt=args.max_rtt)

    print("Computing the Minimum Spanning Tree...")
    mst_edges = prims_mst_edges(args.num_nodes, base_topology)
    
    print("Deriving the Optimized Neighbor Set (ONS) for each node...")
    nodes_ons = find_ons(args.num_nodes, mst_edges)

    print("Creating output JSON file...")
    create_topology_json(args.output_file, base_topology, nodes_ons, args.graph_type, args.num_nodes)

    print("DONS-optimized topology generation complete.")
