import json
import numpy as np
import argparse
import os
import random


def load_graph_from_json(json_file_path):
    """
    Loads a graph from a JSON file and converts it into a weighted distance matrix.
    The JSON is expected to contain 'nodes' and 'edges' lists.
    The 'weight' from the edges is used as the distance.

    Args:
        json_file_path (str): The path to the JSON file.

    Returns:
        tuple: A tuple containing:
               - The number of nodes (int).
               - The distance matrix (np.array).
               - A dictionary mapping original node IDs to their index (int) in the matrix.
    """
    with open(json_file_path, 'r') as f:
        graph_data = json.load(f)

    node_ids = [node['id'] for node in graph_data['nodes']]
    num_nodes = len(node_ids)

    id_to_index = {node_id: i for i, node_id in enumerate(node_ids)}

    distance_matrix = np.full((num_nodes, num_nodes), float('inf'))
    np.fill_diagonal(distance_matrix, 0)

    for edge in graph_data['edges']:
        source_id = edge['source']
        target_id = edge['target']
        weight = edge['weight']

        source_index = id_to_index[source_id]
        target_index = id_to_index[target_id]

        distance_matrix[source_index, target_index] = weight
        distance_matrix[target_index, source_index] = weight

    return num_nodes, distance_matrix, id_to_index


def calculate_distance(cluster_a, cluster_b, distance_matrix):
    """
    Calculates the distance between two clusters using complete linkage.
    """
    current_max_distance = float('-inf')
    found_valid_connection = False

    for node_i in cluster_a:
        for node_j in cluster_b:
            if node_i == node_j:
                continue

            distance = distance_matrix[node_i][node_j]

            if distance == float('inf'):
                continue

            found_valid_connection = True
            if distance > current_max_distance:
                current_max_distance = distance

    if not found_valid_connection:
        return float('inf')
    else:
        return current_max_distance


def agglomerative_clustering(num_nodes, num_clusters, distance_matrix):
    """
    Performs agglomerative clustering.
    """
    clusters = [[i] for i in range(num_nodes)]

    while len(clusters) > num_clusters:
        min_distance = float('inf')
        clusters_to_merge = None

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                distance = calculate_distance(clusters[i], clusters[j], distance_matrix)

                if distance < min_distance:
                    min_distance = distance
                    clusters_to_merge = (i, j)

        if clusters_to_merge is not None:
            index_a, index_b = clusters_to_merge

            if index_a > index_b:
                index_a, index_b = index_b, index_a

            new_cluster = clusters[index_a] + clusters[index_b]

            clusters.pop(index_b)
            clusters.pop(index_a)
            clusters.append(new_cluster)
        else:
            print("Warning: No more clusters could be merged. Remaining clusters might be disconnected.")
            break

    return clusters


def select_cluster_leaders(clusters, index_to_id_map):
    """
    Selects a leader for each cluster randomly.
    """
    cluster_leaders = {}
    for i, cluster in enumerate(clusters):
        leader_index = random.choice(cluster)
        leader_id = index_to_id_map[leader_index]
        cluster_leaders[f"Cluster {i + 1}"] = leader_id
    return cluster_leaders


# Implementation of Algorithm 2
def compute_mst_for_cluster(cluster_nodes, distance_matrix, start_node_index):
    """
    Constructs an MST for a single cluster using a Prim's-like approach
    based on the description of Algorithm 2.

    Args:
        cluster_nodes (list): List of node indices belonging to the cluster.
        distance_matrix (np.array): The global distance matrix.
        start_node_index (int): The index of the starting node for the MST.

    Returns:
        tuple: A tuple containing:
               - The parent array (p) for each node in the cluster's MST.
               - The root node index of the MST.
    """
    # A dictionary to map global index to local position for easier management
    local_index_map = {node: i for i, node in enumerate(cluster_nodes)}

    num_cluster_nodes = len(cluster_nodes)
    start_local_index = local_index_map[start_node_index]

    # d[i] represents the minimum distance to a node from the MST
    d = np.full(num_cluster_nodes, float('inf'))
    # p[i] represents the parent node
    p = np.full(num_cluster_nodes, -1, dtype=int)

    d[start_local_index] = 0
    Q = set(range(num_cluster_nodes))

    while Q:
        u_local_index = min(Q, key=lambda i: d[i])
        u_global_index = cluster_nodes[u_local_index]

        if d[u_local_index] == float('inf'):
            print(f"  Warning: Sub-graph disconnected from start node. Cannot build full MST.")
            break

        Q.remove(u_local_index)

        for v_local_index in Q:
            v_global_index = cluster_nodes[v_local_index]
            weight = distance_matrix[u_global_index, v_global_index]

            if weight < d[v_local_index]:
                d[v_local_index] = weight
                p[v_local_index] = u_local_index

    mst_edges = []
    root_node_index = start_node_index

    for i in range(num_cluster_nodes):
        if i == start_local_index:
            continue

        child_global_index = cluster_nodes[i]
        parent_local_index = p[i]

        if parent_local_index != -1:
            parent_global_index = cluster_nodes[parent_local_index]
            edge_weight = distance_matrix[parent_global_index, child_global_index]
            mst_edges.append((parent_global_index, child_global_index, edge_weight))

    return mst_edges, root_node_index


# Implementation of Algorithm 3
def construct_comprehensive_mst(clusters, distance_matrix):
    """
    Constructs a comprehensive MST (MSTcom) for the entire network.
    This function implements the logic of Algorithm 3.
    """
    comprehensive_mst = []
    root_nodes = []
    cluster_mst_edges = []

    print("\n--- Phase 3: MST Construction (Parallel Logic) ---")

    for i, cluster in enumerate(clusters):
        start_node_index = cluster[0]

        mst_edges, root_node_index = compute_mst_for_cluster(cluster, distance_matrix, start_node_index)

        root_nodes.append(root_node_index)
        cluster_mst_edges.append(mst_edges)

        print(f"  Cluster {i + 1} MST Edges (Root: {root_node_index}):")
        for edge in mst_edges:
            print(f"    Edge: {edge[0]} -> {edge[1]}, Weight: {edge[2]}")

    print("\n--- Connecting MST Root Nodes ---")
    if len(root_nodes) > 1:
        # Step 1: Find the minimum weight edge between each pair of clusters
        root_connection_matrix = np.full((len(clusters), len(clusters)), float('inf'))
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                cluster_a_nodes = clusters[i]
                cluster_b_nodes = clusters[j]

                min_weight = float('inf')
                for node_a in cluster_a_nodes:
                    for node_b in cluster_b_nodes:
                        weight = distance_matrix[node_a, node_b]
                        if weight < min_weight:
                            min_weight = weight

                if min_weight != float('inf'):
                    root_connection_matrix[i, j] = min_weight
                    root_connection_matrix[j, i] = min_weight

        # Step 2: Build a pseudo-cluster of root nodes to find the MST of their connections
        # The indices of the new cluster are just 0, 1, ..., M-1
        pseudo_root_cluster = list(range(len(clusters)))

        # We need to map the pseudo-root indices back to the actual root node indices
        # Let's use the first root node as the start for this inter-cluster MST
        start_pseudo_root_index = 0

        mst_root_edges, _ = compute_mst_for_cluster(pseudo_root_cluster, root_connection_matrix,
                                                    start_pseudo_root_index)

        # Step 3: Add these inter-cluster edges to the comprehensive MST
        print(f"  Inter-cluster connecting edges:")
        for edge in mst_root_edges:
            source_root_id = root_nodes[edge[0]]
            target_root_id = root_nodes[edge[1]]
            weight = edge[2]
            comprehensive_mst.append((source_root_id, target_root_id, weight))
            print(f"    Edge: {source_root_id} -> {target_root_id}, Weight: {weight}")
    else:
        print("  Only one cluster, no inter-cluster connections needed.")

    # Add intra-cluster MST edges to the comprehensive list
    for mst_edges in cluster_mst_edges:
        comprehensive_mst.extend(mst_edges)

    return comprehensive_mst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform Agglomerative Clustering on a network graph from a JSON file.")
    parser.add_argument("--json_file_path", type=str, help="Path to the JSON file containing the graph topology.",
                        required=True)
    parser.add_argument("--num_clusters", type=int, help="Desired number of clusters (M).", required=True)
    args = parser.parse_args()

    # json_file_path = args.json_file_path
    json_file_path = os.path.join("topology", args.json_file_path)
    print(f"Loading network from : {json_file_path}")
    M = args.num_clusters
    print(f"Total cluster(s), M : {M}")

    if not os.path.exists(json_file_path):
        print(f"Error: JSON file not found at '{json_file_path}'")
        exit(1)

    if M <= 0:
        print("Error: The desired number of clusters (M) must be a positive integer.")
        exit(1)

    try:
        num_nodes, distance_matrix, id_to_index = load_graph_from_json(json_file_path)

        if M > num_nodes:
            print(
                f"Warning: Desired number of clusters (M={M}) is greater than the total number of nodes ({num_nodes}).")
            M = num_nodes

        print("\n--- Loaded Distance Matrix ---")
        np.set_printoptions(linewidth=np.inf, precision=2, suppress=True,
                            formatter={'float_kind': lambda x: "inf" if np.isinf(x) else f"{x:.0f}"})
        print(distance_matrix)
        print("\n")

        # Phase 1: Agglomerative Clustering
        clustered_result = agglomerative_clustering(num_nodes, M, distance_matrix)
        index_to_id = {v: k for k, v in id_to_index.items()}
        final_clusters_with_ids = [[index_to_id[node_index] for node_index in cluster] for cluster in clustered_result]

        # Phase 2: Leader Selection and Announcement
        cluster_leaders = select_cluster_leaders(clustered_result, index_to_id)

        print(f"--- BNSF Process Completed ---")
        print(f"\n--- Final Clusters (M = {M}) ---")
        for i, cluster in enumerate(final_clusters_with_ids):
            print(f"Cluster {i + 1}: {cluster}")

        print(f"\n--- Leader Selection & Announcement ---")
        for cluster_name, leader_id in cluster_leaders.items():
            print(f"{cluster_name} Leader: {leader_id}")

        # Phase 3: MST Construction
        comprehensive_mst = construct_comprehensive_mst(clustered_result, distance_matrix)

        print(f"\n--- Comprehensive MST for the Entire Network ---")
        for i, edge in enumerate(comprehensive_mst):
            source_id = index_to_id[edge[0]]
            target_id = index_to_id[edge[1]]
            weight = edge[2]
            print(f"  Edge {i + 1}: {source_id} -> {target_id}, Weight: {weight}")

    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{json_file_path}'. Please ensure it's a valid JSON file.")
        exit(1)
    except KeyError as e:
        print(f"Error: Missing expected key in JSON file: {e}. Please check the JSON structure.")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)