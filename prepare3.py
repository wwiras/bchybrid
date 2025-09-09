from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import subprocess
import sys
import os
import time
# import sqlite3  # Add sqlite3 import for direct schema reference



def get_pod_topology(topology_folder, filename):
    """
    Function : It will read the topology (from a given json file - network topology)
    Input: Topology folder name and filename
    Returns: topology object - if found. False, if not found
    """
    # 1. Load topology JSON
    # It's important to use os.path.join for platform compatibility
    topology_file_path = os.path.join(os.getcwd(), topology_folder, filename)

    if not os.path.exists(topology_file_path):
        print(f"Error: Topology file not found at '{topology_file_path}'. Exiting.", flush=True)
        sys.exit(1)

    try:
        with open(topology_file_path) as f:
            topology = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file '{topology_file_path}'. Exiting.", flush=True)
        topology = False

    return topology


def get_pod_neighbors(topology):
    """
    Creates a dictionary mapping each node to its neighbors.

    Args:
        topology: The topology dictionary containing 'nodes' and 'edges'

    Returns:
        Dictionary {node_id: [neighbor1, neighbor2, ...]}
    """
    neighbor_map = {node['id']: [] for node in topology['nodes']}

    for edge in topology['edges']:
        source = edge['source']
        target = edge['target']

        # Add bidirectional connections for undirected graphs
        neighbor_map[source].append(target)
        if not topology['directed']:  # Assuming 'directed' key exists and is boolean
            neighbor_map[target].append(source)

    return neighbor_map


def get_pod_dplymt():
    """
    Fetches [(index, pod_name, pod_ip)] from Kubernetes or returns False on failure.

    Returns:
        - List of (index, pod_name, pod_ip) tuples on success
        - False on any failure
    """
    cmd = [
        'kubectl',
        'get', 'pods',
        '-l', 'app=bcgossip',
        '-o', 'jsonpath={range .items[*]}{.metadata.name}{" "}{.status.podIP}{"\\n"}{end}'
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True,
            timeout=10
        )

        if not result.stdout.strip():
            print("Error: No pods found with label app=bcgossip")
            return False

        pods_data = [line.split() for line in result.stdout.splitlines() if line]
        pods_data.sort(key=lambda x: x[0])  # Sort by pod name to ensure consistent indexing

        # Add index to each pod entry
        return [(i, name, ip) for i, (name, ip) in enumerate(pods_data)]

    except subprocess.CalledProcessError as e:
        print(f"kubectl failed (exit {e.returncode}): {e.stderr.strip()}")
        return False
    except subprocess.TimeoutExpired:
        print("Error: kubectl command timed out after 10 seconds")
        return False
    except Exception as e:
        print(f"Unexpected error in get_pod_dplymt: {str(e)}")
        return False


def get_pod_mapping(pod_deployment, pod_neighbors, pod_topology):
    """
    Creates {deployment_pod_name: [(neighbor_ip, weight), ...]} mapping
    by incorporating edge weights from the topology.

    Args:
        pod_deployment: List of (index, pod_name, pod_ip) tuples.
        pod_neighbors: Dict {'gossip-0': ['gossip-1', ...], ...}.
        pod_topology: The raw topology dictionary containing 'edges' with 'weight'.

    Returns:
        Dict {deployment_pod_name: [('ip1', weight1), ('ip2', weight2), ...]}
    """
    gossip_id_to_ip = {f'gossip-{index}': ip for index, _, ip in pod_deployment}

    # Create a quick lookup for edge weights (node1_id, node2_id) -> weight
    # Ensure canonical form (smaller_id, larger_id) to handle undirected edges consistently
    edge_weights_lookup = {}
    for edge in pod_topology['edges']:
        source_id = edge['source']
        target_id = edge['target']
        weight = edge['weight']

        # Ensure consistent key regardless of source/target order for undirected graph
        if source_id < target_id:
            edge_weights_lookup[(source_id, target_id)] = weight
        else:
            edge_weights_lookup[(target_id, source_id)] = weight

    result = {}

    for index, deployment_name, _ in pod_deployment:
        gossip_id = f'gossip-{index}'

        if gossip_id in pod_neighbors:
            neighbor_list_with_weights = []
            for neighbor_gossip_id in pod_neighbors[gossip_id]:
                if neighbor_gossip_id in gossip_id_to_ip:
                    neighbor_ip = gossip_id_to_ip[neighbor_gossip_id]

                    # Look up the weight for this specific edge
                    # Use canonical form (smaller ID, larger ID) to find the weight
                    if gossip_id < neighbor_gossip_id:
                        edge_key = (gossip_id, neighbor_gossip_id)
                    else:
                        edge_key = (neighbor_gossip_id, gossip_id)

                    # Default weight to 0 or 1 if somehow not found, though it should be
                    weight = edge_weights_lookup.get(edge_key,
                                                     0)  # Use 0 as a default if edge somehow not in topology edges

                    neighbor_list_with_weights.append((neighbor_ip, weight))
            result[deployment_name] = neighbor_list_with_weights

    return result


def get_num_nodes(namespace='default'):
    """
    Dynamically determines the number of nodes (pods) by counting running pods.
    """
    get_pods_cmd = f"kubectl get pods -n {namespace} --no-headers | grep Running | wc -l"
    try:
        result = subprocess.run(get_pods_cmd, shell=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        num_nodes = int(result.stdout.strip())
        return num_nodes
    except subprocess.CalledProcessError as e:
        print(f"Error getting number of pods: {e.stderr}", flush=True)
        return False


def update_pod_neighbors(pod, neighbors, timeout=300):
    """
    Atomically updates neighbor list (IP and weight) in a pod's SQLite DB.
    Returns (success: bool, output: str) tuple in ALL cases.
    """
    try:
        # 1. Convert neighbors (list of (ip, weight) tuples) to JSON-safe format
        # Each tuple (ip, weight) is directly loaded as a list in sqlite3 (e.g. ['10.1.0.4', 84])
        # values = [(ip, weight) for ip, weight in neighbors]
        # Using json.dumps on the list of tuples for the Python script
        neighbors_json = json.dumps(neighbors)

        # 2. Create properly escaped Python command
        python_script = f"""
import sqlite3
import json

try:
    # Deserialize the JSON string back to a list of lists/tuples
    values_from_json = json.loads('{neighbors_json.replace("'", "\\'")}')
    # Convert to list of tuples for executemany if needed, but json.loads will yield list of lists
    # which works with sqlite3 executemany

    with sqlite3.connect('ned.db') as conn:
        conn.execute('BEGIN TRANSACTION')
        # Update schema to include weight column
        conn.execute('DROP TABLE IF EXISTS NEIGHBORS')
        conn.execute('CREATE TABLE NEIGHBORS (pod_ip TEXT PRIMARY KEY, weight REAL)')
        # Insert both IP and weight
        conn.executemany('INSERT INTO NEIGHBORS VALUES (?, ?)', values_from_json)
        conn.commit()
    print(f"Updated {{len(values_from_json)}} neighbors with IP and Weight")
except Exception as e:
    print(f"Error: {{str(e)}}")
    raise
"""

        # 3. Execute via kubectl with proper quoting
        cmd = [
            'kubectl', 'exec', pod,
            '--', 'python3', '-c', python_script
        ]

        result = subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=timeout)
        return True, result.stdout.strip()

    except subprocess.CalledProcessError as e:
        return False, f"Command failed (exit {e.returncode}): {e.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, f"Command timed out after {timeout} seconds"
    except Exception as e:
        return False, f"Unexpected error in update_pod_neighbors: {str(e)}"


def update_all_pods(pod_mapping, max_retries=3, initial_timeout=300, max_concurrent_updates=10):
    """
    Update neighbors for all pods with extended timeout and retry capabilities,
    now with parallel execution.

    Args:
        pod_mapping: Dictionary of pod to neighbors mapping
        max_retries: Number of retry attempts for failed updates
        initial_timeout: Initial timeout in seconds (will increase with retries)
        max_concurrent_updates: Max number of kubectl exec calls to run in parallel
    """
    pod_list = list(pod_mapping.keys())
    total_pods = len(pod_list)
    success_count = 0
    failure_count = 0
    start_time = time.time()
    retry_queue = []  # Stores (pod, neighbors, retry_count, future_obj)

    print(
        f"\nStarting update for {total_pods} pods (timeout: {initial_timeout}s, max retries: {max_retries}, concurrent: {max_concurrent_updates})...")

    with ThreadPoolExecutor(max_workers=max_concurrent_updates) as executor:
        futures_to_pod = {
            executor.submit(update_pod_neighbors, pod, pod_mapping.get(pod, []), initial_timeout): pod
            for pod in pod_list
        }

        # Track results of initial attempts
        for future in as_completed(futures_to_pod):
            pod = futures_to_pod[future]
            try:
                success, output = future.result()
                if success:
                    success_count += 1
                else:
                    retry_queue.append((pod, pod_mapping.get(pod, []), 1))
                    failure_count += 1
                    print(f"\nInitial attempt failed for {pod}: {output}")
            except Exception as exc:
                retry_queue.append((pod, pod_mapping.get(pod, []), 1))
                failure_count += 1
                print(f"\nInitial attempt for {pod} generated an exception: {exc}")

            # Progress reporting (updated to reflect completion of futures)
            elapsed = time.time() - start_time
            progress = (success_count + failure_count) / total_pods * 100  # Corrected progress calc
            print(
                f"\rProgress: {progress:.1f}% | "
                f"Elapsed: {elapsed:.1f}s | "
                f"Success: {success_count}/{total_pods} | "
                f"Failed: {failure_count} | "
                f"Retries pending: {len(retry_queue)}",
                end='', flush=True
            )

        # Retry logic
        while retry_queue and (time.time() - start_time) < 3600:
            pod_to_retry, neighbors_to_retry, retry_count = retry_queue.pop(0)

            if retry_count > max_retries:
                print(f"\nMax retries exceeded for {pod_to_retry}")
                continue  # Skip if max retries reached

            timeout = initial_timeout * (retry_count + 1)
            print(f"\nRetry #{retry_count} for {pod_to_retry} (timeout: {timeout}s)...", flush=True)

            retry_future = executor.submit(update_pod_neighbors, pod_to_retry, neighbors_to_retry, timeout)

            try:
                success, output = retry_future.result()  # Wait for this specific retry
                if success:
                    success_count += 1
                    failure_count -= 1  # Decrement failure count as it's now a success
                else:
                    retry_queue.append((pod_to_retry, neighbors_to_retry, retry_count + 1))
                    print(f"Retry failed for {pod_to_retry}: {output}")
            except Exception as exc:
                retry_queue.append((pod_to_retry, neighbors_to_retry, retry_count + 1))
                print(f"Retry for {pod_to_retry} generated an exception: {exc}")

            # Update progress after each retry completes
            elapsed = time.time() - start_time
            progress = (success_count + failure_count) / total_pods * 100  # Corrected progress calc
            print(
                f"\rProgress: {progress:.1f}% | "
                f"Elapsed: {elapsed:.1f}s | "
                f"Success: {success_count}/{total_pods} | "
                f"Failed: {failure_count} | "
                f"Retries pending: {len(retry_queue)}",
                end='', flush=True
            )

    # Final summary (unchanged)
    total_time = time.time() - start_time
    print(f"\n\nUpdate completed in {total_time:.1f} seconds")
    print(f"Summary - Total: {total_pods} | Success: {success_count} | Failed: {failure_count}")

    return success_count == total_pods

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get pod mapping and neighbor info based on topology.")
    parser.add_argument("--filename", help="Name of the topology JSON file in the 'topology' folder.")
    parser.add_argument("--topology_folder", default="topology", help="Name of the topology folder from the root.")
    args = parser.parse_args()

    # prepare flag
    prepare = False

    # 1. Get topology from json
    pod_topology = get_pod_topology(args.topology_folder, args.filename)

    if pod_topology:
        # 2. Make sure topology nodes are the same as deployment nodes
        nodes_dplymt = get_num_nodes()

        nodes_topology = len(pod_topology['nodes'])

        if nodes_topology == nodes_dplymt and nodes_topology > 0:  # Ensure nodes_topology is not zero
            print(f"Deployment number of nodes equal to topology nodes: {nodes_topology}")

            # 3. Get pod topology neighbors
            pod_neighbors = get_pod_neighbors(pod_topology)

            # 4. Get pods info from deployment
            pod_dplymt = get_pod_dplymt()

            # 5. Get pod mapping with tuples (IP, Weight)
            if pod_dplymt:
                pod_mapping = get_pod_mapping(pod_dplymt, pod_neighbors, pod_topology)  # Pass pod_topology

                if pod_mapping:
                    update_all_pods(pod_mapping)
                    prepare = True
                else:
                    print("Error: Could not create pod mapping.", flush=True)
            else:
                print("Error: Could not retrieve pod deployment information.", flush=True)
        elif nodes_topology == 0:
            print("Error: Topology file contains no nodes.", flush=True)
            sys.exit(1)
        else:
            print(
                f"Error: Deployment number of nodes ({nodes_dplymt}) and topology nodes ({nodes_topology}) must be equal.",
                flush=True)
            sys.exit(1)

    if prepare:
        print("Platform is now ready for testing..!")
    else:
        print("Platform could not be ready due to errors.")