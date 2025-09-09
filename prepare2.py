from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
import subprocess
import sys
import os
import time


# Note: The grpc-related imports are now removed from prepare.py
# as the signal is now sent via a kubectl exec script that runs inside the container.
# This avoids the ModuleNotFoundError on the host machine.


def get_pod_topology(topology_folder, filename):
    """
    Function : It will read the topology (from a given json file - network topology)
    Input: Topology folder name and filename
    Returns: topology object - if found. False, if not found
    """
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
    """
    neighbor_map = {node['id']: [] for node in topology['nodes']}

    for edge in topology['edges']:
        source = edge['source']
        target = edge['target']

        neighbor_map[source].append(target)
        if not topology['directed']:
            neighbor_map[target].append(source)

    return neighbor_map


def get_pod_dplymt():
    """
    Fetches [(index, pod_name, pod_ip)] from Kubernetes or returns False on failure.
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
        pods_data.sort(key=lambda x: x[0])

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
    """
    gossip_id_to_ip = {f'gossip-{index}': ip for index, _, ip in pod_deployment}
    edge_weights_lookup = {}
    for edge in pod_topology['edges']:
        source_id = edge['source']
        target_id = edge['target']
        weight = edge['weight']
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
                    if gossip_id < neighbor_gossip_id:
                        edge_key = (gossip_id, neighbor_gossip_id)
                    else:
                        edge_key = (neighbor_gossip_id, gossip_id)
                    weight = edge_weights_lookup.get(edge_key, 0)
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
        neighbors_json = json.dumps(neighbors)
        python_script = f"""
import sqlite3
import json
try:
    values_from_json = json.loads('{neighbors_json.replace("'", "\\'")}')
    with sqlite3.connect('ned.db') as conn:
        conn.execute('BEGIN TRANSACTION')
        conn.execute('DROP TABLE IF EXISTS NEIGHBORS')
        conn.execute('CREATE TABLE NEIGHBORS (pod_ip TEXT PRIMARY KEY, weight REAL)')
        conn.executemany('INSERT INTO NEIGHBORS VALUES (?, ?)', values_from_json)
        conn.commit()
    print(f"Updated {{len(values_from_json)}} neighbors with IP and Weight")
except Exception as e:
    print(f"Error: {{str(e)}}")
    raise
"""
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


def notify_pod_for_update_via_kubectl(pod_name):
    """
    Sends a signal to a pod via a kubectl exec command to trigger a neighbor list update.
    This method avoids the host's gRPC dependencies.
    """
    # The python script to run inside the container
    python_script = """
import grpc
import gossip_pb2_grpc
from google.protobuf.empty_pb2 import Empty
try:
    with grpc.insecure_channel('localhost:5050') as channel:
        stub = gossip_pb2_grpc.GossipServiceStub(channel)
        stub.UpdateNeighbors(Empty(), timeout=5)
    print("UpdateNeighbors signal sent successfully to localhost:5050.")
except grpc.RpcError as e:
    print(f"Error sending signal: RPC failed with code {e.code()}")
except Exception as e:
    print(f"Error sending signal: {str(e)}")
"""
    cmd = [
        'kubectl', 'exec', pod_name,
        '--', 'python3', '-c', python_script
    ]

    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True, timeout=10)
        return True, result.stdout.strip()
    except subprocess.CalledProcessError as e:
        return False, f"Command failed (exit {e.returncode}): {e.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return False, "Command timed out."
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def update_all_pods(pod_mapping, pod_dplymt, max_retries=3, initial_timeout=300, max_concurrent_updates=10):
    """
    Performs a two-phase update on all pods:
    1. Updates the SQLite DB in parallel for all pods.
    2. Sends a signal in parallel to all updated pods via kubectl exec.
    """
    pod_list = list(pod_mapping.keys())
    total_pods = len(pod_list)
    db_update_results = {}
    notify_update_results = {}
    start_time = time.time()

    print(f"\nStarting DB update for {total_pods} pods (concurrent: {max_concurrent_updates})...")
    with ThreadPoolExecutor(max_workers=max_concurrent_updates) as executor:
        db_futures = {
            executor.submit(update_pod_neighbors, pod, pod_mapping.get(pod, []), initial_timeout): pod
            for pod in pod_list
        }
        for future in as_completed(db_futures):
            pod_name = db_futures[future]
            try:
                db_update_results[pod_name] = future.result()
            except Exception as exc:
                db_update_results[pod_name] = (False, f"Exception during DB update: {exc}")

    db_success_count = sum(1 for success, _ in db_update_results.values() if success)
    db_failure_count = total_pods - db_success_count
    print(
        f"\nDB update phase complete in {time.time() - start_time:.1f} seconds. Success: {db_success_count}, Failed: {db_failure_count}")

    if db_success_count == 0:
        print("No successful DB updates. Skipping notification phase.")
        return False

    print(f"\nStarting notification for {db_success_count} pods...")
    with ThreadPoolExecutor(max_workers=max_concurrent_updates) as executor:
        notify_futures = {
            executor.submit(notify_pod_for_update_via_kubectl, pod_name): pod_name
            for pod_name, (success, _) in db_update_results.items() if success
        }
        for future in as_completed(notify_futures):
            pod_name = notify_futures[future]
            try:
                notify_update_results[pod_name] = future.result()
            except Exception as exc:
                notify_update_results[pod_name] = (False, f"Exception during notification: {exc}")

    notify_success_count = sum(1 for success, _ in notify_update_results.values() if success)
    notify_failure_count = db_success_count - notify_success_count
    total_time = time.time() - start_time
    print(f"\nNotification phase complete. Success: {notify_success_count}, Failed: {notify_failure_count}")
    print(f"\nTotal process completed in {total_time:.1f} seconds.")
    print(
        f"Overall Summary - Total Pods: {total_pods}, DB Update Success: {db_success_count}, Notification Success: {notify_success_count}")

    return notify_success_count == total_pods


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get pod mapping and neighbor info based on topology.")
    parser.add_argument("--filename", help="Name of the topology JSON file in the 'topology' folder.")
    parser.add_argument("--topology_folder", default="topology", help="Name of the topology folder from the root.")
    args = parser.parse_args()

    prepare = False

    pod_topology = get_pod_topology(args.topology_folder, args.filename)

    if pod_topology:
        nodes_dplymt = get_num_nodes()
        nodes_topology = len(pod_topology['nodes'])

        if nodes_topology == nodes_dplymt and nodes_topology > 0:
            print(f"Deployment number of nodes equal to topology nodes: {nodes_topology}")

            pod_neighbors = get_pod_neighbors(pod_topology)
            pod_dplymt = get_pod_dplymt()

            if pod_dplymt:
                pod_mapping = get_pod_mapping(pod_dplymt, pod_neighbors, pod_topology)

                if pod_mapping:
                    if update_all_pods(pod_mapping, pod_dplymt, max_concurrent_updates=20):
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