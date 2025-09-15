import argparse
import json
import subprocess
import sys
import traceback
import time
import uuid
import select
import random
from datetime import datetime, timedelta, timezone

class Test:
    def __init__(self, num_test, rmax):
        self.num_tests = num_test
        self.rmax = rmax
        self.num_nodes = 0
        self.gossip_delay = 5.0

    def run_command(self, command, suppress_output=False):
        """
        A more robust way to run a command without simulating an interactive shell.
        """
        try:
            if suppress_output:
                result = subprocess.run(command, check=True, text=True, capture_output=True)
            else:
                result = subprocess.run(command, check=True, text=True, stdout=sys.stdout, stderr=sys.stderr)

            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)

            return result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while executing the command.", flush=True)
            print(f"Error message: {e.stderr}", flush=True)
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while executing the command.", flush=True)
            traceback.print_exc()
            sys.exit(1)

    def wait_for_pods_to_be_ready(self, namespace='default', expected_pods=0, timeout=1000):
        print(f"Checking for pods in namespace {namespace}...", flush=True)
        start_time = time.time()
        get_pods_cmd = f"kubectl get pods -n {namespace} --no-headers | grep Running | wc -l"

        while time.time() - start_time < timeout:
            try:
                result = subprocess.run(get_pods_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                running_pods = int(result.stdout.strip())
                if running_pods >= expected_pods:
                    print(f"All {expected_pods} pods are up and running in namespace {namespace}.", flush=True)
                    return True
                else:
                    print(f" {running_pods} pods are up for now in namespace {namespace}. Waiting...", flush=True)
            except subprocess.CalledProcessError as e:
                print(f"Error checking for pods: {e.stderr}", flush=True)
                return False
            time.sleep(1)
        print(f"Timeout waiting for pods to terminate in namespace {namespace}.", flush=True)
        return False

    def get_num_nodes(self, namespace='default'):
        get_pods_cmd = f"kubectl get pods -n {namespace} --no-headers | grep Running | wc -l"
        try:
            result = subprocess.run(get_pods_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            num_nodes = int(result.stdout.strip())
            print(f"Number of running pods (num_nodes): {num_nodes}", flush=True)
            return num_nodes
        except subprocess.CalledProcessError as e:
            print(f"Error getting number of pods: {e.stderr}", flush=True)
            return False

    def select_random_pod(self):
        command = ["kubectl", "get", "pods", "--no-headers", "-o", "jsonpath={.items[*].metadata.name}"]
        stdout, _ = self.run_command(command, suppress_output=True)
        pod_list = [pod for pod in stdout.split() if "running" in subprocess.run(["kubectl", "get", "pod", pod, "-o", "jsonpath={.status.phase}"], check=True, text=True, capture_output=True).stdout.lower()]
        if not pod_list:
            raise Exception("No running pods found.")
        return random.choice(pod_list)

    def _get_malaysian_time(self):
        utc_time = datetime.now(timezone.utc)
        malaysia_offset = timedelta(hours=8)
        malaysia_time = utc_time + malaysia_offset
        return malaysia_time

    def access_pod_and_initiate_gossip(self, pod_name, replicas, unique_id, iteration, rmax):
        time.sleep(self.gossip_delay)

        try:
            start_time_log = self._get_malaysian_time().strftime('%Y/%m/%d %H:%M:%S')
            message = f'{unique_id}-cubaan{replicas}-{iteration}'
            start_log = {
                'event': 'gossip_start',
                'pod_name': pod_name,
                'message': message,
                'start_time': start_time_log,
                'details': f"Gossip propagation started for message: {message} and Rmax: {rmax}"
            }
            print(json.dumps(start_log), flush=True)

            # A more robust command that doesn't simulate an interactive shell
            command = [
                'kubectl', 'exec', pod_name,
                '--', 'python3', 'start.py', '--message', message, '--rmax', str(rmax)
            ]
            
            # Use subprocess.run to execute the command and get its output
            result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=300)
            
            end_time_log = self._get_malaysian_time().strftime('%Y/%m/%d %H:%M:%S')
            end_log = {
                'event': 'gossip_end',
                'pod_name': pod_name,
                'message': message,
                'end_time': end_time_log,
                'details': f"Gossip propagation completed for message: {message}"
            }
            print(json.dumps(end_log), flush=True)
            print(result.stdout, flush=True)
            return True

        except subprocess.CalledProcessError as e:
            error_log = {
                'event': 'gossip_error',
                'pod_name': pod_name,
                'message': message,
                'error': f"Command failed with exit code {e.returncode}",
                'details': f"Error: {e.stderr}"
            }
            print(json.dumps(error_log), flush=True)
            return False
        except subprocess.TimeoutExpired:
            error_log = {
                'event': 'gossip_error',
                'pod_name': pod_name,
                'message': message,
                'error': "Command timed out",
                'details': "kubectl exec command timed out."
            }
            print(json.dumps(error_log), flush=True)
            return False
        except Exception as e:
            error_log = {
                'event': 'gossip_error',
                'pod_name': pod_name,
                'message': message,
                'error': str(e),
                'details': f"Unexpected error: {str(e)}"
            }
            print(json.dumps(error_log), flush=True)
            return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Usage: python automate.py --num_tests <number_of_tests> --rmax <rmax_value>")
    parser.add_argument('--num_tests', required=True, type=int, help="Total number of tests to do")
    parser.add_argument('--rmax', required=True, type=int, help="Rmax value for the hybrid protocol.")
    args = parser.parse_args()

    if args.num_tests > 0:
        test = Test(int(args.num_tests), int(args.rmax))
    else:
        print("Error: --num_tests must be a valid integer greater than 0.", flush=True)
        sys.exit(1)

    test.num_nodes = test.get_num_nodes()
    if test.num_nodes == 0:
        print("Error: total number of nodes cannot be determined or kubernetes is not ready..", flush=True)
        sys.exit(1)

    if test.wait_for_pods_to_be_ready(namespace='default', expected_pods=int(test.num_nodes), timeout=1000):
        unique_id = str(uuid.uuid4())[:4]

        pod_name = test.select_random_pod()
        for nt in range(1, test.num_tests + 1):
            print(f"Selected pod: {pod_name}", flush=True)
            if test.access_pod_and_initiate_gossip(pod_name, int(test.num_nodes), unique_id, nt, test.rmax):
                print(f"Test {nt} complete.", flush=True)
            else:
                print(f"Test {nt} failed.", flush=True)
    else:
        print(f"Pods not ready.", flush=True)

