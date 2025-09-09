import grpc
import os
import socket
from concurrent import futures
import gossip_pb2
import gossip_pb2_grpc
import json
import time
import sqlite3

# Define a constant for max concurrent outbound gRPC calls
MAX_OUTBOUND_GOSSIP_WORKERS = 10


class Node(gossip_pb2_grpc.GossipServiceServicer):

    def __init__(self, service_name):
        self.hostname = socket.gethostname()
        self.host = socket.gethostbyname(self.hostname)
        self.port = '5050'
        self.service_name = service_name
        self.app_name = 'bcgossip'
        self.susceptible_nodes = []
        self.received_message_ids = set()

        # self.get_neighbors()  # Initialize neighbors on startup
        # do not use it yet. this is because when the pods are up,
        # ned.db is not exist yet

        self.executor = futures.ThreadPoolExecutor(max_workers=MAX_OUTBOUND_GOSSIP_WORKERS)

    def get_neighbors(self):
        try:
            conn = sqlite3.connect('ned.db')
            cursor = conn.execute("SELECT pod_ip, weight from NEIGHBORS")

            self.susceptible_nodes = []

            for row in cursor:
                self.susceptible_nodes.append((row[0], row[1]))
            conn.close()
        except sqlite3.Error as e:
            print(f"SQLite error in get_neighbors: {e}", flush=True)
        except Exception as e:
            print(f"Unexpected error in get_neighbors: {str(e)}", flush=True)

    def SendMessage(self, request, context):
        message = request.message
        sender_id = request.sender_id
        received_timestamp = time.time_ns()
        incoming_link_latency = request.latency_ms

        if sender_id == self.host:
            self.received_message_ids.add(message)
            log_message = (f"Gossip initiated by {self.hostname} ({self.host}) at "
                           f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(received_timestamp / 1e9))}")
            self._log_event(message, sender_id, received_timestamp, None,
                            None, 'initiate', log_message)
            self.gossip_message(message, sender_id)
            return gossip_pb2.Acknowledgment(details=f"Done propagate! {self.host} received: '{message}'")

        elif message in self.received_message_ids:
            log_message = f"{self.host} ignoring duplicate message: {message} from {sender_id}"
            self._log_event(message, sender_id, received_timestamp, None,
                            incoming_link_latency, 'duplicate', log_message)
            return gossip_pb2.Acknowledgment(details=f"Duplicate message ignored by ({self.host})")
        else:
            self.received_message_ids.add(message)
            propagation_time = (received_timestamp - request.timestamp) / 1e6
            log_message = (f"({self.hostname}({self.host}) received: '{message}' from {sender_id}"
                           f" in {propagation_time:.2f} ms. Incoming link latency: {incoming_link_latency:.2f} ms")
            self._log_event(message, sender_id, received_timestamp, propagation_time,
                            incoming_link_latency, 'received', log_message)

            self.gossip_message(message, sender_id)

            return gossip_pb2.Acknowledgment(details=f"{self.host} received: '{message}'")

    def _send_gossip_to_peer(self, message, sender_id, peer_ip, peer_weight):
        """Helper function to send a single gossip message to a peer, simulating latency."""
        send_timestamp = time.time_ns()
        try:
            # Introduce latency here: pause execution for the peer_weight duration
            # peer_weight is assumed to be in milliseconds, time.sleep expects seconds
            time.sleep(int(peer_weight) / 1000)  # This line simulates the network latency

            with grpc.insecure_channel(f"{peer_ip}:5050") as channel:
                stub = gossip_pb2_grpc.GossipServiceStub(channel)
                stub.SendMessage(gossip_pb2.GossipMessage(
                    message=message,
                    sender_id=self.host,
                    timestamp=send_timestamp,
                    latency_ms=peer_weight  # This value is passed in the message for the receiver to log
                ))
        except grpc.RpcError as e:
            print(f"Failed to send message: '{message}' to {peer_ip}: RPC Error Code {e.code()} - {e.details()}",
                  flush=True)
        except Exception as e:
            print(f"Unexpected error when sending message to {peer_ip}: {str(e)}", flush=True)

    def gossip_message(self, message, sender_ip):
        """
        Propagates the message to susceptible (neighboring) nodes using a thread pool.
        """
        if not self.susceptible_nodes:
            self.get_neighbors()
            # For topology debugging or info
            print(f"self.susceptible_nodes: {self.susceptible_nodes}", flush=True)

        futures_list = []
        for peer_ip, peer_weight in self.susceptible_nodes:
            if peer_ip != sender_ip:
                future = self.executor.submit(self._send_gossip_to_peer, message, sender_ip, peer_ip, peer_weight)
                futures_list.append(future)

        for future in futures_list:
            try:
                future.result()
            except Exception:
                pass

    def _log_event(self, message, sender_id, received_timestamp, propagation_time, incoming_link_latency, event_type,
                   log_message):
        """Logs the gossip event as structured JSON data."""
        event_data = {
            'message': message,
            'sender_id': sender_id,
            'receiver_id': self.host,
            'received_timestamp': received_timestamp,
            'propagation_time': propagation_time,
            'incoming_link_latency': incoming_link_latency,
            'event_type': event_type,
            'detail': log_message
        }

        print(json.dumps(event_data), flush=True)

    def start_server(self):
        """ Initiating gRPC server """
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        gossip_pb2_grpc.add_GossipServiceServicer_to_server(self, server)
        server.add_insecure_port(f'[::]:{self.port}')
        print(f"{self.hostname}({self.host}) listening on port {self.port}", flush=True)
        server.start()
        server.wait_for_termination()


def run_server():
    service_name = os.getenv('SERVICE_NAME', 'bcgossip-svc')
    node = Node(service_name)
    node.start_server()


if __name__ == '__main__':
    run_server()