import asyncio  # Import asyncio for asynchronous programming
import grpc.aio  # Import the async version of gRPC
import os
import socket
from concurrent import futures  # Still used for the gRPC server's thread pool, but not for outbound gossip
import gossip_pb2
import gossip_pb2_grpc
import json
import time
import sqlite3
from google.protobuf.empty_pb2 import Empty  # <-- Add this import

# Define a constant for max concurrent outbound gRPC calls
# This now controls the concurrency of asyncio tasks, if you were to limit them
# with a semaphore, but for simple scatter/gather, asyncio handles it.
# The internal gRPC server still uses a ThreadPoolExecutor.


class Node(gossip_pb2_grpc.GossipServiceServicer):

    def __init__(self, service_name):
        self.hostname = socket.gethostname()
        self.host = socket.gethostbyname(self.hostname)
        self.port = '5050'
        self.service_name = service_name
        self.app_name = 'bcgossip'
        self.susceptible_nodes = []
        self.received_message_ids = set()

        # Initialize neighbors on startup - this remains synchronous as DB access is typically synchronous
        self.get_neighbors()

        # self.executor is removed as asyncio manages outbound concurrency

    def get_neighbors(self):
        """
        Fetches neighbor IPs and their associated weights from ned.db.
        This remains a synchronous operation as sqlite3 is not inherently async.
        """
        try:
            conn = sqlite3.connect('ned.db')
            cursor = conn.execute("SELECT pod_ip, weight from NEIGHBORS")

            self.susceptible_nodes = []  # Clear the existing list to refresh it

            for row in cursor:
                self.susceptible_nodes.append((row[0], row[1]))  # Append as (IP, weight) tuple
            conn.close()
            # Add a log message to confirm the state refresh
            print(f"Neighbors list refreshed from ned.db. Found {len(self.susceptible_nodes)} neighbors.", flush=True)

        except sqlite3.Error as e:
            print(f"SQLite error in get_neighbors: {e}", flush=True)
        except Exception as e:
            print(f"Unexpected error in get_neighbors: {str(e)}", flush=True)

    # --- New method to handle the UpdateNeighbors RPC ---
    async def UpdateNeighbors(self, request, context):
        """
        Receives an RPC call to update the neighbor list.
        The corresponding method on its Node service is executed, which in turn calls self.get_neighbors().
        """
        print("Received UpdateNeighbors signal. Refreshing state...", flush=True)

        # In-Memory State Refresh: Re-read the database and update the neighbor list
        self.get_neighbors()

        # As per the logic, we should also clear the message cache for a new topology run
        self.received_message_ids.clear()
        print(f"Message cache cleared. New topology active.", flush=True)

        return gossip_pb2.Acknowledgment(details="Neighbors list and message cache have been updated.")

    async def SendMessage(self, request, context):
        """
        Receiving message from other nodes and distributing it.
        """
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

            await self.gossip_message(message, sender_id)
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

            await self.gossip_message(message, sender_id)

            return gossip_pb2.Acknowledgment(details=f"{self.host} received: '{message}'")

        # Updated _send_gossip_to_peer function for validation
    async def _send_gossip_to_peer(self, message, sender_id, peer_ip, peer_weight):
        """Helper function to send a single gossip message to a peer, simulating latency."""
        send_timestamp = time.time_ns()

        try:
            # ---- VALIDATION LOGGING START ----
            # Log the exact time before the sleep
            sleep_start_time = time.perf_counter_ns()
            # --- END VALIDATION LOGGING ---

            # Introduce latency here: use asyncio.sleep for non-blocking delay
            await asyncio.sleep(int(peer_weight) / 1000)

            # ---- VALIDATION LOGGING START ----
            sleep_end_time = time.perf_counter_ns()
            simulated_delay = (sleep_end_time - sleep_start_time) / 1e6
            # Log the actual sleep time vs. the expected sleep time
            print(
                f"DEBUG: Simulating delay for {peer_ip}. Expected: {peer_weight:.2f}ms, Actual: {simulated_delay:.2f}ms",
                flush=True)
            # --- END VALIDATION LOGGING ---

            async with grpc.aio.insecure_channel(f"{peer_ip}:5050") as channel:
                stub = gossip_pb2_grpc.GossipServiceStub(channel)
                await stub.SendMessage(gossip_pb2.GossipMessage(
                    message=message,
                    sender_id=self.host,
                    timestamp=send_timestamp,
                    latency_ms=peer_weight
                ))
        except grpc.aio.AioRpcError as e:
            print(f"Failed to send message: '{message}' to {peer_ip}: RPC Error Code {e.code()} - {e.details()}",
                  flush=True)
        except Exception as e:
            print(f"Unexpected error when sending message to {peer_ip}: {str(e)}", flush=True)

    async def gossip_message(self, message, sender_id):
        """
        Propagates the message to susceptible (neighboring) nodes using asyncio tasks.
        """
        if not self.susceptible_nodes:
            self.get_neighbors()

        tasks = []
        for peer_ip, peer_weight in self.susceptible_nodes:
            if peer_ip != sender_id:
                task = asyncio.create_task(self._send_gossip_to_peer(message, sender_id, peer_ip, peer_weight))
                tasks.append(task)

        await asyncio.gather(*tasks,
                             return_exceptions=True)

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

    async def start_server(self):
        """ Initiating asynchronous gRPC server """
        server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        gossip_pb2_grpc.add_GossipServiceServicer_to_server(self, server)
        server.add_insecure_port(f'[::]:{self.port}')
        print(f"{self.hostname}({self.host}) listening on port {self.port}", flush=True)

        await server.start()
        await server.wait_for_termination()


async def run_server_async():
    service_name = os.getenv('SERVICE_NAME', 'bcgossip-svc')
    node = Node(service_name)
    await node.start_server()


if __name__ == '__main__':
    asyncio.run(run_server_async())