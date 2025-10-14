# ==============================================================================
# node_asyncio.py (SBA-Forwarding with Explicit Feedback)
# ==============================================================================

import asyncio
import grpc.aio
import os
import socket
from concurrent import futures
import gossip_pb2
import gossip_pb2_grpc
import json
import time
import sqlite3
import random 
from google.protobuf.empty_pb2 import Empty

# Define a constant for max concurrent outbound gRPC calls
MAX_CONCURRENT_SENDS = 125 


class Node(gossip_pb2_grpc.GossipServiceServicer):

    def __init__(self, service_name):
        self.hostname = socket.gethostname()
        self.host = socket.gethostbyname(self.hostname)
        self.port = '5050'
        self.service_name = service_name
        self.app_name = 'bcgossip'
        self.susceptible_nodes = []
        self.received_message_ids = set()
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_SENDS)

        # --- SBA-Forwarding State ---
        # Map: {(hop_count, neighbor_ip): Score} where Score in [0, 100]
        # This map holds the node's assessment of its *outbound* links.
        self.reputation_score = {} 
        self.MAX_SCORE = 100
        self.DECAY_RATE = 5      # Score increase (forgiveness) per message processing round
        self.PENALTY_VALUE = 20  # Score decrease on duplicate detection
        self.THRESHOLD_SCORE = 50 # Score below which a neighbor is pruned
        # self.FANOUT_K = 3        # Desired fanout for Random Neighbor Selection (RNS)
        self.FANOUT_K = MAX_CONCURRENT_SENDS  # MAX CONCURRENT as fanout - DNS Blockchain data for Random Neighbor Selection (RNS)
        # ----------------------------

        self.get_neighbors()

    def get_neighbors(self):
        try:
            conn = sqlite3.connect('ned.db')
            cursor = conn.execute("SELECT pod_ip, weight from NEIGHBORS")
            self.susceptible_nodes = []
            for row in cursor:
                # Stores as: (IP_Address, Latency_Weight)
                self.susceptible_nodes.append((row[0], row[1])) 
            conn.close()
            print(f"Neighbors list refreshed from ned.db. Found {len(self.susceptible_nodes)} neighbors.", flush=True)
        except Exception as e:
            print(f"Error in get_neighbors: {e}", flush=True)

    async def UpdateNeighbors(self, request, context):
        # ... (Existing code for neighbor update) ...
        print("Received UpdateNeighbors signal. Refreshing state...", flush=True)
        self.get_neighbors()
        self.received_message_ids.clear()
        print(f"Message cache cleared. New topology active.", flush=True)
        return gossip_pb2.Acknowledgment(details="Neighbors list and message cache have been updated.")

    def _decay_scores(self):
        """Applies forgiveness (decay) to all stored reputation scores."""
        new_scores = {}
        # NOTE: Iterating over items and rebuilding the dict is safe in asyncio
        for (h, neighbor_id), score in self.reputation_score.items():
            new_score = min(self.MAX_SCORE, score + self.DECAY_RATE)
            
            # Only keep the entry if the score is still below max
            if new_score < self.MAX_SCORE:
                new_scores[(h, neighbor_id)] = new_score
            
        self.reputation_score = new_scores

    # --- NEW RPC SERVICER: LEARNING/PENALTY RECEIPT (Executed on the Sender, Node A) ---
    async def NotifyPenalty(self, request, context):
        """
        Received by the node whose path was inefficient (Node A).
        This updates A's local score for the link to the penalized_neighbor_id (Node B).
        """
        message_id = request.message_id
        penalized_ip = request.penalized_neighbor_id # The IP of Node B
        wastage_hop = request.hop_of_wastage
        reporter_ip = request.reporter_node_id
        
        # The key is: (hop_h, the neighbor IP that was the wasteful target)
        reputation_key = (wastage_hop, penalized_ip)
        current_score = self.reputation_score.get(reputation_key, self.MAX_SCORE)
        new_score = max(0, current_score - self.PENALTY_VALUE)
        
        score_change = new_score - current_score

        # Update the score on the local map (Node A's memory)
        if score_change != 0:
            self.reputation_score[reputation_key] = new_score
        
        log_message = (f"{self.host} received penalty from {reporter_ip} for link to {penalized_ip} "
                       f"at hop {wastage_hop}. Score: {current_score} -> {new_score}")
        self._log_event(message_id, reporter_ip, time.time_ns(), None,
                         None, 'penalty_received', log_message, wastage_hop, score_change=score_change)

        return gossip_pb2.Acknowledgment(details="Penalty recorded.")
    
    # --- RPC SERVICER: MESSAGE RECEIPT (Executed on the Receiver, Node B) ---
    async def SendMessage(self, request, context):
        """
        Receiving message (M) from Node A, and distributing it or processing penalty.
        """
        message = request.message
        sender_id = request.sender_id # Node A's IP
        received_timestamp = time.time_ns()
        incoming_link_latency = request.latency_ms
        incoming_round_count = request.round_count  # Hop h

        if sender_id == self.host:
            # Initiation (h=0) - (Unchanged)
            # ... (initiation logic) ...
            self.received_message_ids.add(message)
            log_message = (f"Gossip initiated by {self.hostname} ({self.host})")
            self._log_event(message, sender_id, received_timestamp, None,
                             None, 'initiate', log_message, 0, score_change=0)
            await self.gossip_message(message, sender_id, 0)
            return gossip_pb2.Acknowledgment(details=f"Done propagate! {self.host} received: '{message}'")
        
        elif message in self.received_message_ids:
            # --- DUPLICATE DETECTED: SEND EXPLICIT PENALTY FEEDBACK ---
            
            # The node to notify is the one who sent the duplicate: Node A (sender_id).
            sender_ip_to_notify = sender_id 
            
            # The IP that was wasteful to send to is the current node: Node B (self.host).
            penalized_target_ip = self.host 
            
            # The hop where the wastage occurred
            wastage_hop = incoming_round_count

            # Asynchronously send the penalty back to the sender (Node A)
            asyncio.create_task(self._send_penalty_notification(
                sender_ip_to_notify, 
                message, 
                penalized_target_ip, 
                wastage_hop
            ))

            log_message = (f"{self.host} ignoring duplicate: {message} from {sender_id}. "
                           f"Sent penalty notification to {sender_ip_to_notify}.")
            self._log_event(message, sender_id, received_timestamp, None,
                             incoming_link_latency, 'duplicate_notify', log_message, 
                             incoming_round_count, score_change=0)
            # --------------------------------------------------------
            return gossip_pb2.Acknowledgment(details=f"Duplicate message ignored by ({self.host})")
        
        else:
            # First time receiving a message (Propagation) - (Unchanged)
            self.received_message_ids.add(message)
            propagation_time = (received_timestamp - request.timestamp) / 1e6
            log_message = (f"({self.hostname}({self.host}) received: '{message}' from {sender_id}"
                            f" in {propagation_time:.2f} ms.")
            self._log_event(message, sender_id, received_timestamp, propagation_time,
                             incoming_link_latency, 'received', log_message, 
                             incoming_round_count, score_change=0)

            new_round_count = incoming_round_count + 1
            await self.gossip_message(message, sender_id, new_round_count)
            
            return gossip_pb2.Acknowledgment(details=f"{self.host} received: '{message}'")

    # --- NEW HELPER METHOD ---
    async def _send_penalty_notification(self, target_ip, message_id, penalized_ip, hop):
        """Asynchronously sends the penalty notification to the specified node (Node A)."""
        try:
            # Use semaphore to limit concurrent outbound notification calls if needed, 
            # though usually not as critical as message forwarding.
            async with self.semaphore: 
                async with grpc.aio.insecure_channel(f"{target_ip}:5050") as channel:
                    stub = gossip_pb2_grpc.GossipServiceStub(channel)
                    await stub.NotifyPenalty(gossip_pb2.PenaltyNotification(
                        message_id=message_id,
                        penalized_neighbor_id=penalized_ip, 
                        reporter_node_id=self.host,
                        hop_of_wastage=hop
                    ))
        except Exception as e:
            print(f"Failed to send penalty notification to {target_ip}: {str(e)}", flush=True)

    async def _send_gossip_to_peer(self, message, sender_id, peer_ip, peer_weight, round_count): 
        # ... (Existing code for sending gossip message) ...
        send_timestamp = time.time_ns()
        async with self.semaphore:
            try:
                await asyncio.sleep(int(peer_weight) / 1000) 
                
                async with grpc.aio.insecure_channel(f"{peer_ip}:5050") as channel:
                    stub = gossip_pb2_grpc.GossipServiceStub(channel)
                    await stub.SendMessage(gossip_pb2.GossipMessage(
                        message=message,
                        sender_id=self.host, 
                        timestamp=send_timestamp,
                        latency_ms=peer_weight,
                        round_count=round_count
                    ))
            except Exception as e:
                print(f"Failed to send message: '{message}' to {peer_ip}: {str(e)}", flush=True)

    # --- FORWARDING LOGIC (Adaptive Selection and Pruning) ---
    async def gossip_message(self, message, source_sender_id, round_count=0): 
        if not self.susceptible_nodes:
            self.get_neighbors()
            
        current_hop = round_count
        
        # A. Score Decay (Forgiveness)
        self._decay_scores() 

        # B. Pruning Candidates (Adaptive Forwarding)
        nb_lists_candidate = []
        for peer_ip, peer_weight in self.susceptible_nodes:
            if peer_ip == source_sender_id:
                continue 
            
            # Key = (Current Hop h, Potential Receiver IP j)
            reputation_key = (current_hop, peer_ip) 
            current_score = self.reputation_score.get(reputation_key, self.MAX_SCORE)
            
            # Pruning check
            if current_score >= self.THRESHOLD_SCORE:
                nb_lists_candidate.append((peer_ip, peer_weight))
            else:
                print(f"PRUNED: {peer_ip} at hop {current_hop} due to low score: {current_score}", flush=True)

        # C. Selection (Adaptive RNS)
        if not nb_lists_candidate:
            print(f"Warning: No eligible neighbors for forwarding at hop {current_hop}", flush=True)
            return

        k_prime = min(self.FANOUT_K, len(nb_lists_candidate))
        
        # Perform RNS (Random Neighbor Selection) on the CANDIDATE list
        nb_lists_target = random.sample(nb_lists_candidate, k_prime)

        # D. Forwarding
        tasks = []
        for peer_ip, peer_weight in nb_lists_target:
            task = asyncio.create_task(self._send_gossip_to_peer(
                message, 
                self.host, 
                peer_ip, 
                peer_weight, 
                round_count 
            ))
            tasks.append(task)
            
        print(f"Forwarding from {self.host} at hop {current_hop} to {k_prime} neighbors. Candidates: {len(nb_lists_candidate)}", flush=True)
        await asyncio.gather(*tasks, return_exceptions=True)

    def _log_event(self, message, sender_id, received_timestamp, propagation_time, incoming_link_latency, event_type,
                   log_message, round_count, score_change=0):
        event_data = {
            'message': message,
            'sender_id': sender_id,
            'receiver_id': self.host,
            'received_timestamp': received_timestamp,
            'propagation_time': propagation_time,
            'incoming_link_latency': incoming_link_latency,
            'round_count': round_count,
            'event_type': event_type,
            'score_change': score_change, 
            'detail': log_message
        }
        print(json.dumps(event_data), flush=True)

    async def start_server(self):
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