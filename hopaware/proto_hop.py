import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# --- 1. CONFIGURATION PARAMETERS ---
NUM_NODES = 10         # Total number of nodes
M_INIT = 3             # Parameter for BA model (edges to attach from a new node)
FANOUT = 3             # Number of neighbors selected for forwarding (k)
SIMULATION_ROUNDS = 50 # Number of independent message rounds for learning
MAX_GOSSIP_HOPS = 5    # Max rounds (hops) for a single message to spread

# --- ADAPTIVE MECHANISM PARAMETERS ---
# Decay Window (X): A neighbor is excluded only if its penalty was received 
# within the last X message rounds. This allows for 'forgetting' (Decay).
DECAY_WINDOW = 5       


# --- 2. NODE AND SIMULATION CLASSES ---

class Node:
    def __init__(self, node_id, neighbors):
        self.id = node_id
        self.neighbors = neighbors
        # Local Storage: {(neighbor_id, hop_count): penalty_time (message_id)}
        self.potential_bad_neighbors = {}
        self.received_messages = set()

    def select_fanout_neighbors(self, current_hop, current_message_id):
        """
        Implements the pruning strategy: nb_lists_good = nb_lists_all - nb_lists_bad.
        """
        nb_lists_all = set(self.neighbors)
        nb_lists_bad_for_hop = set()
        
        # 1. Pruning Candidates (Step A & B2 in pseudo-code)
        decay_threshold = current_message_id - DECAY_WINDOW
        
        # Identify bad neighbors that are still "fresh" for this hop count
        for (neighbor_id, hop), penalty_time in self.potential_bad_neighbors.items():
            if hop == current_hop and penalty_time >= decay_threshold:
                nb_lists_bad_for_hop.add(neighbor_id)
        
        # 2. Get Good Neighbor List (Step B3)
        nb_lists_candidate = nb_lists_all - nb_lists_bad_for_hop
        
        k = min(FANOUT, len(nb_lists_candidate))

        if k == 0:
            return set()
            
        # 3. RNS Selection from the Good List (Refinement)
        # We use random.sample for selection without replacement, ensuring diversity
        nb_lists_target = set(random.sample(
            list(nb_lists_candidate), 
            k=k
        ))
        
        return nb_lists_target

    def learn_from_feedback(self, neighbor_id, hop_count, current_message_id):
        """
        Updates the local storage when a forward to a neighbor results in a duplicate.
        """
        # Store the bad behavior tied to the hop count and the current message ID (time)
        self.potential_bad_neighbors[(neighbor_id, hop_count)] = current_message_id


class HAPGossipSimulator:
    def __init__(self, num_nodes, m_init):
        # Create BA graph
        self.graph = nx.barabasi_albert_graph(num_nodes, m_init)
        self.nodes = {}
        
        for node_id in self.graph.nodes:
            neighbors = list(self.graph.neighbors(node_id))
            self.nodes[node_id] = Node(node_id, neighbors)

    def run_single_gossip(self, message_id, start_node_id):
        """
        Simulates one full message dissemination (M) starting at h=1.
        """
        total_messages_sent = 0
        total_duplicates = 0
        
        # Queue: (node_id, hop_count)
        queue = [(start_node_id, 1)] 
        
        # Set to track which nodes have received the message (for duplicate detection)
        received = {start_node_id}
        self.nodes[start_node_id].received_messages.add(message_id)
        
        round_count = 0

        while queue and round_count < MAX_GOSSIP_HOPS:
            round_count += 1
            current_round_senders = queue
            queue = [] # Queue for the next round (h+1)
            
            for sender_id, current_hop in current_round_senders:
                sender_node = self.nodes[sender_id]
                
                # B. Neighbor Selection Strategy
                targets = sender_node.select_fanout_neighbors(current_hop, message_id)
                
                total_messages_sent += len(targets)
                
                # C. Forwarding and D. Learning (Feedback)
                for target_id in targets:
                    is_duplicate = target_id in received
                    
                    if is_duplicate:
                        total_duplicates += 1
                        
                        # D. Learning/Update (Step B3/D in pseudo-code)
                        # The sender learns that this was a bad path at this hop
                        sender_node.learn_from_feedback(target_id, current_hop, message_id)
                        
                    elif target_id not in received:
                        # Successful, unique forward
                        self.nodes[target_id].received_messages.add(message_id)
                        received.add(target_id)
                        # Propagate (add to the next round with increased hop count)
                        queue.append((target_id, current_hop + 1))

        # Calculate metrics
        duplication_rate = total_duplicates / total_messages_sent if total_messages_sent > 0 else 0.0
        coverage = len(received) / NUM_NODES
        
        return duplication_rate, coverage, total_messages_sent


# --- 3. EXECUTION AND RESULTS ANALYSIS ---

# Initialize the simulator
simulator = HAPGossipSimulator(NUM_NODES, M_INIT)

# Select a consistent starting node (e.g., the hub node)
start_node = max(simulator.graph.degree, key=lambda item: item[1])[0]

# Data storage for results
duplication_rates = []
coverage_rates = []

print(f"--- Running HAP-Gossip for {SIMULATION_ROUNDS} Rounds (Decay Window: {DECAY_WINDOW}) ---")
print(f"Starting Node: {start_node} (Hub), Fanout: {FANOUT}, Network Size: {NUM_NODES}")
print("---------------------------------------------------------------------")

for i in range(1, SIMULATION_ROUNDS + 1):
    message_id = i # Use the round number as the unique message ID (time)
    
    # Run a single gossip simulation and get the results
    dup_rate, coverage, messages_sent = simulator.run_single_gossip(message_id, start_node)
    
    # Store results
    duplication_rates.append(dup_rate)
    coverage_rates.append(coverage)
    
    # Print status every 10 rounds or for the first few rounds
    if i % 10 == 0 or i <= 5:
        print(f"Round {i:02d}: Dup Rate={dup_rate:.3f}, Coverage={coverage:.3f}, Msgs Sent={messages_sent}")
        
    # Optional: Stop early if coverage remains low to debug
    if i > 10 and np.mean(coverage_rates[-5:]) < 0.5:
        print(f"Stopping early at Round {i} due to sustained low coverage.")
        # break 

# --- 4. RESULTS PLOTTING ---

# Calculate initial and final averages for analysis
rounds = range(1, len(duplication_rates) + 1)
num_avg_rounds = min(10, len(duplication_rates))
avg_dup_initial = np.mean(duplication_rates[:num_avg_rounds])
avg_dup_final = np.mean(duplication_rates[-num_avg_rounds:])
avg_cov_final = np.mean(coverage_rates[-num_avg_rounds:])

print("\n--- Summary ---")
print(f"Initial Avg Dup Rate (Rounds 1-{num_avg_rounds}): {avg_dup_initial:.3f}")
print(f"Final Avg Dup Rate (Rounds {SIMULATION_ROUNDS - num_avg_rounds + 1}-{SIMULATION_ROUNDS}): {avg_dup_final:.3f}")
print(f"Final Avg Coverage (Goal: 1.0): {avg_cov_final:.3f}")
print("----------------")

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"HAP-Gossip Adaptation: Duplication vs. Coverage\n"
             f"Network: BA({NUM_NODES}, M={M_INIT}), Fanout: {FANOUT}, Decay Window: {DECAY_WINDOW} Rounds", fontsize=16)


# --- Plot 1: Duplication Rate ---
ax1.plot(rounds, duplication_rates, label='Duplication Rate', color='red', alpha=0.7)
ax1.set_ylabel("Duplication Rate (Wastage)", color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Add trendlines for Duplication
ax1.axhline(avg_dup_initial, color='red', linestyle='--', alpha=0.5, 
            label=f'Initial Avg: {avg_dup_initial:.3f}')
ax1.axhline(avg_dup_final, color='darkred', linestyle='-', alpha=0.8, 
            label=f'Final Avg: {avg_dup_final:.3f}')
ax1.legend(loc='upper right', frameon=True)
ax1.set_title("Duplication Rate Reduction Over Time", fontsize=12)


# --- Plot 2: Network Coverage ---
ax2.plot(rounds, coverage_rates, label='Coverage Rate', color='blue', alpha=0.7)
ax2.axhline(1.0, color='green', linestyle='-', alpha=0.8, label='Optimal Coverage (1.0)')
ax2.set_ylim(0.0, 1.05)
ax2.set_xlabel("Gossip Round (Successive Messages)")
ax2.set_ylabel("Network Coverage (Reliability)", color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.legend(loc='lower right', frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
# plt.show()
