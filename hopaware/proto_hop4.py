import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# --- 1. UPDATED CONFIGURATION PARAMETERS (Final Logic) ---
NUM_NODES = 10         
M_INIT = 3             
FANOUT_MAX = 3         # Maximum fanout allowed (k_max)
FANOUT_MIN = 2         # CRITICAL: Minimum fanout guaranteed (k_min)
SIMULATION_ROUNDS = 50 
MAX_GOSSIP_HOPS = 5    

# --- ADAPTIVE MECHANISM PARAMETERS ---
DECAY_WINDOW = 15      # Increased memory to stabilize learning


# --- 2. NODE AND SIMULATION CLASSES ---

class Node:
    def __init__(self, node_id, neighbors):
        self.id = node_id
        self.neighbors = neighbors
        self.potential_bad_neighbors = {}
        self.received_messages = set()

    def select_fanout_neighbors(self, current_hop, current_message_id):
        """
        Implements Hop-Aware Pruning (HAP) with a GUARANTEED MINIMUM FANOUT (HMF).
        If the good list is too small, it dips into the bad list to meet k_min.
        """
        nb_lists_all = set(self.neighbors)
        nb_lists_bad_for_hop = set()
        
        # 1. Prune Stale/Irrelevant Penalties
        decay_threshold = current_message_id - DECAY_WINDOW
        
        for (neighbor_id, hop), penalty_time in self.potential_bad_neighbors.items():
            if hop == current_hop and penalty_time >= decay_threshold:
                nb_lists_bad_for_hop.add(neighbor_id)
        
        # 2. Determine Good and Bad Pools
        nb_lists_candidate_good = nb_lists_all - nb_lists_bad_for_hop
        nb_lists_candidate_bad = nb_lists_bad_for_hop

        # 3. Dynamic Selection Strategy (Mixing Good and Bad)
        
        # A. Start by selecting all available good neighbors (up to FANOUT_MAX)
        num_to_select_from_good = min(FANOUT_MAX, len(nb_lists_candidate_good))
        
        selected_targets = set(random.sample(
            list(nb_lists_candidate_good), 
            k=num_to_select_from_good
        ))
        
        # B. Check for Minimum Fanout Requirement
        num_current_targets = len(selected_targets)
        
        if num_current_targets < FANOUT_MIN:
            # We must use "bad" neighbors (Forced Exploration) to meet k_min.
            
            num_remaining_slots = FANOUT_MIN - num_current_targets
            
            # Select randomly from the available bad list (neighbors not yet chosen)
            available_bad = list(nb_lists_candidate_bad - selected_targets)
            
            num_to_select_from_bad = min(num_remaining_slots, len(available_bad))
            
            selected_bad = set(random.sample(available_bad, k=num_to_select_from_bad))
            
            selected_targets.update(selected_bad)
        
        # C. Ensure we don't exceed FANOUT_MAX (though this should be impossible based on logic)
        final_targets = selected_targets
        
        return final_targets

    def learn_from_feedback(self, neighbor_id, hop_count, current_message_id):
        """
        Updates the local storage when a forward to a neighbor results in a duplicate.
        """
        self.potential_bad_neighbors[(neighbor_id, hop_count)] = current_message_id


class HAPGossipSimulator:
    def __init__(self, num_nodes, m_init):
        self.graph = nx.barabasi_albert_graph(num_nodes, m_init)
        self.nodes = {}
        
        for node_id in self.graph.nodes:
            neighbors = list(self.graph.neighbors(node_id))
            self.nodes[node_id] = Node(node_id, neighbors)

    def run_single_gossip(self, message_id, start_node_id):
        total_messages_sent = 0
        total_duplicates = 0
        
        for node in self.nodes.values():
            node.received_messages.discard(message_id)

        queue = [(start_node_id, 1)] 
        received = {start_node_id}
        self.nodes[start_node_id].received_messages.add(message_id)
        
        round_count = 0

        while queue and round_count < MAX_GOSSIP_HOPS:
            round_count += 1
            current_round_senders = queue
            queue = []
            
            for sender_id, current_hop in current_round_senders:
                sender_node = self.nodes[sender_id]
                
                targets = sender_node.select_fanout_neighbors(current_hop, message_id)
                
                total_messages_sent += len(targets)
                
                for target_id in targets:
                    is_duplicate = target_id in received
                    
                    if is_duplicate:
                        total_duplicates += 1
                        sender_node.learn_from_feedback(target_id, current_hop, message_id)
                        
                    elif target_id not in received:
                        self.nodes[target_id].received_messages.add(message_id)
                        received.add(target_id)
                        queue.append((target_id, current_hop + 1))

        duplication_rate = total_duplicates / total_messages_sent if total_messages_sent > 0 else 0.0
        coverage = len(received) / NUM_NODES
        
        return duplication_rate, coverage, total_messages_sent


# --- 3. EXECUTION AND RESULTS ANALYSIS ---

# Initialize the simulator
simulator = HAPGossipSimulator(NUM_NODES, M_INIT)

start_node = max(simulator.graph.degree, key=lambda item: item[1])[0]

# Data storage for results
duplication_rates = []
coverage_rates = []

print(f"--- Running HMF-Gossip (Min/Max Fanout) for {SIMULATION_ROUNDS} Rounds ---")
print(f"Starting Node: {start_node} (Hub), Fanout: Min={FANOUT_MIN}, Max={FANOUT_MAX}, Decay Window: {DECAY_WINDOW}")
print("-----------------------------------------------------------------------------------")

for i in range(1, SIMULATION_ROUNDS + 1):
    message_id = i
    
    dup_rate, coverage, messages_sent = simulator.run_single_gossip(message_id, start_node)
    
    duplication_rates.append(dup_rate)
    coverage_rates.append(coverage)
    
    if i % 10 == 0 or i <= 5:
        print(f"Round {i:02d}: Dup Rate={dup_rate:.3f}, Coverage={coverage:.3f}, Msgs Sent={messages_sent}")
        

# --- 4. SUMMARY AND PLOTTING ---

rounds = range(1, len(duplication_rates) + 1)
num_avg_rounds = min(10, len(duplication_rates))
avg_dup_initial = np.mean(duplication_rates[:num_avg_rounds])
avg_dup_final = np.mean(duplication_rates[-num_avg_rounds:])
avg_cov_final = np.mean(coverage_rates[-num_avg_rounds:])

print("\n--- Summary ---")
print(f"Initial Avg Dup Rate (Rounds 1-{num_avg_rounds}): {avg_dup_initial:.3f}")
print(f"Final Avg Dup Rate (Rounds {SIMULATION_ROUNDS - num_avg_rounds + 1}-{SIMULATION_ROUNDS}): {avg_dup_final:.3f}")
print(f"Final Avg Coverage (Goal: 1.0): {avg_cov_final:.3f}")
print("-----------------")

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"HMF-Gossip Adaptation: Duplication vs. Coverage (Min/Max Fanout)\n"
             f"Network: BA({NUM_NODES}, M={M_INIT}), Fanout: Min={FANOUT_MIN}, Max={FANOUT_MAX}, Decay Window: {DECAY_WINDOW} Rounds", fontsize=16)


# --- Plot 1: Duplication Rate ---
ax1.plot(rounds, duplication_rates, label='Duplication Rate', color='red', alpha=0.7)
ax1.set_ylabel("Duplication Rate (Wastage)", color='red')
ax1.tick_params(axis='y', labelcolor='red')

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
plt.show()
