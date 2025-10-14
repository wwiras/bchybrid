import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

# --- 1. CONFIGURATION PARAMETERS ---
NUM_NODES = 10         # Total number of nodes
M_INIT = 2             # Parameter for BA model: edges attached from a new node
FANOUT = 3             # The fixed number of neighbors a node attempts to send to (k)
SIMULATION_ROUNDS = 5 # Number of independent gossip events to simulate
MAX_GOSSIP_HOPS = 5    # Max number of rounds (hops) for a single message to spread
PENALTY = -1.0         # Penalty applied for sending to a duplicate-receiving neighbor
REWARD = 0.1           # Small reward for a successful, unique forward
INITIAL_SCORE = 0.0    # Initial reputation score for all neighbors

# --- 2. NODE AND SIMULATION CLASSES ---

class Node:
    def __init__(self, node_id, neighbors):
        self.id = node_id
        self.neighbors = neighbors
        # Reputation structure: {neighbor_id: {hop_count: score}}
        self.reputations = {n: {} for n in neighbors}
        self.received_messages = set()

    def get_forwarding_candidates(self, hop_count):
        """
        Retrieves neighbors and their reputation scores for the *current* hop count.
        Reputation defaults to INITIAL_SCORE if no history exists.
        """
        candidates = {}
        for neighbor_id in self.neighbors:
            score = self.reputations[neighbor_id].get(hop_count, INITIAL_SCORE)
            candidates[neighbor_id] = score
        return candidates

    def select_fanout_neighbors(self, hop_count):
        """
        Selects k=FANOUT neighbors using a weighted RNS based on reputation.
        (Higher score = higher probability of selection).
        """
        candidates = self.get_forwarding_candidates(hop_count)
        
        # 1. Map scores to positive weights:
        # Shift all scores by a factor to ensure all weights are positive.
        # This is a key Game Theory step: maximizing expected utility (score).
        score_shift = abs(PENALTY) + 1.0 
        weights = {
            n: score_shift + score
            for n, score in candidates.items()
        }

        k = min(FANOUT, len(self.neighbors))
        if k == 0:
            return set()
            
        # 2. Perform Weighted Random Selection
        selected_neighbors = random.choices(
            list(weights.keys()), 
            weights=list(weights.values()), 
            k=k
        )
        
        return set(selected_neighbors)

    def learn_from_feedback(self, neighbor_id, hop_count, is_duplicate):
        """
        Updates the reputation score based on whether the forward was a duplicate.
        """
        # Get current score, initialize if necessary
        current_score = self.reputations.get(neighbor_id, {}).get(hop_count, INITIAL_SCORE)
        
        if is_duplicate:
            # Apply a penalty for wasteful forwarding (reducing payoff)
            new_score = current_score + PENALTY
        else:
            # Apply a small reward for successful, unique forward (increasing payoff)
            new_score = current_score + REWARD

        # Ensure the nested structure exists and store the updated score
        if neighbor_id not in self.reputations:
            self.reputations[neighbor_id] = {}
            
        self.reputations[neighbor_id][hop_count] = new_score

class AdaptiveGossipSimulator:
    def __init__(self, num_nodes, m_init):
        self.graph = nx.barabasi_albert_graph(num_nodes, m_init)
        self.nodes = {}
        
        for node_id in self.graph.nodes:
            neighbors = list(self.graph.neighbors(node_id))
            self.nodes[node_id] = Node(node_id, neighbors)

    def run_single_gossip(self, message_id, start_node_id):
        """
        Simulates one full gossip run from a start node,
        allowing nodes to learn and adapt their reputation tables.
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
            queue = []
            
            for sender_id, current_hop in current_round_senders:
                sender_node = self.nodes[sender_id]
                
                # Sender selects neighbors based on learned reputation for current hop
                targets = sender_node.select_fanout_neighbors(current_hop)
                
                total_messages_sent += len(targets)
                
                # --- The Core Learning Loop ---
                for target_id in targets:
                    is_duplicate = target_id in received
                    
                    # 1. Update the sender's local reputation (learning)
                    sender_node.learn_from_feedback(target_id, current_hop, is_duplicate)

                    # 2. Propagate
                    if is_duplicate:
                        total_duplicates += 1
                    elif target_id not in received:
                        self.nodes[target_id].received_messages.add(message_id)
                        received.add(target_id)
                        # Add to the next round with increased hop count
                        queue.append((target_id, current_hop + 1))

        # Calculate metrics
        duplication_rate = total_duplicates / total_messages_sent if total_messages_sent > 0 else 0.0
        coverage = len(received) / NUM_NODES
        
        return duplication_rate, coverage, total_messages_sent


# --- 3. EXECUTION AND VISUALIZATION ---

# Initialize the simulator
simulator = AdaptiveGossipSimulator(NUM_NODES, M_INIT)

# Select a consistent starting node (e.g., the hub node)
start_node = max(simulator.graph.degree, key=lambda item: item[1])[0]

# Data storage for results
duplication_rates = []
coverage_rates = []

print(f"--- Running Adaptive Gossip for {SIMULATION_ROUNDS} Rounds ---")

for i in range(1, SIMULATION_ROUNDS + 1):
    message_id = f"MSG_{i}"
    
    # Run a single gossip simulation and get the results
    dup_rate, coverage, messages_sent = simulator.run_single_gossip(message_id, start_node)
    
    # Store results
    duplication_rates.append(dup_rate)
    coverage_rates.append(coverage)
    
    # Print status every 10 rounds
    if i % 10 == 0 or i == 1:
        print(f"Round {i:02d}: Dup Rate={dup_rate:.3f}, Coverage={coverage:.3f}, Msgs Sent={messages_sent}")
        
    # Check for catastrophic failure (e.g., coverage drops to 0)
    if coverage < 0.1 and i > 1:
        print(f"Stopping early at Round {i} due to low coverage.")
        break

# --- 4. RESULTS PLOTTING ---

rounds = range(1, len(duplication_rates) + 1)
avg_dup_initial = np.mean(duplication_rates[:10])
avg_dup_final = np.mean(duplication_rates[-10:])

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"Adaptive Gossip Learning on BA Network (N={NUM_NODES}, Fanout={FANOUT})", fontsize=16)


# --- Plot 1: Duplication Rate ---
ax1.plot(rounds, duplication_rates, label='Duplication Rate', color='red', alpha=0.7)
ax1.set_ylabel("Duplication Rate (Duplicates / Total Messages)", color='red')
ax1.tick_params(axis='y', labelcolor='red')

# Add trendlines for Duplication
ax1.axhline(avg_dup_initial, color='red', linestyle='--', alpha=0.5, 
            label=f'Initial Avg Dup: {avg_dup_initial:.3f}')
ax1.axhline(avg_dup_final, color='darkred', linestyle='-', alpha=0.8, 
            label=f'Final Avg Dup: {avg_dup_final:.3f}')
ax1.legend(loc='upper right', frameon=True)
ax1.set_title(f"Duplication Reduction: {avg_dup_initial:.3f} $\\rightarrow$ {avg_dup_final:.3f}", fontsize=12)


# --- Plot 2: Network Coverage ---
ax2.plot(rounds, coverage_rates, label='Coverage Rate', color='blue', alpha=0.7)
ax2.axhline(1.0, color='green', linestyle='-', alpha=0.8, label='Optimal Coverage (1.0)')
ax2.set_ylim(0.0, 1.05)
ax2.set_xlabel("Gossip Round (Successive Messages)")
ax2.set_ylabel("Network Coverage (Nodes Informed / Total Nodes)", color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.legend(loc='lower right', frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
# plt.show()