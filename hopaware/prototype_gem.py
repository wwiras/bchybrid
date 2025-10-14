import networkx as nx
import random
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION PARAMETERS ---
NUM_NODES = 10  # Total number of nodes
M_INIT = 2      # Parameter for BA model: number of edges to attach from a new node to existing nodes
FANOUT = 3      # The fixed number of neighbors a node attempts to send to (k)
SCORE_DECAY = 0.5 # Factor to reduce penalty (low score) over time/new messages (not used in this single-run model, but good practice)
PENALTY = -1.0  # Penalty applied for sending to a duplicate-receiving neighbor
INITIAL_SCORE = 0.0 # Initial reputation score for all neighbors

# --- 2. NODE AND SIMULATION CLASSES ---

class Node:
    def __init__(self, node_id, neighbors):
        self.id = node_id
        self.neighbors = neighbors
        # Reputation structure: {neighbor_id: {hop_count: score}}
        # Example: {3: {1: 0.0, 2: -1.0}} -> neighbor 3 at h=2 is a 'bad' path
        self.reputations = {n: {} for n in neighbors}
        self.received_messages = set()

    def get_forwarding_candidates(self, hop_count):
        """
        Retrieves neighbors and their reputation scores for the *current* hop count.
        Reputation defaults to INITIAL_SCORE if no history exists for this hop count.
        """
        candidates = {}
        for neighbor_id in self.neighbors:
            # Check the reputation for this specific neighbor and hop count
            score = self.reputations[neighbor_id].get(hop_count, INITIAL_SCORE)
            candidates[neighbor_id] = score
        return candidates

    def select_fanout_neighbors(self, hop_count):
        """
        Selects k=FANOUT neighbors using a weighted RNS based on reputation.
        Lower scores mean lower probability of selection.
        """
        candidates = self.get_forwarding_candidates(hop_count)
        
        # 1. Map scores to positive weights (higher score = higher weight)
        # We need to shift all scores to be positive to use them as selection weights.
        # Max penalty is PENALTY (-1.0). Shifting by abs(PENALTY) + 1.0 ensures all are > 1.0.
        score_shift = abs(PENALTY) + 1.0
        weights = {
            n: score_shift + score  # All weights are now > 1.0 (or some small positive number)
            for n, score in candidates.items()
        }

        # Handle case where FANOUT is larger than the number of neighbors
        k = min(FANOUT, len(self.neighbors))

        if k == 0:
            return set()
            
        # 2. Perform Weighted Random Selection
        # The 'choices' function uses weights for selection probability
        selected_neighbors = random.choices(
            list(weights.keys()), 
            weights=list(weights.values()), 
            k=k
        )
        
        return set(selected_neighbors)

    def learn_from_feedback(self, neighbor_id, hop_count, is_duplicate):
        """
        Updates the reputation score based on feedback.
        """
        # Initialize the score for this hop_count if it doesn't exist
        current_score = self.reputations[neighbor_id].get(hop_count, INITIAL_SCORE)
        
        if is_duplicate:
            # Apply a penalty for wasteful forwarding at this specific hop
            new_score = current_score + PENALTY
        else:
            # Optionally: Apply a small reward for successful forwarding
            new_score = current_score + 0.1 

        # Store the updated score tied to the hop count
        if hop_count not in self.reputations[neighbor_id]:
            self.reputations[neighbor_id] = {}
            
        self.reputations[neighbor_id][hop_count] = new_score

class AdaptiveGossipSimulator:
    def __init__(self, num_nodes, m_init):
        # Create Barabási-Albert graph
        self.graph = nx.barabasi_albert_graph(num_nodes, m_init)
        self.nodes = {}
        
        # Initialize Node objects based on the graph
        for node_id in self.graph.nodes:
            neighbors = list(self.graph.neighbors(node_id))
            self.nodes[node_id] = Node(node_id, neighbors)

        self.total_messages_sent = 0
        self.total_duplicates = 0
        self.gossiped_nodes = set()

    def simulate_gossip_round(self, message_id, start_node_id, max_rounds=5):
        """
        Simulates one full gossip run from a start node.
        """
        self.total_messages_sent = 0
        self.total_duplicates = 0
        self.gossiped_nodes = set()
        
        # Queue: (node_id, hop_count, source_node_id)
        queue = [(start_node_id, 1, None)] 
        
        # Set to track which nodes have received the message (for duplicate detection)
        received = {start_node_id}
        self.nodes[start_node_id].received_messages.add(message_id)
        self.gossiped_nodes.add(start_node_id)
        
        round_count = 0

        print(f"--- Starting Gossip from Node {start_node_id} (Message {message_id}) ---")
        
        while queue and round_count < max_rounds:
            round_count += 1
            current_round_senders = queue
            queue = []
            
            print(f"Round {round_count}: Nodes informed: {len(received)}/{NUM_NODES}")

            for sender_id, current_hop, _ in current_round_senders:
                sender_node = self.nodes[sender_id]
                
                # Node selects neighbors based on learned reputation for current hop
                targets = sender_node.select_fanout_neighbors(current_hop)
                
                self.total_messages_sent += len(targets)
                
                # --- The Core Learning Mechanism ---
                
                # Check which neighbors are already "informed"
                for target_id in targets:
                    is_duplicate = target_id in received
                    
                    # 1. Update the sender's local reputation (learning)
                    sender_node.learn_from_feedback(target_id, current_hop, is_duplicate)

                    # 2. Propagate (if not duplicate, add to received set and next round queue)
                    if is_duplicate:
                        self.total_duplicates += 1
                    elif target_id not in self.gossiped_nodes:
                        self.nodes[target_id].received_messages.add(message_id)
                        received.add(target_id)
                        self.gossiped_nodes.add(target_id)
                        # Add to the next round with increased hop count
                        queue.append((target_id, current_hop + 1, sender_id))

        print(f"Gossip Complete. Total Hops: {round_count}. Total Duplicates: {self.total_duplicates}")
        return self.total_messages_sent, self.total_duplicates, len(received)

    def print_reputations(self, node_id):
        """Helper to display the local knowledge of a node."""
        print(f"\n--- Node {node_id} Learned Reputation (Hop-Aware) ---")
        reps = self.nodes[node_id].reputations
        for neighbor, hop_data in reps.items():
            if hop_data:
                print(f"Neighbor {neighbor}: {hop_data}")

# --- 3. EXECUTION AND VISUALIZATION ---

# Create the simulator
simulator = AdaptiveGossipSimulator(NUM_NODES, M_INIT)

# Get the node with the highest degree (best centrality for a start)
start_node = max(simulator.graph.degree, key=lambda item: item[1])[0]

# --- RUN 1: WARM-UP (Learning Phase) ---
print("=========================================")
print("          RUN 1: LEARNING PHASE          ")
print("=========================================")
msg_id_1 = "MSG_A"
sent_1, dup_1, informed_1 = simulator.simulate_gossip_round(msg_id_1, start_node)

print(f"\nRESULTS RUN 1:")
print(f"  Total Messages Sent: {sent_1}")
print(f"  Total Duplicates: {dup_1}")
print(f"  Duplication Rate: {dup_1 / sent_1 if sent_1 > 0 else 0.0:.2f}")

# Check the learned reputation of the starting node
simulator.print_reputations(start_node)


# --- RUN 2: ADAPTIVE FORWARDING (Testing Phase) ---
# The node now uses the reputation learned in RUN 1
print("\n=========================================")
print("          RUN 2: ADAPTIVE PHASE          ")
print("=========================================")
# Reset received status, but keep reputation
for node in simulator.nodes.values():
    node.received_messages.clear()
simulator.total_messages_sent = 0
simulator.total_duplicates = 0
simulator.gossiped_nodes = set()

# Gossip a new message (simulating the *same type* of network traffic)
msg_id_2 = "MSG_B"
sent_2, dup_2, informed_2 = simulator.simulate_gossip_round(msg_id_2, start_node)

print(f"\nRESULTS RUN 2 (Adaptive):")
print(f"  Total Messages Sent: {sent_2}")
print(f"  Total Duplicates: {dup_2}")
print(f"  Duplication Rate: {dup_2 / sent_2 if sent_2 > 0 else 0.0:.2f}")


# --- VISUALIZATION ---
plt.figure(figsize=(10, 6))

# Define node colors based on degree (high degree nodes are important)
degrees = dict(simulator.graph.degree())
node_colors = [degrees[node] for node in simulator.graph.nodes()]

# Define node sizes based on "bad" reputation (a heuristic)
# For simplicity, we aggregate all negative scores a node has given out
def get_node_penalty_size(node):
    penalty = 0
    for neighbor_data in node.reputations.values():
        for score in neighbor_data.values():
            if score < 0:
                penalty += abs(score)
    # Scale size: larger for nodes that penalized more (learned a lot)
    return 500 + penalty * 200

node_sizes = [get_node_penalty_size(simulator.nodes[n]) for n in simulator.graph.nodes()]

# Use the spring layout
pos = nx.spring_layout(simulator.graph, seed=42)

# Draw the graph
nx.draw(
    simulator.graph, 
    pos, 
    with_labels=True, 
    node_color=node_colors, 
    cmap=plt.cm.coolwarm, # Use a colormap to show degrees
    node_size=node_sizes, 
    font_weight='bold', 
    edge_color='gray'
)

# Add titles and legend
plt.title(f"BA Network Gossip Simulation (N={NUM_NODES}, Fanout={FANOUT})\n"
          f"Learning Outcome: Run 1 Dup={dup_1/sent_1:.2f} | Run 2 Dup={dup_2/sent_2:.2f}",
          fontsize=14)
sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm)
sm.set_array(node_colors)
cbar = plt.colorbar(sm, label='Node Degree (Centrality)')
plt.show()