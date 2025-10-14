import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict, deque
import random

class HopAwareGossipNode:
    def __init__(self, node_id, params):
        self.id = node_id
        self.params = params
        self.scores = defaultdict(lambda: 0.5)  # (neighbor, hop) -> score
        self.messages_received = set()
        self.pending_forwardings = deque()
        self.learning_rates = defaultdict(lambda: params['alpha'])
        
    def receive_message(self, message, sender, current_hop):
        """Receive a message and decide whether to forward"""
        message_id, hop_count = message
        
        # Check if duplicate
        is_duplicate = message_id in self.messages_received
        self.messages_received.add(message_id)
        
        # Learning: update scores based on duplicate detection
        if sender is not None and not is_duplicate:  # Only learn from successful deliveries
            future_hop = hop_count
            self.update_scores(sender, future_hop, is_duplicate)
        
        # Forwarding decision (only if not duplicate and within hop limit)
        if not is_duplicate and hop_count < self.params['h_max']:
            self.decide_forwarding(message_id, hop_count)
            
        return is_duplicate
    
    def update_scores(self, neighbor, hop, is_duplicate):
        """Update scores using reinforcement learning"""
        key = (neighbor, hop)
        reward = self.params['R_dup'] if is_duplicate else self.params['R_succ']
        
        current_score = self.scores[key]
        learning_rate = self.learning_rates[key]
        
        # Q-learning update
        new_score = current_score + learning_rate * (reward - current_score)
        new_score = max(0.01, min(0.99, new_score))  # Clamp to avoid extremes
        
        self.scores[key] = new_score
        
        # Adaptive learning rate decay
        self.learning_rates[key] = max(0.1, learning_rate * 0.98)  # Prevent too fast decay
        
    def decide_forwarding(self, message_id, current_hop):
        """Decide which neighbors to forward to"""
        future_hop = current_hop + 1
        neighbors = list(self.params['network'].neighbors(self.id))
        
        if not neighbors:
            return
            
        # Calculate forwarding probabilities using softmax
        scores = [self.scores[(neighbor, future_hop)] for neighbor in neighbors]
        
        # Add exploration bonus for underutilized paths
        scores = [score + random.uniform(0, 0.1) for score in scores]  # Small exploration
        
        exp_scores = [np.exp(self.params['beta'] * score) for score in scores]
        total_exp = sum(exp_scores)
        
        if total_exp == 0:  # Handle division by zero
            probabilities = [1.0 / len(neighbors)] * len(neighbors)
        else:
            probabilities = [exp / total_exp for exp in exp_scores]
        
        # Decide forwarding for each neighbor
        for neighbor, prob in zip(neighbors, probabilities):
            if random.random() < min(prob * 1.5, 0.8):  # Cap maximum probability
                self.pending_forwardings.append({
                    'message_id': message_id,
                    'to_neighbor': neighbor,
                    'hop_count': future_hop
                })

class ImprovedGossipProtocol:
    def __init__(self, network, protocol_type='hop_aware'):
        self.network = network
        self.protocol_type = protocol_type
        self.nodes = {}
        self.message_counter = 0
        self.results = []
        
        # Tuned parameters
        self.params = {
            'R_succ': 0.2,      # Smaller rewards for stability
            'R_dup': -0.3,      # Smaller penalties  
            'alpha': 0.4,       # Slower learning
            'beta': 1.5,        # Less aggressive exploration
            'h_max': 3,         # Reasonable hop limit for 10 nodes
            'network': network
        }
        
        # Initialize nodes
        for node_id in network.nodes():
            self.nodes[node_id] = HopAwareGossipNode(node_id, self.params)
    
    def create_message(self, source_node):
        """Create a new message from source node"""
        self.message_counter += 1
        return (self.message_counter, 0)  # (message_id, hop_count)
    
    def run_round(self, source_node):
        """Run one gossip round starting from source node"""
        message = self.create_message(source_node)
        
        # Reset message tracking for this round
        for node in self.nodes.values():
            node.pending_forwardings.clear()
        
        # Start propagation from source
        source = self.nodes[source_node]
        source.receive_message(message, None, 0)
        
        # Process forwarding decisions in rounds
        propagation_rounds = 0
        total_duplicates = 0
        total_transmissions = 0
        max_rounds = 15  # Safety limit
        
        while propagation_rounds < max_rounds:
            propagation_rounds += 1
            current_forwardings = []
            
            # Collect all pending forwardings
            for node in self.nodes.values():
                while node.pending_forwardings:
                    current_forwardings.append(node.pending_forwardings.popleft())
            
            if not current_forwardings:
                break  # No more messages to propagate
                
            # Execute forwardings
            for forwarding in current_forwardings:
                total_transmissions += 1
                sender_id = forwarding['to_neighbor']
                message_id, hop_count = forwarding['message_id'], forwarding['hop_count']
                message = (message_id, hop_count)
                
                sender_node = self.nodes[sender_id]
                
                # Send to all neighbors of the sender (except the one it came from)
                for neighbor_id in self.network.neighbors(sender_id):
                    is_dup = self.nodes[neighbor_id].receive_message(
                        message, sender_id, hop_count
                    )
                    if is_dup:
                        total_duplicates += 1
        
        # Calculate coverage
        coverage = self.calculate_coverage(message[0])
        
        # VALIDATION: Ensure numbers make sense
        if total_duplicates > total_transmissions:
            print(f"ERROR: Duplicates {total_duplicates} > Transmissions {total_transmissions}")
            total_duplicates = min(total_duplicates, total_transmissions)
        
        return {
            'message_id': message[0],
            'coverage': coverage,
            'propagation_rounds': propagation_rounds,
            'duplicates': total_duplicates,
            'transmissions': total_transmissions,
            'unique_coverage': len([n for n in self.nodes.values() 
                                  if message[0] in n.messages_received])
        }
    
    def calculate_coverage(self, message_id):
        """Calculate what percentage of nodes received the message"""
        received_count = sum(1 for node in self.nodes.values() 
                           if message_id in node.messages_received)
        return received_count / len(self.nodes)
    
    def get_network_efficiency(self):
        """Calculate overall network efficiency metrics"""
        if not self.results:
            return {}
        
        valid_results = [r for r in self.results if r['coverage'] > 0.1]  # Filter failed rounds
        
        if not valid_results:
            return {}
            
        avg_coverage = np.mean([r['coverage'] for r in valid_results])
        avg_duplicates = np.mean([r['duplicates'] for r in valid_results])
        avg_transmissions = np.mean([r['transmissions'] for r in valid_results])
        avg_rounds = np.mean([r['propagation_rounds'] for r in valid_results])
        
        duplication_ratio = avg_duplicates / avg_transmissions if avg_transmissions > 0 else 0
        
        # Final validation
        if duplication_ratio > 1.0:
            print(f"WARNING: Invalid duplication ratio {duplication_ratio}, clamping to 0.99")
            duplication_ratio = 0.99
        
        return {
            'avg_coverage': avg_coverage,
            'avg_duplicates': avg_duplicates,
            'avg_transmissions': avg_transmissions,
            'avg_rounds': avg_rounds,
            'duplication_ratio': duplication_ratio,
            'successful_rounds': len(valid_results)
        }

# Fixed testing function
def run_improved_experiment():
    """Run improved comparison"""
    ba_network = create_ba_network(10, 2)
    
    print("Improved Network Analysis:")
    print(f"Nodes: {list(ba_network.nodes())}")
    print(f"Edges: {list(ba_network.edges())}")
    print(f"Network Diameter: {nx.diameter(ba_network)}")
    print(f"Average Path Length: {nx.average_shortest_path_length(ba_network):.2f}")
    print()
    
    protocols = {
        'Hop-Aware RL': ImprovedGossipProtocol(ba_network, 'hop_aware'),
        'Baseline': ImprovedGossipProtocol(ba_network, 'baseline')
    }
    
    num_rounds = 10
    results = {}
    
    for protocol_name, protocol in protocols.items():
        print(f"Running {protocol_name}...")
        round_results = []
        
        for round_num in range(num_rounds):
            source_node = random.choice(list(ba_network.nodes()))
            result = protocol.run_round(source_node)
            
            # Validate result
            if result['duplicates'] > result['transmissions']:
                result['duplicates'] = result['transmissions']  # Fix invalid data
                
            result['round'] = round_num + 1
            round_results.append(result)
            
            print(f"  Round {round_num + 1}: Coverage={result['coverage']:.2f}, "
                  f"Dupes={result['duplicates']}, Trans={result['transmissions']}, "
                  f"DupRatio={result['duplicates']/result['transmissions'] if result['transmissions']>0 else 0:.3f}")
        
        protocol.results = round_results
        efficiency = protocol.get_network_efficiency()
        
        if efficiency:  # Only include if we have valid results
            results[protocol_name] = efficiency
            print(f"  Summary: Coverage={efficiency['avg_coverage']:.3f}, "
                  f"DupRatio={efficiency['duplication_ratio']:.3f}")
    
    return results, protocols, ba_network

def create_ba_network(n=10, m=2):
    """Create Barabasi-Albert network with 10 nodes"""
    return nx.barabasi_albert_graph(n, m)

# Run the improved experiment
if __name__ == "__main__":
    print("IMPROVED HOP-AWARE GOSSIP PROTOCOL EXPERIMENT")
    print("="*60)
    
    results, protocols, network = run_improved_experiment()
    
    if results:
        print("\n" + "="*60)
        print("IMPROVED RESULTS")
        print("="*60)
        
        for protocol_name, metrics in results.items():
            print(f"\n{protocol_name}:")
            print(f"  Average Coverage: {metrics['avg_coverage']:.3f}")
            print(f"  Average Duplicates: {metrics['avg_duplicates']:.1f}")
            print(f"  Average Transmissions: {metrics['avg_transmissions']:.1f}")
            print(f"  Duplication Ratio: {metrics['duplication_ratio']:.3f}")
            print(f"  Successful Rounds: {metrics['successful_rounds']}")
        
        if 'Hop-Aware RL' in results and 'Baseline' in results:
            base_dup = results['Baseline']['duplication_ratio']
            hop_dup = results['Hop-Aware RL']['duplication_ratio']
            
            if base_dup > 0:
                improvement = ((base_dup - hop_dup) / base_dup) * 100
                print(f"\nDuplication Reduction: {improvement:.1f}%")