import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import defaultdict
import random

class HopAwareGossipNode:
    def __init__(self, node_id, params):
        self.id = node_id
        self.params = params
        self.scores = defaultdict(lambda: 0.5)  # (neighbor, hop) -> score
        self.messages_received = set()
        self.forwarding_history = []
        self.learning_rates = defaultdict(lambda: params['alpha'])
        
    def receive_message(self, message, sender, current_hop):
        """Receive a message and decide whether to forward"""
        message_id, hop_count = message
        
        # Check if duplicate
        is_duplicate = message_id in self.messages_received
        self.messages_received.add(message_id)
        
        # Learning: update scores based on duplicate detection
        if sender is not None:
            future_hop = hop_count + 1
            self.update_scores(sender, future_hop, is_duplicate)
        
        # Forwarding decision
        if hop_count < self.params['h_max']:
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
        new_score = max(0.0, min(1.0, new_score))  # Clamp to [0,1]
        
        self.scores[key] = new_score
        
        # Adaptive learning rate decay
        self.learning_rates[key] *= 0.95
        
    def decide_forwarding(self, message_id, current_hop):
        """Decide which neighbors to forward to"""
        future_hop = current_hop + 1
        neighbors = list(self.params['network'].neighbors(self.id))
        
        if not neighbors:
            return
            
        # Calculate forwarding probabilities using softmax
        scores = [self.scores[(neighbor, future_hop)] for neighbor in neighbors]
        exp_scores = [np.exp(self.params['beta'] * score) for score in scores]
        total_exp = sum(exp_scores)
        
        probabilities = [exp / total_exp for exp in exp_scores]
        
        # Decide forwarding for each neighbor
        for neighbor, prob in zip(neighbors, probabilities):
            if random.random() < prob:
                self.forwarding_history.append({
                    'message_id': message_id,
                    'to_neighbor': neighbor,
                    'hop_count': future_hop,
                    'timestamp': time.time()
                })

class GossipProtocol:
    def __init__(self, network, protocol_type='hop_aware'):
        self.network = network
        self.protocol_type = protocol_type
        self.nodes = {}
        self.message_counter = 0
        self.results = []
        
        # Protocol parameters
        self.params = {
            'R_succ': 0.3,      # Reward for successful delivery
            'R_dup': -0.4,      # Penalty for duplicate
            'alpha': 0.6,       # Learning rate
            'beta': 2.0,        # Exploration parameter
            'h_max': 4,         # Maximum hops
            'network': network
        }
        
        # Initialize nodes
        for node_id in network.nodes():
            if protocol_type == 'hop_aware':
                self.nodes[node_id] = HopAwareGossipNode(node_id, self.params)
            else:
                self.nodes[node_id] = HopAwareGossipNode(node_id, self.params)
    
    def create_message(self, source_node):
        """Create a new message from source node"""
        self.message_counter += 1
        return (self.message_counter, 0)  # (message_id, hop_count)
    
    def run_round(self, source_node):
        """Run one gossip round starting from source node"""
        message = self.create_message(source_node)
        
        # Initialize propagation
        source = self.nodes[source_node]
        source.receive_message(message, None, 0)
        
        # Process forwarding decisions
        propagation_rounds = 0
        total_duplicates = 0
        total_transmissions = 0
        
        while any(node.forwarding_history for node in self.nodes.values()):
            propagation_rounds += 1
            current_forwardings = []
            
            # Collect all pending forwardings
            for node in self.nodes.values():
                current_forwardings.extend(node.forwarding_history)
                node.forwarding_history = []
            
            # Execute forwardings
            for forwarding in current_forwardings:
                total_transmissions += 1
                sender = forwarding['to_neighbor']
                message_id, hop_count = forwarding['message_id'], forwarding['hop_count']
                message = (message_id, hop_count)
                
                # Send to all neighbors of the sender (simplified)
                sender_node = self.nodes[sender]
                for neighbor in self.network.neighbors(sender):
                    if neighbor != sender:  # Avoid self-loops
                        is_dup = self.nodes[neighbor].receive_message(
                            message, sender, hop_count
                        )
                        if is_dup:
                            total_duplicates += 1
            
            # Break if taking too long
            if propagation_rounds > 20:
                break
        
        # Calculate coverage
        coverage = self.calculate_coverage(message[0])
        
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
        
        avg_coverage = np.mean([r['coverage'] for r in self.results])
        avg_duplicates = np.mean([r['duplicates'] for r in self.results])
        avg_transmissions = np.mean([r['transmissions'] for r in self.results])
        avg_rounds = np.mean([r['propagation_rounds'] for r in self.results])
        
        return {
            'avg_coverage': avg_coverage,
            'avg_duplicates': avg_duplicates,
            'avg_transmissions': avg_transmissions,
            'avg_rounds': avg_rounds,
            'duplication_ratio': avg_duplicates / avg_transmissions if avg_transmissions > 0 else 0
        }

class BaselineProtocol(GossipProtocol):
    """Baseline protocol using simple probabilistic flooding"""
    def __init__(self, network, forward_prob=0.7):
        super().__init__(network, 'baseline')
        self.forward_prob = forward_prob
    
    def decide_forwarding(self, message_id, current_hop):
        """Baseline: simple probabilistic forwarding"""
        neighbors = list(self.params['network'].neighbors(self.id))
        
        for neighbor in neighbors:
            if random.random() < self.forward_prob:
                self.forwarding_history.append({
                    'message_id': message_id,
                    'to_neighbor': neighbor,
                    'hop_count': current_hop + 1,
                    'timestamp': time.time()
                })

def create_ba_network(n=10, m=3):
    """Create Barabasi-Albert network with 10 nodes"""
    return nx.barabasi_albert_graph(n, m)

def run_comparison_experiment():
    """Run comparison between Hop-Aware and Baseline protocols"""
    # Create identical network for both protocols
    ba_network = create_ba_network(10, 2)
    
    print("Network Structure:")
    print(f"Nodes: {list(ba_network.nodes())}")
    print(f"Edges: {list(ba_network.edges())}")
    print(f"Average Degree: {np.mean([d for n, d in ba_network.degree()]):.2f}")
    print()
    
    # Initialize protocols
    hop_aware_protocol = GossipProtocol(ba_network, 'hop_aware')
    baseline_protocol = BaselineProtocol(ba_network, forward_prob=0.7)
    
    protocols = {
        'Hop-Aware RL': hop_aware_protocol,
        'Baseline Flooding': baseline_protocol
    }
    
    # Run multiple gossip rounds
    num_rounds = 15
    results = {}
    
    for protocol_name, protocol in protocols.items():
        print(f"Running {protocol_name}...")
        round_results = []
        
        for round_num in range(num_rounds):
            # Random source node for each round
            source_node = random.choice(list(ba_network.nodes()))
            result = protocol.run_round(source_node)
            result['round'] = round_num + 1
            round_results.append(result)
            
            if round_num < 5:  # Show first 5 rounds in detail
                print(f"  Round {round_num + 1}: Coverage={result['coverage']:.2f}, "
                      f"Duplicates={result['duplicates']}, Transmissions={result['transmissions']}")
        
        protocol.results = round_results
        results[protocol_name] = protocol.get_network_efficiency()
    
    return results, protocols, ba_network

def plot_results(results, protocols):
    """Plot comparative results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Prepare data
    protocol_names = list(results.keys())
    coverage = [results[name]['avg_coverage'] for name in protocol_names]
    duplicates = [results[name]['avg_duplicates'] for name in protocol_names]
    duplication_ratio = [results[name]['duplication_ratio'] for name in protocol_names]
    transmissions = [results[name]['avg_transmissions'] for name in protocol_names]
    
    # Plot 1: Coverage comparison
    bars1 = ax1.bar(protocol_names, coverage, color=['skyblue', 'lightcoral'])
    ax1.set_ylabel('Average Coverage')
    ax1.set_title('Message Coverage Comparison')
    ax1.set_ylim(0, 1)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Plot 2: Duplicate messages
    bars2 = ax2.bar(protocol_names, duplicates, color=['skyblue', 'lightcoral'])
    ax2.set_ylabel('Average Duplicate Messages')
    ax2.set_title('Duplicate Messages Comparison')
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')
    
    # Plot 3: Duplication ratio
    bars3 = ax3.bar(protocol_names, duplication_ratio, color=['skyblue', 'lightcoral'])
    ax3.set_ylabel('Duplication Ratio')
    ax3.set_title('Efficiency: Duplication Ratio')
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Plot 4: Learning progression
    for protocol_name, protocol in protocols.items():
        coverage_progression = [r['coverage'] for r in protocol.results]
        ax4.plot(range(1, len(coverage_progression) + 1), coverage_progression, 
                marker='o', label=protocol_name)
    ax4.set_xlabel('Round Number')
    ax4.set_ylabel('Coverage')
    ax4.set_title('Learning Progression Over Rounds')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_learned_scores(protocol, network):
    """Analyze what the Hop-Aware protocol learned"""
    print("\n" + "="*50)
    print("HOP-AWARE PROTOCOL LEARNING ANALYSIS")
    print("="*50)
    
    hop_aware_node = protocol.nodes[0]  # Look at first node
    
    print("\nLearned Scores for Node 0:")
    print("Format: (neighbor, hop) -> score")
    
    # Group scores by hop
    scores_by_hop = defaultdict(dict)
    for (neighbor, hop), score in hop_aware_node.scores.items():
        scores_by_hop[hop][neighbor] = score
    
    for hop in sorted(scores_by_hop.keys()):
        print(f"\nHop {hop}:")
        for neighbor, score in sorted(scores_by_hop[hop].items()):
            status = "↑ PREFERRED" if score > 0.6 else "↓ AVOIDED" if score < 0.4 else "○ NEUTRAL"
            print(f"  → Node {neighbor}: {score:.3f} {status}")
    
    # Calculate learning effectiveness
    initial_transmissions = protocol.results[0]['transmissions']
    final_transmissions = protocol.results[-1]['transmissions']
    reduction = ((initial_transmissions - final_transmissions) / initial_transmissions) * 100
    
    print(f"\nLearning Effectiveness:")
    print(f"Initial transmissions per message: {initial_transmissions:.1f}")
    print(f"Final transmissions per message: {final_transmissions:.1f}")
    print(f"Reduction: {reduction:.1f}%")

# Run the complete experiment
if __name__ == "__main__":
    print("HOP-AWARE REINFORCEMENT GOSSIP PROTOCOL EXPERIMENT")
    print("Network: Barabasi-Albert (10 nodes, m=2)")
    print("="*60)
    
    results, protocols, network = run_comparison_experiment()
    
    # Print final results
    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)
    
    for protocol_name, metrics in results.items():
        print(f"\n{protocol_name}:")
        print(f"  Average Coverage: {metrics['avg_coverage']:.3f}")
        print(f"  Average Duplicates: {metrics['avg_duplicates']:.1f}")
        print(f"  Average Transmissions: {metrics['avg_transmissions']:.1f}")
        print(f"  Duplication Ratio: {metrics['duplication_ratio']:.3f}")
        print(f"  Average Propagation Rounds: {metrics['avg_rounds']:.1f}")
    
    # Calculate improvements
    base_dup = results['Baseline Flooding']['duplication_ratio']
    hop_dup = results['Hop-Aware RL']['duplication_ratio']
    improvement = ((base_dup - hop_dup) / base_dup) * 100
    
    base_trans = results['Baseline Flooding']['avg_transmissions']
    hop_trans = results['Hop-Aware RL']['avg_transmissions']
    trans_improvement = ((base_trans - hop_trans) / base_trans) * 100
    
    print(f"\nIMPROVEMENT SUMMARY:")
    print(f"Duplication Reduction: {improvement:.1f}%")
    print(f"Transmission Reduction: {trans_improvement:.1f}%")
    print(f"Coverage Maintained: {results['Hop-Aware RL']['avg_coverage']:.3f} vs {results['Baseline Flooding']['avg_coverage']:.3f}")
    
    # Analyze learned behavior
    analyze_learned_scores(protocols['Hop-Aware RL'], network)
    
    # Plot results
    plot_results(results, protocols)