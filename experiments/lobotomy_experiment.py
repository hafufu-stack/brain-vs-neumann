"""
Lobotomy Experiment: Fault Tolerance Test
==========================================

Gemini's Hypothesis:
"Brain-like systems have bypass circuits, so they should maintain
performance even when 10-50% of neurons are destroyed."

This experiment tests:
- Classification accuracy after destroying 10%, 20%, 30%, 40%, 50% of neurons
- Comparison: Hypercube (structured) vs Random (unstructured) vs Full Connection
- Prediction: Hypercube degrades gracefully, Full Connection crashes

Author: Hiroto Funasaki (roll)
Date: 2026-01-21
"""

import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import time

plt.rcParams["font.family"] = "MS Gothic"


def create_hypercube_mask(dim):
    """Create hypercube adjacency matrix"""
    n = 2 ** dim
    mask = np.zeros((n, n))
    for node in range(n):
        for d in range(dim):
            neighbor = node ^ (1 << d)
            mask[node, neighbor] = 1
    return mask


class RobustSNN:
    """SNN with ability to destroy neurons"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, topology='hypercube', seed=42):
        np.random.seed(seed)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.topology = topology
        
        # Weights
        self.W_in = np.random.randn(hidden_dim, input_dim) * 0.1
        self.W_out = np.random.randn(output_dim, hidden_dim) * 0.1
        
        # Recurrent weights with topology
        W_rec = np.random.randn(hidden_dim, hidden_dim) * 0.1
        
        if topology == 'hypercube':
            dim = int(np.log2(hidden_dim))
            mask = create_hypercube_mask(dim)
            W_rec *= mask
        elif topology == 'random':
            density = 11 / hidden_dim  # Same sparsity as hypercube
            mask = (np.random.rand(hidden_dim, hidden_dim) < density).astype(float)
            W_rec *= mask
        # else: full connection
        
        self.W_rec = W_rec
        self.state = np.zeros(hidden_dim)
        self.alive = np.ones(hidden_dim, dtype=bool)  # All neurons alive initially
        
    def destroy_neurons(self, fraction):
        """Destroy a fraction of neurons"""
        n_destroy = int(self.hidden_dim * fraction)
        destroy_idx = np.random.choice(self.hidden_dim, n_destroy, replace=False)
        self.alive[destroy_idx] = False
        
        # Zero out weights for dead neurons
        self.W_in[destroy_idx, :] = 0
        self.W_out[:, destroy_idx] = 0
        self.W_rec[destroy_idx, :] = 0
        self.W_rec[:, destroy_idx] = 0
        
    def forward(self, x):
        """Forward pass with dead neurons masked"""
        # Input
        h_in = np.tanh(self.W_in @ x)
        h_in[~self.alive] = 0  # Dead neurons produce nothing
        
        # Recurrent
        h_rec = np.tanh(self.W_rec @ self.state)
        h_rec[~self.alive] = 0
        
        # Combined
        self.state = 0.8 * self.state + 0.2 * (h_in + h_rec)
        self.state[~self.alive] = 0  # Dead neurons stay dead
        
        # Threshold
        spikes = (self.state > 0.5).astype(float)
        self.state = self.state * (1 - spikes * 0.5)
        
        # Output
        out = self.W_out @ self.state
        return out
    
    def predict(self, x):
        """Get prediction"""
        self.state = np.zeros(self.hidden_dim)
        for _ in range(3):
            out = self.forward(x)
        return np.argmax(out)
    
    def train_step(self, x, target, lr=0.01):
        """Simple gradient descent"""
        self.state = np.zeros(self.hidden_dim)
        for _ in range(3):
            out = self.forward(x)
        
        exp_out = np.exp(out - np.max(out))
        probs = exp_out / (np.sum(exp_out) + 1e-10)
        
        grad = probs.copy()
        grad[target] -= 1
        
        self.W_out -= lr * np.outer(grad, self.state)
        
        return -np.log(probs[target] + 1e-10)


def generate_mnist_like_data(n_classes=10, n_samples=1000, input_dim=784):
    """Generate MNIST-like classification data"""
    np.random.seed(42)
    
    X = []
    y = []
    
    for c in range(n_classes):
        # Each class has a distinct pattern (like digit shape)
        pattern = np.zeros(input_dim)
        # Create a "digit-like" pattern
        start = c * (input_dim // n_classes)
        pattern[start:start + input_dim // n_classes] = 1.0
        
        for _ in range(n_samples // n_classes):
            # Add noise
            noise = np.random.randn(input_dim) * 0.3
            sample = pattern + noise
            X.append(sample)
            y.append(c)
    
    X = np.array(X)
    y = np.array(y)
    
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def run_lobotomy_experiment(args):
    """Train model, destroy neurons, test"""
    topology, destroy_fraction, seed = args
    
    input_dim = 784
    hidden_dim = 512  # 2^9
    n_classes = 10
    
    # Generate data
    X, y = generate_mnist_like_data(n_classes, n_samples=800, input_dim=input_dim)
    
    n_train = int(len(X) * 0.8)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    
    # Create model
    model = RobustSNN(input_dim, hidden_dim, n_classes, topology=topology, seed=seed)
    
    # Train with all neurons alive
    epochs = 15
    for epoch in range(epochs):
        for i in range(len(X_train)):
            model.train_step(X_train[i], y_train[i], lr=0.01)
    
    # Test before destruction
    correct_before = sum(model.predict(X_test[i]) == y_test[i] for i in range(len(X_test)))
    acc_before = correct_before / len(X_test)
    
    # Destroy neurons
    if destroy_fraction > 0:
        model.destroy_neurons(destroy_fraction)
    
    # Test after destruction
    correct_after = sum(model.predict(X_test[i]) == y_test[i] for i in range(len(X_test)))
    acc_after = correct_after / len(X_test)
    
    return topology, destroy_fraction, acc_before, acc_after


def main():
    print("\n" + "=" * 70)
    print("   LOBOTOMY EXPERIMENT: FAULT TOLERANCE TEST")
    print("   Testing graceful degradation when neurons are destroyed")
    print("=" * 70)
    
    destroy_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    topologies = ['hypercube', 'random', 'full']
    n_trials = 3
    
    print(f"\n  Topologies: {topologies}")
    print(f"  Destruction rates: {[f'{f*100:.0f}%' for f in destroy_fractions]}")
    print("-" * 60)
    
    # Prepare tasks
    tasks = []
    for topo in topologies:
        for frac in destroy_fractions:
            for trial in range(n_trials):
                tasks.append((topo, frac, 42 + trial))
    
    # Run in parallel
    print(f"\n  Running {len(tasks)} experiments...")
    t0 = time.time()
    
    n_workers = min(12, cpu_count())
    with Pool(n_workers) as pool:
        results = pool.map(run_lobotomy_experiment, tasks)
    
    print(f"  Completed in {time.time() - t0:.1f}s")
    
    # Aggregate results
    results_dict = {topo: {frac: [] for frac in destroy_fractions} for topo in topologies}
    
    for topo, frac, acc_before, acc_after in results:
        results_dict[topo][frac].append(acc_after)
    
    # Print results
    print("\n" + "=" * 70)
    print("   RESULTS: Accuracy After Neuron Destruction")
    print("=" * 70)
    
    print(f"\n  {'Destruction':>12} | {'Hypercube':>12} | {'Random':>12} | {'Full':>12}")
    print("  " + "-" * 55)
    
    hypercube_accs = []
    random_accs = []
    full_accs = []
    
    for frac in destroy_fractions:
        h_acc = np.mean(results_dict['hypercube'][frac]) * 100
        r_acc = np.mean(results_dict['random'][frac]) * 100
        f_acc = np.mean(results_dict['full'][frac]) * 100
        
        hypercube_accs.append(h_acc)
        random_accs.append(r_acc)
        full_accs.append(f_acc)
        
        winner = "🏆" if h_acc >= r_acc and h_acc >= f_acc else ""
        print(f"  {frac*100:>10.0f}% | {h_acc:>10.1f}% | {r_acc:>10.1f}% | {f_acc:>10.1f}% {winner}")
    
    # Analysis
    print("\n" + "=" * 70)
    print("   FAULT TOLERANCE ANALYSIS")
    print("=" * 70)
    
    # Graceful degradation score (area under curve)
    auc_hypercube = np.mean(hypercube_accs)
    auc_random = np.mean(random_accs)
    auc_full = np.mean(full_accs)
    
    # Retention at 50% destruction
    ret_50_hypercube = hypercube_accs[-1] / hypercube_accs[0] * 100 if hypercube_accs[0] > 0 else 0
    ret_50_random = random_accs[-1] / random_accs[0] * 100 if random_accs[0] > 0 else 0
    ret_50_full = full_accs[-1] / full_accs[0] * 100 if full_accs[0] > 0 else 0
    
    print(f"""
    Graceful Degradation Score (average accuracy):
    - Hypercube: {auc_hypercube:.1f}%
    - Random:    {auc_random:.1f}%
    - Full:      {auc_full:.1f}%
    
    Retention at 50% destruction:
    - Hypercube: {ret_50_hypercube:.1f}% of original
    - Random:    {ret_50_random:.1f}% of original
    - Full:      {ret_50_full:.1f}% of original
    """)
    
    if ret_50_hypercube > ret_50_random and ret_50_hypercube > ret_50_full:
        print("    ✅ HYPERCUBE WINS: Best fault tolerance!")
        print("    → Structured connections provide more bypass routes")
    elif ret_50_hypercube > ret_50_full:
        print("    ⚠️ Hypercube better than Full, comparable to Random")
    else:
        print("    ❌ Unexpected: Full or Random has better tolerance")
    
    # Plot
    plt.figure(figsize=(10, 6))
    x = [f * 100 for f in destroy_fractions]
    plt.plot(x, hypercube_accs, 'b-o', label='Hypercube (Structured)', linewidth=2)
    plt.plot(x, random_accs, 'r--s', label='Random Sparse', linewidth=2)
    plt.plot(x, full_accs, 'g:^', label='Full Connection', linewidth=2)
    plt.xlabel('Neurons Destroyed (%)', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Lobotomy Experiment: Graceful Degradation', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('results/fig_lobotomy.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n  Figure saved: results/fig_lobotomy.png")
    
    # Save results
    with open("results/lobotomy_results.txt", "w", encoding="utf-8") as f:
        f.write("Lobotomy Experiment Results\n")
        f.write("=" * 40 + "\n\n")
        for i, frac in enumerate(destroy_fractions):
            f.write(f"{frac*100:.0f}% destroyed:\n")
            f.write(f"  Hypercube: {hypercube_accs[i]:.1f}%\n")
            f.write(f"  Random: {random_accs[i]:.1f}%\n")
            f.write(f"  Full: {full_accs[i]:.1f}%\n\n")
    
    print("  Results saved: results/lobotomy_results.txt")
    
    return results_dict


if __name__ == "__main__":
    main()
