"""
Base System Generalization: N-1 Dimension Hypothesis
=====================================================

Sonnet's Hypothesis:
"For n-base number system, (n-1) dimensional hypercube is optimal."

This experiment tests:
- Base-2 (binary) → 1D optimal?
- Base-3 (ternary) → 2D optimal?
- Base-5 (quinary) → 4D optimal?
- Base-6 (senary) → 5D optimal?
- Base-8 (octal) → 7D optimal?
- Base-10 (decimal) → 9D optimal?
- Base-12 (duodecimal) → 11D optimal? (requires 2048 neurons)
- Base-16 (hexadecimal) → 15D optimal? (too large, approximate)

Author: Hiroto Funasaki (roll)
Date: 2026-01-21
"""

import numpy as np
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import time

plt.rcParams["font.family"] = "MS Gothic"


def create_hypercube_mask(dim, target_size=None):
    """Create hypercube adjacency matrix, optionally resize"""
    n = 2 ** dim
    mask = np.zeros((n, n))
    for node in range(n):
        for d in range(dim):
            neighbor = node ^ (1 << d)
            mask[node, neighbor] = 1
    
    if target_size and target_size != n:
        # Resize mask to target size
        new_mask = np.zeros((target_size, target_size))
        for i in range(target_size):
            for j in range(target_size):
                orig_i = (i * n) // target_size
                orig_j = (j * n) // target_size
                new_mask[i, j] = mask[orig_i, orig_j]
        return new_mask
    
    return mask


class HypercubeSNN:
    """SNN with hypercube topology for classification"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, hypercube_dim, seed=42):
        np.random.seed(seed)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Create hypercube mask and resize to hidden_dim
        self.mask = create_hypercube_mask(hypercube_dim, target_size=hidden_dim)
        
        # Weights
        self.W_in = np.random.randn(hidden_dim, input_dim) * 0.1
        self.W_out = np.random.randn(output_dim, hidden_dim) * 0.1
        
        W_rec = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.W_rec = W_rec * self.mask
        
        self.state = np.zeros(hidden_dim)
        
    def forward(self, x):
        h_in = np.tanh(self.W_in @ x)
        h_rec = np.tanh(self.W_rec @ self.state)
        self.state = 0.8 * self.state + 0.2 * (h_in + h_rec)
        
        spikes = (self.state > 0.5).astype(float)
        self.state = self.state * (1 - spikes * 0.5)
        
        return self.W_out @ self.state
    
    def predict(self, x):
        self.state = np.zeros(self.hidden_dim)
        for _ in range(5):
            out = self.forward(x)
        return np.argmax(out)
    
    def train_step(self, x, target, lr=0.01):
        self.state = np.zeros(self.hidden_dim)
        for _ in range(5):
            out = self.forward(x)
        
        exp_out = np.exp(out - np.max(out))
        probs = exp_out / (np.sum(exp_out) + 1e-10)
        
        grad = probs.copy()
        grad[target] -= 1
        
        self.W_out -= lr * np.outer(grad, self.state)
        
        return -np.log(probs[target] + 1e-10)


def generate_base_data(n_classes, n_samples=1000, input_dim=256, noise_level=0.8):
    """Generate challenging classification data"""
    np.random.seed(42)
    
    X = []
    y = []
    
    # Create overlapping class centers
    centers = []
    for c in range(n_classes):
        center = np.random.randn(input_dim)
        centers.append(center)
    
    for c in range(n_classes):
        for _ in range(n_samples // n_classes):
            # High noise for challenging task
            noise = np.random.randn(input_dim) * noise_level
            sample = centers[c] + noise
            X.append(sample)
            y.append(c)
    
    X = np.array(X)
    y = np.array(y)
    
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def run_base_experiment(args):
    """Train and evaluate model for a specific base system"""
    n_classes, hypercube_dim, hidden_dim, seed = args
    
    input_dim = 256
    
    # Generate data
    X, y = generate_base_data(n_classes, n_samples=500, input_dim=input_dim, noise_level=0.8)
    
    n_train = int(len(X) * 0.8)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    
    # Create model
    model = HypercubeSNN(input_dim, hidden_dim, n_classes, hypercube_dim, seed=seed)
    
    # Train
    epochs = 30
    for epoch in range(epochs):
        for i in range(len(X_train)):
            model.train_step(X_train[i], y_train[i], lr=0.02)
    
    # Evaluate
    correct = sum(model.predict(X_test[i]) == y_test[i] for i in range(len(X_test)))
    accuracy = correct / len(X_test)
    
    return n_classes, hypercube_dim, accuracy


def main():
    print("\n" + "=" * 70)
    print("   BASE SYSTEM GENERALIZATION: N-1 DIMENSION HYPOTHESIS")
    print("   Testing if n-base → (n-1)-dimensional hypercube is optimal")
    print("=" * 70)
    
    # Base systems to test
    base_systems = [
        (2, "Binary"),
        (3, "Ternary"),
        (5, "Quinary"),
        (6, "Senary"),
        (8, "Octal"),
        (10, "Decimal"),
        (12, "Duodecimal"),
        (16, "Hexadecimal"),
    ]
    
    # Corresponding hypercube dimensions to test
    # For each base, we test optimal_dim = base - 1, and some alternatives
    hidden_dim = 256
    n_trials = 3
    
    print(f"\n  Hidden neurons: {hidden_dim}")
    print(f"  Trials per experiment: {n_trials}")
    print("-" * 60)
    
    # Collect results
    all_results = {}
    
    for n_classes, base_name in base_systems:
        print(f"\n  Testing Base-{n_classes} ({base_name})...")
        
        # Optimal dimension according to hypothesis
        optimal_dim = n_classes - 1
        
        # Test dimensions around optimal (ensure at least 3 dims)
        # Cap optimal_dim at 11 since we can't go higher
        optimal_dim_capped = min(optimal_dim, 11)
        min_dim = max(1, optimal_dim_capped - 2)
        max_dim = min(11, optimal_dim_capped + 2)
        if max_dim < min_dim:
            min_dim = max(1, max_dim - 2)
        test_dims = list(range(min_dim, max_dim + 1))
        
        # Prepare tasks
        tasks = []
        for dim in test_dims:
            for trial in range(n_trials):
                tasks.append((n_classes, dim, hidden_dim, 42 + trial))
        
        # Run
        n_workers = min(8, cpu_count())
        with Pool(n_workers) as pool:
            results = pool.map(run_base_experiment, tasks)
        
        # Aggregate
        dim_results = {dim: [] for dim in test_dims}
        for _, dim, acc in results:
            dim_results[dim].append(acc)
        
        # Find best dimension
        best_dim = max(test_dims, key=lambda d: np.mean(dim_results[d]))
        best_acc = np.mean(dim_results[best_dim]) * 100
        
        all_results[n_classes] = {
            'name': base_name,
            'optimal_hypothesis': optimal_dim,
            'best_actual': best_dim,
            'dim_results': {d: np.mean(dim_results[d]) * 100 for d in test_dims}
        }
        
        match = "✅" if best_dim == optimal_dim else "❌"
        print(f"    Hypothesis: {optimal_dim}D | Best: {best_dim}D ({best_acc:.1f}%) {match}")
    
    # Summary
    print("\n" + "=" * 70)
    print("   SUMMARY: N-1 DIMENSION HYPOTHESIS")
    print("=" * 70)
    
    print(f"\n  {'Base':>12} | {'Hypothesis':>10} | {'Best':>6} | {'Match':>6}")
    print("  " + "-" * 45)
    
    matches = 0
    for n_classes, base_name in base_systems:
        r = all_results[n_classes]
        match = "✅" if r['best_actual'] == r['optimal_hypothesis'] else "❌"
        if r['best_actual'] == r['optimal_hypothesis']:
            matches += 1
        print(f"  Base-{n_classes:>2} ({base_name:>10}) | {r['optimal_hypothesis']:>8}D | {r['best_actual']:>4}D | {match}")
    
    print(f"\n  Hypothesis Match Rate: {matches}/{len(base_systems)} ({matches/len(base_systems)*100:.0f}%)")
    
    if matches >= len(base_systems) * 0.6:
        print("\n  ✅ HYPOTHESIS SUPPORTED: n-base → (n-1)D is generally optimal!")
    else:
        print("\n  ⚠️ Hypothesis partially supported, needs more investigation")
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    for n_classes, base_name in base_systems[:6]:  # Plot first 6 for clarity
        r = all_results[n_classes]
        dims = sorted(r['dim_results'].keys())
        accs = [r['dim_results'][d] for d in dims]
        plt.plot(dims, accs, 'o-', label=f'Base-{n_classes}', linewidth=2)
        
        # Mark hypothesized optimal
        opt_dim = r['optimal_hypothesis']
        if opt_dim in dims:
            plt.scatter([opt_dim], [r['dim_results'][opt_dim]], s=200, marker='*', zorder=5)
    
    plt.xlabel('Hypercube Dimension', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Base System vs Optimal Hypercube Dimension', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig('results/fig_base_system.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n  Figure saved: results/fig_base_system.png")
    
    # Save results
    with open("results/base_system_results.txt", "w", encoding="utf-8") as f:
        f.write("Base System Generalization Results\n")
        f.write("=" * 40 + "\n\n")
        for n_classes, base_name in base_systems:
            r = all_results[n_classes]
            f.write(f"Base-{n_classes} ({base_name}):\n")
            f.write(f"  Hypothesis: {r['optimal_hypothesis']}D\n")
            f.write(f"  Best: {r['best_actual']}D\n")
            f.write(f"  Accuracies: {r['dim_results']}\n\n")
    
    print("  Results saved: results/base_system_results.txt")
    
    return all_results


if __name__ == "__main__":
    main()
