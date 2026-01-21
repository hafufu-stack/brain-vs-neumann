"""
Magic Number 10: The Dimensionality Cliff
==========================================

Gemini's Hypothesis:
"If the brain is 11-dimensional, classification performance should 
drop sharply when the number of classes exceeds 11 (the cliff)."

This experiment tests:
- Classification accuracy for 2, 5, 8, 10, 11, 12, 15, 20 classes
- Using 11D hypercube (2048 neurons)
- Prediction: Performance cliff at 12+ classes

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


class SimpleSNN:
    """Simplified SNN for classification"""
    
    def __init__(self, input_dim, hidden_dim, output_dim, mask=None, seed=42):
        np.random.seed(seed)
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Weights
        self.W_in = np.random.randn(hidden_dim, input_dim) * 0.1
        self.W_out = np.random.randn(output_dim, hidden_dim) * 0.1
        
        # Recurrent weights with topology
        W_rec = np.random.randn(hidden_dim, hidden_dim) * 0.1
        if mask is not None:
            # Resize mask if needed
            if mask.shape[0] != hidden_dim:
                ratio = hidden_dim / mask.shape[0]
                new_mask = np.zeros((hidden_dim, hidden_dim))
                for i in range(hidden_dim):
                    for j in range(hidden_dim):
                        orig_i = int(i / ratio) % mask.shape[0]
                        orig_j = int(j / ratio) % mask.shape[0]
                        new_mask[i, j] = mask[orig_i, orig_j]
                mask = new_mask
            W_rec *= mask
        self.W_rec = W_rec
        
        # State
        self.state = np.zeros(hidden_dim)
        
    def forward(self, x):
        """Forward pass with simple LIF dynamics"""
        # Input
        h_in = np.tanh(self.W_in @ x)
        
        # Recurrent (simplified)
        h_rec = np.tanh(self.W_rec @ self.state)
        
        # Combined
        self.state = 0.8 * self.state + 0.2 * (h_in + h_rec)
        
        # Apply threshold (simplified spike)
        spikes = (self.state > 0.5).astype(float)
        self.state = self.state * (1 - spikes * 0.5)  # Reset
        
        # Output
        out = self.W_out @ self.state
        return out
    
    def predict(self, x):
        """Get prediction"""
        self.state = np.zeros(self.hidden_dim)
        for _ in range(3):  # Run for a few steps
            out = self.forward(x)
        return np.argmax(out)
    
    def train_step(self, x, target, lr=0.01):
        """Simple gradient descent"""
        self.state = np.zeros(self.hidden_dim)
        for _ in range(3):
            out = self.forward(x)
        
        # Softmax
        exp_out = np.exp(out - np.max(out))
        probs = exp_out / (np.sum(exp_out) + 1e-10)
        
        # Cross-entropy gradient
        grad = probs.copy()
        grad[target] -= 1
        
        # Update output weights (simplified)
        self.W_out -= lr * np.outer(grad, self.state)
        
        return -np.log(probs[target] + 1e-10)


def generate_classification_data(n_classes, n_samples=500, input_dim=100):
    """Generate synthetic classification data"""
    np.random.seed(42)
    
    X = []
    y = []
    
    for c in range(n_classes):
        # Each class has a distinct pattern
        center = np.random.randn(input_dim)
        for _ in range(n_samples // n_classes):
            sample = center + np.random.randn(input_dim) * 0.5
            X.append(sample)
            y.append(c)
    
    X = np.array(X)
    y = np.array(y)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def train_and_evaluate(args):
    """Train and evaluate a model"""
    n_classes, topology, hidden_dim, seed = args
    
    input_dim = 100
    
    # Create mask
    if topology == 'hypercube':
        dim = 11  # Fixed 11D
        mask = create_hypercube_mask(dim)
    else:
        mask = None
    
    # Generate data
    X, y = generate_classification_data(n_classes, n_samples=500, input_dim=input_dim)
    
    # Split
    n_train = int(len(X) * 0.8)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    
    # Create model
    model = SimpleSNN(input_dim, hidden_dim, n_classes, mask=mask, seed=seed)
    
    # Train
    epochs = 20
    for epoch in range(epochs):
        for i in range(len(X_train)):
            model.train_step(X_train[i], y_train[i], lr=0.01)
    
    # Evaluate
    correct = 0
    for i in range(len(X_test)):
        pred = model.predict(X_test[i])
        if pred == y_test[i]:
            correct += 1
    
    accuracy = correct / len(X_test)
    return n_classes, topology, accuracy


def main():
    print("\n" + "=" * 70)
    print("   MAGIC NUMBER 10: THE DIMENSIONALITY CLIFF")
    print("   Testing if performance drops sharply at 12+ classes")
    print("=" * 70)
    
    # Test different number of classes
    class_counts = [2, 5, 8, 10, 11, 12, 15, 20]
    hidden_dim = 512  # Use 512 neurons (9D scale)
    n_trials = 3
    
    print(f"\n  Neurons: {hidden_dim}, Trials: {n_trials}")
    print("-" * 60)
    
    # Prepare tasks
    tasks = []
    for n_classes in class_counts:
        for trial in range(n_trials):
            tasks.append((n_classes, 'hypercube', hidden_dim, 42 + trial))
            tasks.append((n_classes, 'random', hidden_dim, 100 + trial))
    
    # Run in parallel
    print(f"\n  Running {len(tasks)} experiments...")
    t0 = time.time()
    
    n_workers = min(12, cpu_count())
    with Pool(n_workers) as pool:
        results = pool.map(train_and_evaluate, tasks)
    
    print(f"  Completed in {time.time() - t0:.1f}s")
    
    # Aggregate results
    hypercube_results = {n: [] for n in class_counts}
    random_results = {n: [] for n in class_counts}
    
    for n_classes, topology, accuracy in results:
        if topology == 'hypercube':
            hypercube_results[n_classes].append(accuracy)
        else:
            random_results[n_classes].append(accuracy)
    
    # Print results
    print("\n" + "=" * 70)
    print("   RESULTS: Classification Accuracy by Number of Classes")
    print("=" * 70)
    
    print(f"\n  {'Classes':>7} | {'Hypercube':>12} | {'Random':>12} | {'Difference':>10}")
    print("  " + "-" * 55)
    
    hypercube_accs = []
    random_accs = []
    
    for n in class_counts:
        h_acc = np.mean(hypercube_results[n]) * 100
        r_acc = np.mean(random_results[n]) * 100
        diff = h_acc - r_acc
        
        hypercube_accs.append(h_acc)
        random_accs.append(r_acc)
        
        cliff = "⚠️ CLIFF?" if n > 11 and h_acc < hypercube_accs[-2] * 0.9 else ""
        print(f"  {n:>7} | {h_acc:>10.1f}% | {r_acc:>10.1f}% | {diff:>+9.1f}% {cliff}")
    
    # Check for cliff
    print("\n" + "=" * 70)
    print("   CLIFF ANALYSIS")
    print("=" * 70)
    
    # Find performance drop
    max_acc_idx = np.argmax(hypercube_accs)
    max_acc_classes = class_counts[max_acc_idx]
    
    print(f"""
    Peak performance: {class_counts[max_acc_idx]} classes ({hypercube_accs[max_acc_idx]:.1f}%)
    
    Performance at key points:
    - 10 classes: {hypercube_accs[class_counts.index(10)]:.1f}%
    - 11 classes: {hypercube_accs[class_counts.index(11)]:.1f}%
    - 12 classes: {hypercube_accs[class_counts.index(12)]:.1f}%
    
    Drop from 11 → 12: {hypercube_accs[class_counts.index(11)] - hypercube_accs[class_counts.index(12)]:.1f}%
    """)
    
    if hypercube_accs[class_counts.index(12)] < hypercube_accs[class_counts.index(11)] * 0.95:
        print("    ✅ CLIFF DETECTED: Performance drops at 12 classes!")
        print("    → Supports hypothesis: 11D brain optimal for ≤11 categories")
    else:
        print("    ❌ No significant cliff detected")
        print("    → May need more neurons or different task design")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(class_counts, hypercube_accs, 'b-o', label='11D Hypercube', linewidth=2)
    plt.plot(class_counts, random_accs, 'r--s', label='Random Sparse', linewidth=2)
    plt.axvline(x=11, color='green', linestyle=':', label='11D Boundary')
    plt.xlabel('Number of Classes', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Magic Number 10: Does Performance Cliff at 12 Classes?', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('results/fig_dimensionality_cliff.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n  Figure saved: results/fig_dimensionality_cliff.png")
    
    # Save results
    with open("results/dimensionality_cliff.txt", "w", encoding="utf-8") as f:
        f.write("Dimensionality Cliff Experiment\n")
        f.write("=" * 40 + "\n\n")
        for i, n in enumerate(class_counts):
            f.write(f"{n} classes: Hypercube={hypercube_accs[i]:.1f}%, Random={random_accs[i]:.1f}%\n")
    
    print("  Results saved: results/dimensionality_cliff.txt")
    
    return hypercube_accs, random_accs


if __name__ == "__main__":
    main()
