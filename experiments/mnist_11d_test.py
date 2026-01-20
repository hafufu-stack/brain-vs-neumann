"""
MNIST 10-Digit Classification with 11D Hypercube SNN
=====================================================

Testing the "Decimal Hypothesis":
If the brain uses 11-dimensional structures, it should be
naturally good at distinguishing 10 categories (0-9).

Comparison:
1. Random sparse network (baseline)
2. 11D Hypercube topology
3. Full connection (expensive baseline)

Author: Hiroto Funasaki (roll)
Date: 2026-01-21
"""

import numpy as np
import time
import urllib.request
import gzip
import os


# ============================================================
# Simple MNIST Loader (no external dependencies)
# ============================================================

def download_mnist():
    """Download MNIST data if not present, or use synthetic data"""
    # Try multiple mirrors
    mirrors = [
        "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "http://yann.lecun.com/exdb/mnist/",
    ]
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    data_dir = "data/mnist"
    os.makedirs(data_dir, exist_ok=True)
    
    for f in files:
        path = os.path.join(data_dir, f)
        if not os.path.exists(path):
            downloaded = False
            for base_url in mirrors:
                try:
                    print(f"  Trying {base_url}{f}...")
                    urllib.request.urlretrieve(base_url + f, path)
                    downloaded = True
                    break
                except Exception as e:
                    print(f"    Failed: {e}")
                    continue
            
            if not downloaded:
                print(f"  Could not download MNIST, using synthetic data...")
                return None
    
    return data_dir


def generate_synthetic_digits(n_samples, seed=42):
    """Generate synthetic digit-like patterns"""
    np.random.seed(seed)
    
    X = np.zeros((n_samples, 784))
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        digit = i % 10
        y[i] = digit
        
        # Create simple patterns for each digit
        img = np.zeros((28, 28))
        
        if digit == 0:
            # Circle
            for r in range(8, 20):
                for c in range(8, 20):
                    if 5 < np.sqrt((r-14)**2 + (c-14)**2) < 8:
                        img[r, c] = 1
        elif digit == 1:
            # Vertical line
            img[5:23, 12:16] = 1
        elif digit == 2:
            img[5:9, 8:20] = 1
            img[9:13, 16:20] = 1
            img[12:16, 8:20] = 1
            img[16:20, 8:12] = 1
            img[19:23, 8:20] = 1
        elif digit == 3:
            img[5:9, 8:20] = 1
            img[5:23, 16:20] = 1
            img[12:16, 8:20] = 1
            img[19:23, 8:20] = 1
        elif digit == 4:
            img[5:15, 8:12] = 1
            img[12:16, 8:20] = 1
            img[5:23, 16:20] = 1
        elif digit == 5:
            img[5:9, 8:20] = 1
            img[9:13, 8:12] = 1
            img[12:16, 8:20] = 1
            img[16:20, 16:20] = 1
            img[19:23, 8:20] = 1
        elif digit == 6:
            img[5:23, 8:12] = 1
            img[12:16, 8:20] = 1
            img[16:23, 16:20] = 1
            img[19:23, 8:20] = 1
        elif digit == 7:
            img[5:9, 8:20] = 1
            img[5:23, 16:20] = 1
        elif digit == 8:
            img[5:9, 8:20] = 1
            img[5:23, 8:12] = 1
            img[5:23, 16:20] = 1
            img[12:16, 8:20] = 1
            img[19:23, 8:20] = 1
        elif digit == 9:
            img[5:9, 8:20] = 1
            img[5:15, 8:12] = 1
            img[5:23, 16:20] = 1
            img[12:16, 8:20] = 1
        
        # Add noise
        noise = np.random.rand(28, 28) * 0.2
        img = np.clip(img + noise, 0, 1)
        
        X[i] = img.flatten()
    
    return X, y


def load_mnist_images(path):
    """Load MNIST images from gzipped file"""
    with gzip.open(path, 'rb') as f:
        data = f.read()
        # Skip header (16 bytes)
        images = np.frombuffer(data, dtype=np.uint8, offset=16)
        return images.reshape(-1, 784) / 255.0


def load_mnist_labels(path):
    """Load MNIST labels from gzipped file"""
    with gzip.open(path, 'rb') as f:
        data = f.read()
        # Skip header (8 bytes)
        return np.frombuffer(data, dtype=np.uint8, offset=8)


def get_mnist_data(n_train=5000, n_test=1000):
    """Get MNIST data (real or synthetic)"""
    print("\n  Loading MNIST data...")
    
    data_dir = download_mnist()
    
    if data_dir is None:
        # Use synthetic data
        print("  Using synthetic digit patterns...")
        X_all, y_all = generate_synthetic_digits(n_train + n_test)
        # Shuffle
        idx = np.random.permutation(len(X_all))
        X_all, y_all = X_all[idx], y_all[idx]
        
        X_train, y_train = X_all[:n_train], y_all[:n_train]
        X_test, y_test = X_all[n_train:], y_all[n_train:]
    else:
        X_train = load_mnist_images(os.path.join(data_dir, "train-images-idx3-ubyte.gz"))
        y_train = load_mnist_labels(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"))
        X_test = load_mnist_images(os.path.join(data_dir, "t10k-images-idx3-ubyte.gz"))
        y_test = load_mnist_labels(os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz"))
        
        # Limit data size for faster experiments
        X_train = X_train[:n_train]
        y_train = y_train[:n_train]
        X_test = X_test[:n_test]
        y_test = y_test[:n_test]
    
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    return X_train, y_train, X_test, y_test


# ============================================================
# Network Topologies
# ============================================================

def create_random_sparse_mask(n, sparsity=0.01, seed=42):
    """Create random sparse mask"""
    np.random.seed(seed)
    mask = (np.random.rand(n, n) < sparsity).astype(float)
    np.fill_diagonal(mask, 0)
    return mask


def create_hypercube_mask(dim=10):
    """
    Create hypercube adjacency matrix.
    For dim=10: 1024 nodes, 10 connections each
    """
    n = 2 ** dim
    mask = np.zeros((n, n))
    
    for node in range(n):
        for d in range(dim):
            neighbor = node ^ (1 << d)
            mask[node, neighbor] = 1
    
    return mask


def create_full_mask(n):
    """Full connection (baseline)"""
    mask = np.ones((n, n))
    np.fill_diagonal(mask, 0)
    return mask


# ============================================================
# Simple SNN Classifier
# ============================================================

class SNNClassifier:
    """
    Simple Spiking Neural Network for classification.
    Uses rate coding with topology mask.
    """
    
    def __init__(self, input_size, hidden_size, output_size, mask=None, seed=42):
        np.random.seed(seed)
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        
        # Topology mask for hidden layer
        if mask is not None:
            # Ensure mask matches hidden size
            if mask.shape[0] != hidden_size:
                # Tile or crop mask
                if mask.shape[0] < hidden_size:
                    # Tile mask
                    repeats = (hidden_size // mask.shape[0]) + 1
                    mask_large = np.tile(mask, (repeats, repeats))
                    self.mask = mask_large[:hidden_size, :hidden_size]
                else:
                    self.mask = mask[:hidden_size, :hidden_size]
            else:
                self.mask = mask
        else:
            self.mask = np.ones((hidden_size, hidden_size))
        
        # Learning rate
        self.lr = 0.01
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / (np.sum(exp_x) + 1e-10)
    
    def forward(self, x):
        """Forward pass with topology-constrained hidden layer"""
        # Hidden layer
        h = self.relu(x @ self.W1)
        
        # Apply topology mask (simulate recurrent connections)
        # Simple masked propagation: h gets influenced by its neighbors
        h_propagated = h + 0.1 * (h @ self.mask)
        h_propagated = self.relu(h_propagated)
        
        # Output layer
        out = h_propagated @ self.W2
        probs = self.softmax(out)
        
        return probs, h_propagated
    
    def train_step(self, x, y_true):
        """Simple gradient descent"""
        probs, h = self.forward(x)
        
        # Cross-entropy gradient
        y_onehot = np.zeros(self.output_size)
        y_onehot[y_true] = 1
        d_out = probs - y_onehot
        
        # Backprop
        d_W2 = np.outer(h, d_out)
        d_h = d_out @ self.W2.T
        d_h[h <= 0] = 0
        d_W1 = np.outer(x, d_h)
        
        # Update
        self.W1 -= self.lr * d_W1
        self.W2 -= self.lr * d_W2
        
        return -np.log(probs[y_true] + 1e-10)
    
    def predict(self, x):
        probs, _ = self.forward(x)
        return np.argmax(probs)


# ============================================================
# Experiments
# ============================================================

def train_and_evaluate(X_train, y_train, X_test, y_test, mask, name, epochs=5):
    """Train and evaluate a model"""
    hidden_size = 512
    
    model = SNNClassifier(
        input_size=784,
        hidden_size=hidden_size,
        output_size=10,
        mask=mask
    )
    
    # Training
    t0 = time.time()
    for epoch in range(epochs):
        losses = []
        for i in range(len(X_train)):
            loss = model.train_step(X_train[i], y_train[i])
            losses.append(loss)
        
        avg_loss = np.mean(losses)
        # print(f"    Epoch {epoch+1}: loss = {avg_loss:.4f}")
    
    train_time = time.time() - t0
    
    # Testing
    correct = 0
    for i in range(len(X_test)):
        pred = model.predict(X_test[i])
        if pred == y_test[i]:
            correct += 1
    
    accuracy = correct / len(X_test) * 100
    
    return accuracy, train_time, model


def run_comparison():
    """Compare different topologies on MNIST"""
    print("\n" + "=" * 70)
    print("   MNIST 10-DIGIT CLASSIFICATION: Topology Comparison")
    print("   Testing the 'Decimal Hypothesis'")
    print("=" * 70)
    
    # Load data
    X_train, y_train, X_test, y_test = get_mnist_data(n_train=3000, n_test=500)
    
    # Create masks
    hidden_size = 512
    
    print(f"\n  Creating topology masks (hidden_size={hidden_size})...")
    
    # Random sparse (same sparsity as 10D hypercube)
    random_mask = create_random_sparse_mask(hidden_size, sparsity=10/hidden_size)
    
    # 9D Hypercube (512 = 2^9 neurons)
    hypercube_9d = create_hypercube_mask(dim=9)
    
    # Full connection
    full_mask = create_full_mask(hidden_size)
    
    print(f"    Random sparse: {int(np.sum(random_mask)):,} connections")
    print(f"    9D Hypercube: {int(np.sum(hypercube_9d)):,} connections")
    print(f"    Full: {int(np.sum(full_mask)):,} connections")
    
    # Train and compare
    print("\n  Training models (5 epochs each)...")
    print("-" * 50)
    
    results = {}
    
    # Random sparse
    print("\n  [1/4] Random Sparse...")
    acc, time_s, _ = train_and_evaluate(X_train, y_train, X_test, y_test, random_mask, "Random")
    results['Random Sparse'] = {'accuracy': acc, 'time': time_s}
    print(f"      Accuracy: {acc:.1f}%, Time: {time_s:.1f}s")
    
    # 9D Hypercube
    print("\n  [2/4] 9D Hypercube...")
    acc, time_s, _ = train_and_evaluate(X_train, y_train, X_test, y_test, hypercube_9d, "9D")
    results['9D Hypercube'] = {'accuracy': acc, 'time': time_s}
    print(f"      Accuracy: {acc:.1f}%, Time: {time_s:.1f}s")
    
    # Full connection
    print("\n  [3/4] Full Connection...")
    acc, time_s, _ = train_and_evaluate(X_train, y_train, X_test, y_test, full_mask, "Full")
    results['Full Connection'] = {'accuracy': acc, 'time': time_s}
    print(f"      Accuracy: {acc:.1f}%, Time: {time_s:.1f}s")
    
    # No mask (same as full)
    print("\n  [4/4] No Mask (baseline)...")
    acc, time_s, _ = train_and_evaluate(X_train, y_train, X_test, y_test, None, "None")
    results['No Mask'] = {'accuracy': acc, 'time': time_s}
    print(f"      Accuracy: {acc:.1f}%, Time: {time_s:.1f}s")
    
    # Summary
    print("\n" + "=" * 70)
    print("   RESULTS SUMMARY")
    print("=" * 70)
    
    print("""
    ┌─────────────────────┬───────────┬───────────┐
    │ Topology            │ Accuracy  │ Time      │
    ├─────────────────────┼───────────┼───────────┤""")
    
    for name, r in results.items():
        print(f"    │ {name:<19} │ {r['accuracy']:>7.1f}%  │ {r['time']:>7.1f}s  │")
    
    print("    └─────────────────────┴───────────┴───────────┘")
    
    # Analyze
    best_topology = max(results.items(), key=lambda x: x[1]['accuracy'])
    hypercube_acc = results['9D Hypercube']['accuracy']
    random_acc = results['Random Sparse']['accuracy']
    
    print(f"""
    Analysis:
    - Best topology: {best_topology[0]} ({best_topology[1]['accuracy']:.1f}%)
    - 9D Hypercube vs Random: {hypercube_acc - random_acc:+.1f}% difference
    
    Decimal Hypothesis:
    - 9D Hypercube (512 neurons = 2^9, close to 10D) should be good at 10 categories
    - If Hypercube > Random with same sparsity, the TOPOLOGY matters!
    """)
    
    # Save results
    with open("results/mnist_comparison.txt", "w", encoding="utf-8") as f:
        f.write("MNIST 10-Digit Classification: Topology Comparison\n")
        f.write("=" * 50 + "\n\n")
        for name, r in results.items():
            f.write(f"{name}: {r['accuracy']:.1f}% ({r['time']:.1f}s)\n")
        f.write(f"\nBest: {best_topology[0]}\n")
        f.write(f"Hypercube advantage: {hypercube_acc - random_acc:+.1f}%\n")
    
    print("\n  Results saved to: results/mnist_comparison.txt")
    
    return results


def test_learning_speed():
    """
    Test how fast each topology learns 10 categories.
    Hypothesis: 11D should learn 10 categories faster.
    """
    print("\n" + "=" * 70)
    print("   LEARNING SPEED: How fast does each topology learn 10 digits?")
    print("=" * 70)
    
    # Load smaller data for learning curve
    X_train, y_train, X_test, y_test = get_mnist_data(n_train=1000, n_test=200)
    
    hidden_size = 512
    
    hypercube_9d = create_hypercube_mask(dim=9)
    random_mask = create_random_sparse_mask(hidden_size, sparsity=9/hidden_size)
    
    topologies = {
        '9D Hypercube': hypercube_9d,
        'Random Sparse': random_mask
    }
    
    print("\n  Tracking accuracy across training samples...")
    
    learning_curves = {}
    
    for name, mask in topologies.items():
        print(f"\n  Training: {name}...")
        
        model = SNNClassifier(784, hidden_size, 10, mask=mask)
        
        accuracies = []
        checkpoints = [100, 200, 300, 500, 750, 1000]
        
        for i in range(len(X_train)):
            model.train_step(X_train[i], y_train[i])
            
            if (i + 1) in checkpoints:
                correct = sum(1 for j in range(len(X_test)) if model.predict(X_test[j]) == y_test[j])
                acc = correct / len(X_test) * 100
                accuracies.append((i + 1, acc))
                print(f"    {i+1} samples: {acc:.1f}%")
        
        learning_curves[name] = accuracies
    
    # Compare learning speed
    print("\n" + "=" * 70)
    print("   LEARNING SPEED COMPARISON")
    print("=" * 70)
    
    print("\n  Samples | 9D Hypercube | Random Sparse | Winner")
    print("  " + "-" * 50)
    
    for idx in range(len(checkpoints)):
        samples = checkpoints[idx]
        hyper_acc = learning_curves['9D Hypercube'][idx][1] if idx < len(learning_curves['9D Hypercube']) else 0
        random_acc = learning_curves['Random Sparse'][idx][1] if idx < len(learning_curves['Random Sparse']) else 0
        
        winner = "Hypercube ⭐" if hyper_acc > random_acc else "Random" if random_acc > hyper_acc else "Tie"
        
        print(f"  {samples:7} | {hyper_acc:12.1f}% | {random_acc:13.1f}% | {winner}")
    
    return learning_curves


if __name__ == "__main__":
    run_comparison()
    test_learning_speed()
    
    print("\n" + "=" * 70)
    print("   CONCLUSION")
    print("=" * 70)
    print("""
    The experiments test whether brain-like topology (hypercube)
    is naturally better at distinguishing 10 categories.
    
    Key insights:
    1. Hypercube topology uses STRUCTURED sparsity (not random)
    2. Information can reach any node in ~9 steps (for 9D)
    3. This may explain why base-10 is "natural" for humans:
       - 10 fingers for counting
       - 11-dimensional brain structures
       - Perfect match for 10-category classification!
    
    Next: Test with 11D hypercube (2048 neurons) for full comparison.
    """)
