# Brain vs Von Neumann Computer Simulation

🧠 **Von Neumann Architecture vs Brain-like Architecture: A Comprehensive Study**

## Overview

This project tests two hypotheses:

> 1. "The brain encodes more information by using the **TIMING** of signals, not just their presence/absence."
> 2. "The 11-dimensional structure of the brain explains why humans naturally use **base-10**."

## Key Results

| Metric | Von Neumann | Brain-like | Advantage |
|--------|-------------|------------|-----------|
| **Information Capacity** | 2^n | 10^n | **9.77×10⁶x** (10 units) |
| **Pattern Matching** | O(n) serial | O(1) parallel | **102x faster** |
| **Fault Tolerance** | 1 bit = crash | 30% failure OK | **✅** |
| **Energy Efficiency** | 6000 pJ | 60 pJ | **100x** |

## 🆕 11-Dimensional Brain Structure

Based on Blue Brain Project's discovery that the brain has up to 11-dimensional structures:

### Information Propagation Speed

| Steps | 2D Grid | 11D Hypercube |
|-------|---------|---------------|
| 11 | 78 nodes | **2,048 nodes (complete!)** |
| 88 | 2,025 (complete) | - |

**11D Hypercube is 8x faster!**

### Decimal Hypothesis: Why Base-10?

| Base | Hypercube Advantage |
|------|---------------------|
| Base-8 | +2.8% |
| **Base-10** | **+4.2%** ⭐ |
| Base-12 | -5.8% |
| Base-16 | -0.7% |

**Base-10 shows the highest advantage!**

### MNIST Classification (10 digits)

| Topology | Accuracy | Connections |
|----------|----------|-------------|
| **9D Hypercube** | **90.6%** ⭐ | 4,608 |
| Random Sparse | 89.8% | 5,106 |
| Full Connection | 72.0% | 261,632 |

## File Structure

```
brain-vs-neumann/
├── README.md
├── visualize.py              # Generate publication figures
├── experiments/
│   ├── dim11_brain_structure.py  # 11D clique simulation
│   ├── hypercube_11d.py          # 11D hypercube propagation
│   ├── mnist_11d_test.py         # MNIST with 11D topology
│   ├── base_comparison.py        # Why base-10?
│   ├── brain_vs_neumann_sim.py   # Basic simulation
│   ├── validation_suite.py       # Comprehensive tests
│   └── realworld_tasks.py        # Real-world task comparison
├── figures/
│   ├── fig_hypercube_propagation.png
│   ├── fig_decimal_hypothesis.png
│   └── (6 more figures)
└── results/
    ├── 11dim_results.txt
    ├── hypercube_summary.txt
    ├── mnist_comparison.txt
    ├── base_comparison.txt
    └── (more results)
```

## Usage

```bash
# Run 11D brain structure simulation
python experiments/dim11_brain_structure.py

# Run 11D hypercube propagation test
python experiments/hypercube_11d.py

# Run MNIST with 11D topology
python experiments/mnist_11d_test.py

# Run base comparison (why base-10?)
python experiments/base_comparison.py

# Generate figures
python visualize.py
```

## Publications

- **Zenodo**: [10.5281/zenodo.18307795](https://zenodo.org/records/18307795)
- **Zenn**: [Why Base-10?](https://zenn.dev/cell_activation/articles/19802ec8c99764)

## Related Work

- [SNN Language Model](https://github.com/hafufu-stack/snn-language-model) - SNN for NLP
- [SNN Comprypto](https://github.com/hafufu-stack/temporal-coding-simulation/tree/main/snn-comprypto) - SNN for encryption

## Author

ろーる ([@hafufu-stack](https://github.com/hafufu-stack))
- **ORCID**: 0009-0004-2517-0177
- **note**: [https://note.com/cell_activation](https://note.com/cell_activation)
- **Zenn**: [https://zenn.dev/cell_activation](https://zenn.dev/cell_activation)

## License

CC BY 4.0
