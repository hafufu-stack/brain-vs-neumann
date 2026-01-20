# Brain vs Von Neumann Computer Simulation

🧠 **Von Neumann Architecture vs Brain-like Architecture: A Comprehensive Study**

## Overview

This project tests the hypothesis:

> "The brain encodes more information by using the **TIMING** of signals, not just their presence/absence."

## Key Results

| Metric | Von Neumann | Brain-like | Advantage |
|--------|-------------|------------|-----------|
| **Information Capacity** | 2^n | 10^n | **9.77×10⁶x** (10 units) |
| **Pattern Matching** | O(n) serial | O(1) parallel | **102x faster** |
| **Fault Tolerance** | 1 bit = crash | 30% failure OK | **✅** |
| **Energy Efficiency** | 6000 pJ | 60 pJ | **100x** |

## Real-World Task Results

| Task | Brain-like | Von Neumann |
|------|-----------|-------------|
| Pattern Recognition | ✓ | ✓ |
| Associative Memory | ✓ | ✗ |
| Sequence Prediction | ✓ | ✓ |
| Sensor Fusion | ✓ | ✓ |
| **Total** | **4/4** | **3/4** |

## Scaling Analysis

- **100 dimensions**: Brain has **7.89×10⁶⁹x** more capacity!
- Energy efficiency: Brain is **100x** better at all scales

## File Structure

```
brain-vs-neumann/
├── README.md
├── visualize.py              # Generate publication figures
├── experiments/
│   ├── brain_vs_neumann_sim.py    # Basic simulation
│   ├── brain_vs_neumann_lif.py    # LIF neuron integration
│   ├── validation_suite.py        # Comprehensive tests
│   ├── realworld_tasks.py         # Real-world task comparison
│   └── scaling_analysis.py        # Scaling analysis
├── figures/                  # Publication-quality figures
│   ├── fig1_information_capacity.png
│   ├── fig2_architecture.png
│   ├── fig3_pattern_matching.png
│   ├── fig4_fault_tolerance.png
│   ├── fig5_energy.png
│   ├── fig6_summary.png
│   └── fig_scaling_*.png
└── results/                  # Experiment results
    ├── results.txt
    ├── validation_results.txt
    ├── realworld_results.txt
    └── scaling_results.txt
```

## Usage

```bash
# Run basic simulation
python experiments/brain_vs_neumann_sim.py

# Run validation suite
python experiments/validation_suite.py

# Run real-world tasks
python experiments/realworld_tasks.py

# Run scaling analysis
python experiments/scaling_analysis.py

# Generate figures
python visualize.py
```

## Related Work

- [SNN Language Model](https://github.com/hafufu-stack/snn-language-model) - SNN for NLP
- [SNN Comprypto](https://github.com/hafufu-stack/snn-comprypto) - SNN for encryption

## Author

ろーる ([@hafufu-stack](https://github.com/hafufu-stack))
- **note**: [https://note.com/cell_activation](https://note.com/cell_activation)
- **Zenn**: [https://zenn.dev/cell_activation](https://zenn.dev/cell_activation)

## License

CC BY 4.0
