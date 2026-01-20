# Brain vs Von Neumann Computer Simulation

🧠 **Von Neumann Architecture vs Brain-like Architecture: A Comparative Study**

## Overview

This project tests the hypothesis:

> "The brain encodes more information by using the **TIMING** of signals, not just their presence/absence."

## Key Results

| Metric | Von Neumann | Brain-like | Advantage |
|--------|-------------|------------|-----------|
| **Information Capacity** | 2^n | 10^n | **9,765,625x** |
| **Pattern Matching** | O(n) serial | O(1) parallel | **102x faster** |
| **Fault Tolerance** | 1 bit = crash | 30% failure OK | **✅** |
| **Energy Efficiency** | 6000 pJ | 150 pJ | **40x** |

## The Hypothesis

Traditional von Neumann computers use **discrete bits** (0 or 1).

Brain-like computers use **temporal coding**:
- Same number of neurons
- But information is encoded in **WHEN** they fire
- Result: Exponentially more information capacity!

## Example

```
10 bits (Von Neumann):     2^10 = 1,024 patterns
10 neurons × 10 timings:  10^10 = 10,000,000,000 patterns
                          → 9.7 million times more!
```

## Usage

```bash
python brain_vs_neumann_sim.py
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
