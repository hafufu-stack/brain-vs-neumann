"""
Brain vs Von Neumann: Advanced Coding Schemes
==============================================

Comparing advanced neural coding schemes:
1. Rate Coding (basic)
2. Phase Coding (temporal)
3. Burst Coding (from 10-neuron-memory)
4. Correlation Coding (MD-LD style)

Author: Hiroto Funasaki (roll)
Date: 2026-01-20
"""

import numpy as np
import time


class LIFNeuron:
    """LIF neuron with burst detection"""
    
    def __init__(self, dt=0.05, tau=10.0):
        self.dt = dt
        self.tau = tau
        self.v_rest = -65.0
        self.v_thresh = -50.0
        self.v_reset = -70.0
        self.v = self.v_rest
        self.spike_times = []
    
    def reset(self):
        self.v = self.v_rest
        self.spike_times = []
    
    def step(self, I_syn, t):
        dv = (-(self.v - self.v_rest) + I_syn) / self.tau * self.dt
        self.v += dv
        if self.v >= self.v_thresh:
            self.spike_times.append(t)
            self.v = self.v_reset
            return True
        return False


# ============================================================
# Coding Schemes
# ============================================================

class RateCoding:
    """
    Rate Coding: Information in spike COUNT
    Traditional approach, simple but limited.
    """
    
    def __init__(self, num_neurons=10, max_rate=10):
        self.num_neurons = num_neurons
        self.max_rate = max_rate
    
    def information_capacity(self):
        # Each neuron can fire 0 to max_rate times
        return (self.max_rate + 1) ** self.num_neurons
    
    def encode(self, value):
        """Encode value as spike counts"""
        counts = []
        remaining = value
        for _ in range(self.num_neurons):
            counts.append(remaining % (self.max_rate + 1))
            remaining //= (self.max_rate + 1)
        return counts
    
    def decode(self, counts):
        """Decode spike counts to value"""
        value = 0
        for i, c in enumerate(counts):
            value += c * ((self.max_rate + 1) ** i)
        return value


class PhaseCoding:
    """
    Phase Coding: Information in spike TIMING
    More capacity with same neurons.
    """
    
    def __init__(self, num_neurons=10, phase_bins=10):
        self.num_neurons = num_neurons
        self.phase_bins = phase_bins
    
    def information_capacity(self):
        return self.phase_bins ** self.num_neurons
    
    def encode(self, value):
        """Encode value as spike phases"""
        phases = []
        remaining = value
        for _ in range(self.num_neurons):
            phases.append(remaining % self.phase_bins)
            remaining //= self.phase_bins
        return phases
    
    def decode(self, phases):
        """Decode phases to value"""
        value = 0
        for i, p in enumerate(phases):
            value += p * (self.phase_bins ** i)
        return value


class BurstCoding:
    """
    Burst Coding: Information in BOTH first spike time AND inter-spike interval
    From 10-neuron-memory project.
    Even more capacity!
    """
    
    def __init__(self, num_neurons=10, phase_bins=5, isi_bins=5):
        self.num_neurons = num_neurons
        self.phase_bins = phase_bins  # First spike timing
        self.isi_bins = isi_bins      # Inter-spike interval
        self.states_per_neuron = phase_bins * isi_bins
    
    def information_capacity(self):
        # Each neuron encodes: phase × ISI
        return self.states_per_neuron ** self.num_neurons
    
    def encode(self, value):
        """Encode value as (phase, ISI) pairs"""
        pairs = []
        remaining = value
        for _ in range(self.num_neurons):
            state = remaining % self.states_per_neuron
            phase = state // self.isi_bins
            isi = state % self.isi_bins
            pairs.append((phase, isi))
            remaining //= self.states_per_neuron
        return pairs
    
    def decode(self, pairs):
        """Decode (phase, ISI) pairs to value"""
        value = 0
        for i, (phase, isi) in enumerate(pairs):
            state = phase * self.isi_bins + isi
            value += state * (self.states_per_neuron ** i)
        return value


class CorrelationCoding:
    """
    Correlation Coding: Information in RELATIVE timing between neurons
    Inspired by MD-LD correlation in brain.
    """
    
    def __init__(self, num_neurons=10, delay_bins=10):
        self.num_neurons = num_neurons
        self.delay_bins = delay_bins
        # Reference neuron (like MD) + other neurons relative to it
        self.info_neurons = num_neurons - 1
    
    def information_capacity(self):
        # First neuron is reference, others encode relative delays
        return self.delay_bins ** self.info_neurons
    
    def encode(self, value):
        """Encode value as relative delays from reference"""
        delays = [0]  # Reference neuron fires at time 0
        remaining = value
        for _ in range(self.info_neurons):
            delay = remaining % self.delay_bins
            delays.append(delay)
            remaining //= self.delay_bins
        return delays
    
    def decode(self, delays):
        """Decode relative delays to value"""
        value = 0
        for i, d in enumerate(delays[1:]):  # Skip reference
            value += d * (self.delay_bins ** i)
        return value


def compare_coding_schemes():
    """Compare all coding schemes"""
    print("\n" + "=" * 70)
    print("   CODING SCHEME COMPARISON")
    print("=" * 70)
    
    schemes = {
        'Rate Coding': RateCoding(num_neurons=10, max_rate=10),
        'Phase Coding': PhaseCoding(num_neurons=10, phase_bins=10),
        'Burst Coding': BurstCoding(num_neurons=10, phase_bins=5, isi_bins=5),
        'Correlation Coding': CorrelationCoding(num_neurons=10, delay_bins=10),
    }
    
    # Von Neumann reference
    von_capacity = 2 ** 10
    
    print("\n[1] Information Capacity (10 units)")
    print("-" * 50)
    
    results = {}
    
    for name, scheme in schemes.items():
        capacity = scheme.information_capacity()
        ratio = capacity / von_capacity
        
        print(f"\n  {name}:")
        print(f"    Capacity: {capacity:,.0f}")
        print(f"    vs Von Neumann (10 bits): {ratio:,.0f}x")
        
        results[name] = {
            'capacity': capacity,
            'ratio': ratio
        }
    
    print(f"\n  Von Neumann (10 bits):")
    print(f"    Capacity: {von_capacity:,}")
    
    # Test encoding accuracy
    print("\n[2] Encoding/Decoding Accuracy")
    print("-" * 50)
    
    test_values = [0, 123, 456, 789, 999]
    
    for name, scheme in schemes.items():
        correct = 0
        for value in test_values:
            encoded = scheme.encode(value)
            decoded = scheme.decode(encoded)
            if decoded == value:
                correct += 1
        
        accuracy = correct / len(test_values) * 100
        print(f"  {name}: {accuracy:.0f}%")
        results[name]['accuracy'] = accuracy
    
    # Summary
    print("\n" + "=" * 70)
    print("   CODING SCHEME SUMMARY")
    print("=" * 70)
    
    print("""
    ┌─────────────────────┬───────────────┬──────────────┬──────────┐
    │ Coding Scheme       │ Capacity      │ vs Von (10b) │ Accuracy │
    ├─────────────────────┼───────────────┼──────────────┼──────────┤""")
    
    for name, r in results.items():
        if r['capacity'] >= 1e9:
            cap_str = f"{r['capacity']:.2e}"
        else:
            cap_str = f"{r['capacity']:,.0f}"
        print(f"    │ {name:<19} │ {cap_str:>13} │ {r['ratio']:>10,.0f}x │ {r['accuracy']:>6.0f}% │")
    
    print(f"    │ {'Von Neumann (10 bits)':<19} │ {von_capacity:>13,} │ {'1':>10}x │ {'100':>6}% │")
    print("    └─────────────────────┴───────────────┴──────────────┴──────────┘")
    
    print("""
    Key Insights:
    
    1. BURST CODING (phase + ISI) is most efficient:
       - Uses 2 dimensions per neuron
       - 25 states per neuron (5×5) vs 10 for phase alone
    
    2. All brain-like schemes beat Von Neumann by 10^6+ x
    
    3. Temporal coding is the key differentiator!
    """)
    
    # Save results
    with open("results/coding_comparison.txt", "w", encoding="utf-8") as f:
        f.write("Coding Scheme Comparison Results\n")
        f.write("=" * 40 + "\n\n")
        for name, r in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Capacity: {r['capacity']:.2e}\n")
            f.write(f"  Ratio: {r['ratio']:.0f}x\n")
            f.write(f"  Accuracy: {r['accuracy']:.0f}%\n\n")
    
    print("  Results saved to: results/coding_comparison.txt")
    
    return results


if __name__ == "__main__":
    compare_coding_schemes()
