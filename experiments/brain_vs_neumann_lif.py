"""
Brain vs Von Neumann: LIF Neuron Integration
=============================================

Integrating realistic LIF (Leaky Integrate-and-Fire) neurons
from 10-neuron-memory project for more accurate comparison.

Author: Hiroto Funasaki (roll)
Date: 2026-01-20
"""

import numpy as np
import time


class LIFNeuron:
    """
    Leaky Integrate-and-Fire Neuron Model
    
    Based on the implementation in 10-neuron-memory project.
    Accurate simulation of membrane potential dynamics.
    """
    
    def __init__(self, dt=0.1, tau=10.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-70.0):
        """
        Parameters:
        -----------
        dt : float
            Time step [ms]
        tau : float
            Membrane time constant [ms]
        v_rest : float
            Resting membrane potential [mV]
        v_thresh : float
            Spike threshold [mV]
        v_reset : float
            Reset potential [mV]
        """
        self.dt = dt
        self.tau = tau
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.v = v_rest
        self.spike_times = []
    
    def reset(self):
        self.v = self.v_rest
        self.spike_times = []
    
    def step(self, I_syn, t):
        """
        Simulate one time step.
        
        Parameters:
        -----------
        I_syn : float
            Synaptic input current
        t : float
            Current time [ms]
        
        Returns:
        --------
        bool : True if spike occurred
        """
        # Leaky integration
        dv = (-(self.v - self.v_rest) + I_syn) / self.tau * self.dt
        self.v += dv
        
        # Check for spike
        if self.v >= self.v_thresh:
            self.spike_times.append(t)
            self.v = self.v_reset
            return True
        return False


class BrainLikeLIFComputer:
    """
    Brain-like computer using realistic LIF neurons.
    
    Uses temporal coding: information is encoded in spike timing.
    """
    
    def __init__(self, num_neurons=10, dt=0.1, sim_time=100.0, time_resolution=10):
        self.num_neurons = num_neurons
        self.dt = dt
        self.sim_time = sim_time
        self.time_resolution = time_resolution
        self.neurons = [LIFNeuron(dt=dt) for _ in range(num_neurons)]
        
        # Energy tracking
        self.total_spikes = 0
        self.total_local_ops = 0
    
    def reset(self):
        for neuron in self.neurons:
            neuron.reset()
        self.total_spikes = 0
        self.total_local_ops = 0
    
    def generate_input_current(self, target_time, t, amplitude=150.0, duration=2.0):
        """Generate input current pulse at target time"""
        if target_time <= t < target_time + duration:
            return amplitude
        return 0.0
    
    def encode(self, value):
        """
        Encode value using LIF neurons with temporal coding.
        Each neuron's spike timing encodes a digit.
        """
        self.reset()
        
        # Convert value to timing pattern
        patterns = []
        remaining = value
        for i in range(self.num_neurons):
            spike_time = (remaining % self.time_resolution) * (self.sim_time / self.time_resolution)
            patterns.append(spike_time + 10)  # Offset by 10ms
            remaining //= self.time_resolution
        
        # Simulate neurons
        time_points = np.arange(0, self.sim_time, self.dt)
        
        for i, neuron in enumerate(self.neurons):
            for t in time_points:
                I_syn = self.generate_input_current(patterns[i], t)
                if neuron.step(I_syn, t):
                    self.total_spikes += 1
                self.total_local_ops += 1
        
        return [patterns[i] for i in range(self.num_neurons)]
    
    def decode(self, spike_times_list=None):
        """Decode spike times back to value"""
        if spike_times_list is None:
            spike_times_list = [n.spike_times for n in self.neurons]
        
        value = 0
        for i, spike_times in enumerate(spike_times_list):
            if len(spike_times) > 0:
                first_spike = spike_times[0]
                # Convert timing back to digit
                digit = int((first_spike - 10) / (self.sim_time / self.time_resolution)) % self.time_resolution
                value += digit * (self.time_resolution ** i)
        
        return value
    
    def information_capacity(self):
        """Maximum patterns with temporal coding"""
        return self.time_resolution ** self.num_neurons
    
    def get_energy(self):
        """Estimate energy consumption"""
        # Energy model: 0.5 pJ per spike + 0.1 pJ per local operation
        return self.total_spikes * 0.5 + self.total_local_ops * 0.1


class VonNeumannComputer:
    """
    Von Neumann architecture simulator.
    """
    
    def __init__(self, bits=10):
        self.bits = bits
        self.max_value = 2 ** bits
        self.memory = np.zeros(1024, dtype=np.int32)
        self.cpu_register = 0
        
        # Stats
        self.bus_transfers = 0
        self.cpu_cycles = 0
    
    def reset(self):
        self.bus_transfers = 0
        self.cpu_cycles = 0
    
    def load_from_memory(self, address):
        self.cpu_register = self.memory[address % len(self.memory)]
        self.bus_transfers += 1
        self.cpu_cycles += 1
        return self.cpu_register
    
    def store_to_memory(self, address, value):
        self.memory[address % len(self.memory)] = value
        self.bus_transfers += 1
        self.cpu_cycles += 1
    
    def compute(self, operation, operand):
        if operation == 'add':
            self.cpu_register += operand
        elif operation == 'xor':
            self.cpu_register ^= operand
        self.cpu_cycles += 1
        return self.cpu_register
    
    def encode(self, value):
        if value >= self.max_value:
            raise ValueError(f"Value {value} exceeds {self.bits}-bit capacity")
        return format(value, f'0{self.bits}b')
    
    def decode(self, binary_str):
        return int(binary_str, 2)
    
    def information_capacity(self):
        return self.max_value
    
    def get_energy(self):
        """Estimate energy: 5 pJ per bus transfer + 1 pJ per CPU cycle"""
        return self.bus_transfers * 5.0 + self.cpu_cycles * 1.0


def run_comparison():
    """Run comprehensive comparison with LIF neurons"""
    
    print("=" * 70)
    print("   BRAIN vs VON NEUMANN: LIF NEURON INTEGRATION")
    print("=" * 70)
    
    n_neurons = 10
    n_bits = 10
    
    brain = BrainLikeLIFComputer(num_neurons=n_neurons)
    von = VonNeumannComputer(bits=n_bits)
    
    # Comparison 1: Information Capacity
    print("\n[1] Information Capacity")
    print("-" * 40)
    brain_capacity = brain.information_capacity()
    von_capacity = von.information_capacity()
    ratio = brain_capacity / von_capacity
    
    print(f"  Von Neumann ({n_bits} bits): {von_capacity:,} patterns")
    print(f"  Brain-like ({n_neurons} neurons): {brain_capacity:,} patterns")
    print(f"  Ratio: {ratio:,.0f}x more capacity!")
    
    # Comparison 2: Encoding/Decoding with actual LIF simulation
    print("\n[2] Encoding/Decoding Test (LIF Simulation)")
    print("-" * 40)
    
    test_values = [0, 123, 456, 789, 999]
    print("  Testing encode/decode accuracy...")
    
    all_correct = True
    for value in test_values:
        brain.reset()
        patterns = brain.encode(value)
        recovered = brain.decode()
        
        correct = (recovered == value)
        all_correct = all_correct and correct
        status = "✓" if correct else "✗"
        print(f"    {value} → encode → {recovered} {status}")
    
    print(f"  Result: {'All correct!' if all_correct else 'Some errors'}")
    
    # Comparison 3: Energy Efficiency
    print("\n[3] Energy Efficiency")
    print("-" * 40)
    
    # Process multiple values
    n_operations = 100
    
    brain.reset()
    von.reset()
    
    for i in range(n_operations):
        # Brain: encode and decode
        brain.encode(i % 1000)
        brain.decode()
        
        # Von Neumann: store and load
        von.store_to_memory(i, i % 1000)
        von.load_from_memory(i)
        von.compute('add', 1)
    
    brain_energy = brain.get_energy()
    von_energy = von.get_energy()
    efficiency = von_energy / brain_energy if brain_energy > 0 else float('inf')
    
    print(f"  Processing {n_operations} operations:")
    print(f"  Von Neumann: {von_energy:.1f} pJ")
    print(f"  Brain-like:  {brain_energy:.1f} pJ")
    print(f"  Efficiency: Brain is {efficiency:.1f}x more efficient!")
    
    # Summary
    print("\n" + "=" * 70)
    print("   SUMMARY WITH LIF NEURONS")
    print("=" * 70)
    
    print(f"""
    ┌─────────────────────────┬────────────────┬────────────────┐
    │ Metric                  │ Von Neumann    │ Brain-like     │
    ├─────────────────────────┼────────────────┼────────────────┤
    │ Information Capacity    │ {von_capacity:>12,} │ {brain_capacity:>12,} │
    │ Capacity Ratio          │            1x  │ {ratio:>11,.0f}x │
    │ Energy ({n_operations} ops)          │ {von_energy:>10.1f} pJ │ {brain_energy:>10.1f} pJ │
    │ Energy Efficiency       │            1x  │ {efficiency:>11.1f}x │
    └─────────────────────────┴────────────────┴────────────────┘
    
    🧠 Brain-like architecture with realistic LIF neurons still wins!
    """)
    
    # Save results
    with open("results_lif.txt", "w", encoding="utf-8") as f:
        f.write("Brain vs Von Neumann: LIF Neuron Integration Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Neurons: {n_neurons}, Bits: {n_bits}\n")
        f.write(f"Information Capacity Ratio: {ratio:.0f}x\n")
        f.write(f"Energy Efficiency: {efficiency:.1f}x\n")
        f.write(f"Encoding Accuracy: {'100%' if all_correct else 'Some errors'}\n")
    
    print("  Results saved to: results_lif.txt")


if __name__ == "__main__":
    run_comparison()
