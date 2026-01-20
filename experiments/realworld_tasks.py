"""
Brain vs Von Neumann: Real-World Task Comparison
=================================================

Testing on realistic tasks to demonstrate practical advantages.

Author: Hiroto Funasaki (roll)
Date: 2026-01-20
"""

import numpy as np
import time


# ============================================================
# LIF Neuron Implementation
# ============================================================

class LIFNeuron:
    """High-precision LIF neuron"""
    
    def __init__(self, dt=0.05, tau=10.0, v_rest=-65.0, v_thresh=-50.0, v_reset=-70.0):
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
        dv = (-(self.v - self.v_rest) + I_syn) / self.tau * self.dt
        self.v += dv
        if self.v >= self.v_thresh:
            self.spike_times.append(t)
            self.v = self.v_reset
            return True
        return False


# ============================================================
# Architectures
# ============================================================

class BrainComputer:
    """Brain-like architecture with temporal coding"""
    
    def __init__(self, num_neurons=20, dt=0.05, sim_time=50.0):
        self.num_neurons = num_neurons
        self.dt = dt
        self.sim_time = sim_time
        self.neurons = [LIFNeuron(dt=dt) for _ in range(num_neurons)]
        self.phase_bins = 8
        
        self.ops = 0
        self.spikes = 0
        self.time_elapsed = 0
    
    def reset(self):
        for n in self.neurons:
            n.reset()
        self.ops = 0
        self.spikes = 0
    
    def encode_pattern(self, pattern):
        """Encode a pattern (array) into spike timing"""
        self.reset()
        
        phase_duration = self.sim_time / self.phase_bins
        time_points = np.arange(0, self.sim_time, self.dt)
        
        for i, val in enumerate(pattern[:self.num_neurons]):
            target_time = (val % self.phase_bins) * phase_duration + phase_duration/2
            
            for t in time_points:
                if abs(t - target_time) < 1.5:
                    I_syn = 200.0
                else:
                    I_syn = 0.0
                
                if self.neurons[i].step(I_syn, t):
                    self.spikes += 1
                self.ops += 1
    
    def pattern_match(self, query, patterns):
        """Find best matching pattern using spike correlation"""
        t0 = time.time()
        
        # Encode query
        self.encode_pattern(query)
        query_spikes = [n.spike_times[0] if n.spike_times else -1 for n in self.neurons]
        
        best_match = -1
        best_score = -float('inf')
        
        for idx, pattern in enumerate(patterns):
            self.encode_pattern(pattern)
            pattern_spikes = [n.spike_times[0] if n.spike_times else -1 for n in self.neurons]
            
            # Correlation based on timing similarity
            score = 0
            for q, p in zip(query_spikes, pattern_spikes):
                if q >= 0 and p >= 0:
                    score += 1.0 / (1.0 + abs(q - p))
            
            if score > best_score:
                best_score = score
                best_match = idx
        
        self.time_elapsed = time.time() - t0
        return best_match
    
    def associative_recall(self, partial, stored):
        """Recall complete pattern from partial input"""
        t0 = time.time()
        
        # Find most similar stored pattern
        best_match = self.pattern_match(partial, stored)
        self.time_elapsed = time.time() - t0
        
        return stored[best_match] if best_match >= 0 else None
    
    def get_energy(self):
        return self.spikes * 0.5 + self.ops * 0.01


class VonNeumannComputer:
    """Von Neumann architecture"""
    
    def __init__(self, word_size=32):
        self.word_size = word_size
        self.memory = {}
        self.registers = [0] * 8
        
        self.bus_transfers = 0
        self.cycles = 0
        self.time_elapsed = 0
    
    def reset(self):
        self.bus_transfers = 0
        self.cycles = 0
    
    def store(self, addr, value):
        self.memory[addr] = value
        self.bus_transfers += 1
        self.cycles += 1
    
    def load(self, addr):
        self.bus_transfers += 1
        self.cycles += 1
        return self.memory.get(addr, 0)
    
    def compute(self, a, b, op='add'):
        self.cycles += 1
        if op == 'add':
            return a + b
        elif op == 'sub':
            return abs(a - b)
        elif op == 'xor':
            return a ^ b
        return a
    
    def pattern_match(self, query, patterns):
        """Sequential pattern matching"""
        t0 = time.time()
        self.reset()
        
        best_match = -1
        best_score = float('inf')
        
        for idx, pattern in enumerate(patterns):
            # Store pattern in memory
            for i, val in enumerate(pattern):
                self.store(i, val)
            
            # Compare with query
            total_diff = 0
            for i, q in enumerate(query):
                p = self.load(i)
                diff = self.compute(q, p, 'sub')
                total_diff = self.compute(total_diff, diff, 'add')
            
            if total_diff < best_score:
                best_score = total_diff
                best_match = idx
        
        self.time_elapsed = time.time() - t0
        return best_match
    
    def associative_recall(self, partial, stored):
        """Recall using sequential search"""
        t0 = time.time()
        
        best_match = self.pattern_match(partial, stored)
        self.time_elapsed = time.time() - t0
        
        return stored[best_match] if best_match >= 0 else None
    
    def get_energy(self):
        return self.bus_transfers * 5.0 + self.cycles * 1.0


# ============================================================
# Real-World Task Tests
# ============================================================

def test_pattern_recognition():
    """Task 1: Pattern recognition (like image recognition)"""
    print("\n" + "=" * 60)
    print("  TASK 1: Pattern Recognition")
    print("=" * 60)
    
    # Create patterns (simulating image features)
    np.random.seed(42)
    n_patterns = 50
    pattern_size = 20
    
    patterns = [np.random.randint(0, 8, pattern_size) for _ in range(n_patterns)]
    
    # Query (with slight noise)
    query_idx = 25
    query = patterns[query_idx].copy()
    noise_idx = np.random.choice(pattern_size, 3, replace=False)
    for i in noise_idx:
        query[i] = (query[i] + np.random.randint(1, 3)) % 8
    
    # Test
    brain = BrainComputer(num_neurons=pattern_size)
    von = VonNeumannComputer()
    
    brain_result = brain.pattern_match(query, patterns)
    von_result = von.pattern_match(query, patterns)
    
    print(f"\n  Patterns: {n_patterns}")
    print(f"  Pattern size: {pattern_size}")
    print(f"  Query corrupted: 3 elements")
    
    print(f"\n  Results:")
    print(f"    Brain-like: Found pattern {brain_result} ({'✓' if brain_result == query_idx else '✗'})")
    print(f"    Von Neumann: Found pattern {von_result} ({'✓' if von_result == query_idx else '✗'})")
    
    print(f"\n  Performance:")
    print(f"    Brain-like: {brain.time_elapsed*1000:.2f}ms, {brain.get_energy():.1f}pJ")
    print(f"    Von Neumann: {von.time_elapsed*1000:.2f}ms, {von.get_energy():.1f}pJ")
    
    return {
        'brain_correct': brain_result == query_idx,
        'von_correct': von_result == query_idx,
        'brain_energy': brain.get_energy(),
        'von_energy': von.get_energy()
    }


def test_associative_memory():
    """Task 2: Associative memory recall"""
    print("\n" + "=" * 60)
    print("  TASK 2: Associative Memory Recall")
    print("=" * 60)
    
    # Create stored memories
    np.random.seed(123)
    n_memories = 20
    memory_size = 16
    
    memories = [np.random.randint(0, 8, memory_size) for _ in range(n_memories)]
    
    # Test partial recall (50% of data missing)
    test_idx = 10
    partial = memories[test_idx].copy()
    
    # Mask half the data
    mask = np.random.choice(memory_size, memory_size // 2, replace=False)
    partial[mask] = -1  # Mark as unknown
    
    brain = BrainComputer(num_neurons=memory_size)
    von = VonNeumannComputer()
    
    # For brain, use only known values
    partial_known = partial.copy()
    partial_known[partial_known < 0] = 0
    
    brain_recalled = brain.associative_recall(partial_known, memories)
    von_recalled = von.associative_recall(partial_known, memories)
    
    # Check accuracy
    brain_match = np.array_equal(brain_recalled, memories[test_idx]) if brain_recalled is not None else False
    von_match = np.array_equal(von_recalled, memories[test_idx]) if von_recalled is not None else False
    
    print(f"\n  Stored memories: {n_memories}")
    print(f"  Memory size: {memory_size}")
    print(f"  Missing data: 50%")
    
    print(f"\n  Results:")
    print(f"    Brain-like: {'Correct recall' if brain_match else 'Incorrect'} ✓" if brain_match else f"    Brain-like: Incorrect")
    print(f"    Von Neumann: {'Correct recall' if von_match else 'Incorrect'} ✓" if von_match else f"    Von Neumann: Incorrect")
    
    print(f"\n  Energy:")
    print(f"    Brain-like: {brain.get_energy():.1f}pJ")
    print(f"    Von Neumann: {von.get_energy():.1f}pJ")
    
    return {
        'brain_correct': brain_match,
        'von_correct': von_match
    }


def test_sequence_prediction():
    """Task 3: Sequence prediction (language-model-like)"""
    print("\n" + "=" * 60)
    print("  TASK 3: Sequence Prediction")
    print("=" * 60)
    
    # Create sequences (simulating text patterns)
    sequences = [
        [0, 1, 2, 3, 4, 5, 6, 7],  # Ascending
        [7, 6, 5, 4, 3, 2, 1, 0],  # Descending
        [0, 2, 4, 6, 0, 2, 4, 6],  # Even repeat
        [1, 3, 5, 7, 1, 3, 5, 7],  # Odd repeat
        [0, 0, 1, 1, 2, 2, 3, 3],  # Doubles
    ]
    
    # Query: first 6 elements, predict rest
    test_idx = 2
    query = sequences[test_idx][:6]
    expected = sequences[test_idx]
    
    brain = BrainComputer(num_neurons=8)
    von = VonNeumannComputer()
    
    # Pad query
    query_padded = list(query) + [0, 0]
    
    brain_match = brain.pattern_match(query_padded, sequences)
    von_match = von.pattern_match(query_padded, sequences)
    
    brain_correct = brain_match == test_idx
    von_correct = von_match == test_idx
    
    print(f"\n  Sequences: {len(sequences)}")
    print(f"  Query (first 6): {query}")
    print(f"  Expected: pattern {test_idx}")
    
    print(f"\n  Results:")
    print(f"    Brain-like: Predicted pattern {brain_match} ({'✓' if brain_correct else '✗'})")
    print(f"    Von Neumann: Predicted pattern {von_match} ({'✓' if von_correct else '✗'})")
    
    return {
        'brain_correct': brain_correct,
        'von_correct': von_correct
    }


def test_sensor_fusion():
    """Task 4: Multi-sensor fusion (robotics-like)"""
    print("\n" + "=" * 60)
    print("  TASK 4: Sensor Fusion")
    print("=" * 60)
    
    # Simulate sensor data (3 sensors, 5 readings each = 15 dimensions)
    np.random.seed(456)
    
    # Known states
    states = []
    for i in range(10):
        # Each state is 3 sensors x 5 readings
        state = np.random.randint(0, 8, 15)
        states.append(state)
    
    # Current sensor reading (state 5 with noise)
    current = states[5].copy()
    # Add noise to some sensors
    noise_idx = np.random.choice(15, 4, replace=False)
    for i in noise_idx:
        current[i] = (current[i] + np.random.randint(1, 3)) % 8
    
    brain = BrainComputer(num_neurons=15)
    von = VonNeumannComputer()
    
    brain_result = brain.pattern_match(current, states)
    von_result = von.pattern_match(current, states)
    
    print(f"\n  Known states: {len(states)}")
    print(f"  Sensors: 3 × 5 readings = 15 dim")
    print(f"  Noisy readings: 4")
    
    print(f"\n  Results:")
    print(f"    Brain-like: Identified state {brain_result} ({'✓' if brain_result == 5 else '✗'})")
    print(f"    Von Neumann: Identified state {von_result} ({'✓' if von_result == 5 else '✗'})")
    
    energy_ratio = von.get_energy() / brain.get_energy() if brain.get_energy() > 0 else 0
    print(f"\n  Energy ratio: Brain is {energy_ratio:.1f}x more efficient")
    
    return {
        'brain_correct': brain_result == 5,
        'von_correct': von_result == 5,
        'energy_ratio': energy_ratio
    }


def run_all_real_world_tests():
    """Run all real-world task tests"""
    print("\n" + "=" * 70)
    print("   BRAIN vs VON NEUMANN: REAL-WORLD TASK COMPARISON")
    print("=" * 70)
    
    results = {}
    
    results['pattern'] = test_pattern_recognition()
    results['memory'] = test_associative_memory()
    results['sequence'] = test_sequence_prediction()
    results['sensor'] = test_sensor_fusion()
    
    # Summary
    print("\n" + "=" * 70)
    print("   REAL-WORLD TASK SUMMARY")
    print("=" * 70)
    
    brain_wins = sum([
        results['pattern']['brain_correct'],
        results['memory']['brain_correct'],
        results['sequence']['brain_correct'],
        results['sensor']['brain_correct']
    ])
    
    von_wins = sum([
        results['pattern']['von_correct'],
        results['memory']['von_correct'],
        results['sequence']['von_correct'],
        results['sensor']['von_correct']
    ])
    
    print(f"""
    ┌─────────────────────────┬─────────────┬─────────────┐
    │ Task                    │ Brain-like  │ Von Neumann │
    ├─────────────────────────┼─────────────┼─────────────┤
    │ Pattern Recognition     │ {'✓' if results['pattern']['brain_correct'] else '✗'}           │ {'✓' if results['pattern']['von_correct'] else '✗'}           │
    │ Associative Memory      │ {'✓' if results['memory']['brain_correct'] else '✗'}           │ {'✓' if results['memory']['von_correct'] else '✗'}           │
    │ Sequence Prediction     │ {'✓' if results['sequence']['brain_correct'] else '✗'}           │ {'✓' if results['sequence']['von_correct'] else '✗'}           │
    │ Sensor Fusion           │ {'✓' if results['sensor']['brain_correct'] else '✗'}           │ {'✓' if results['sensor']['von_correct'] else '✗'}           │
    ├─────────────────────────┼─────────────┼─────────────┤
    │ Total Correct           │ {brain_wins}/4         │ {von_wins}/4         │
    └─────────────────────────┴─────────────┴─────────────┘
    
    🧠 Brain-like: {brain_wins}/4 tasks correct
    💻 Von Neumann: {von_wins}/4 tasks correct
    
    Additional findings:
    - Pattern recognition: Brain {results['pattern']['von_energy']/results['pattern']['brain_energy']:.1f}x more energy efficient
    - Sensor fusion: Brain {results['sensor']['energy_ratio']:.1f}x more energy efficient
    """)
    
    # Save results
    with open("realworld_results.txt", "w", encoding="utf-8") as f:
        f.write("Real-World Task Comparison Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Brain-like: {brain_wins}/4 correct\n")
        f.write(f"Von Neumann: {von_wins}/4 correct\n\n")
        f.write("Tasks:\n")
        f.write(f"  Pattern Recognition: Brain={'✓' if results['pattern']['brain_correct'] else '✗'}, Von={'✓' if results['pattern']['von_correct'] else '✗'}\n")
        f.write(f"  Associative Memory: Brain={'✓' if results['memory']['brain_correct'] else '✗'}, Von={'✓' if results['memory']['von_correct'] else '✗'}\n")
        f.write(f"  Sequence Prediction: Brain={'✓' if results['sequence']['brain_correct'] else '✗'}, Von={'✓' if results['sequence']['von_correct'] else '✗'}\n")
        f.write(f"  Sensor Fusion: Brain={'✓' if results['sensor']['brain_correct'] else '✗'}, Von={'✓' if results['sensor']['von_correct'] else '✗'}\n")
    
    print("  Results saved to: realworld_results.txt")
    
    return results


if __name__ == "__main__":
    run_all_real_world_tests()
