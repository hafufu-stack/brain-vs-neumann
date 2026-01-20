"""
Brain vs Von Neumann: Enhanced Visualization
=============================================

Comprehensive comparison with publication-quality figures.

Author: Hiroto Funasaki (roll)
Date: 2026-01-20
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # For saving figures without display

# Try to use Japanese fonts
try:
    import japanize_matplotlib
except:
    pass


def plot_information_capacity():
    """Plot: Information capacity comparison"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    units = np.array([2, 4, 6, 8, 10, 12])
    von_neumann = 2 ** units
    brain_like = 10 ** units
    
    # Linear scale
    ax1.semilogy(units, von_neumann, 'b-o', linewidth=2, markersize=10, label='Von Neumann (2^n)')
    ax1.semilogy(units, brain_like, 'r-s', linewidth=2, markersize=10, label='Brain-like (10^n)')
    ax1.set_xlabel('Number of Units (bits / neurons)', fontsize=12)
    ax1.set_ylabel('Information Capacity (patterns)', fontsize=12)
    ax1.set_title('Information Capacity Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(units)
    
    # Ratio
    ratio = brain_like / von_neumann
    bars = ax2.bar(units, ratio, color=['#2ecc71' if r > 1000 else '#3498db' for r in ratio], 
                   edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Number of Units', fontsize=12)
    ax2.set_ylabel('Capacity Ratio (Brain / Von Neumann)', fontsize=12)
    ax2.set_title('How Many Times More Capacity?', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_xticks(units)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, r in zip(bars, ratio):
        if r >= 1e6:
            label = f'{r/1e6:.0f}M×'
        elif r >= 1e3:
            label = f'{r/1e3:.0f}K×'
        else:
            label = f'{r:.0f}×'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.5, 
                 label, ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fig1_information_capacity.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig1_information_capacity.png")


def plot_architecture_comparison():
    """Plot: Architecture diagram"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Von Neumann
    ax1 = axes[0]
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Von Neumann Architecture', fontsize=14, fontweight='bold', pad=20)
    
    # CPU
    cpu = plt.Rectangle((1, 6), 2.5, 2), 
    ax1.add_patch(plt.Rectangle((1, 6), 2.5, 2, facecolor='#3498db', edgecolor='black', linewidth=2))
    ax1.text(2.25, 7, 'CPU', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # RAM
    ax1.add_patch(plt.Rectangle((4, 6), 2.5, 2, facecolor='#e74c3c', edgecolor='black', linewidth=2))
    ax1.text(5.25, 7, 'RAM', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Storage
    ax1.add_patch(plt.Rectangle((7, 6), 2.5, 2, facecolor='#9b59b6', edgecolor='black', linewidth=2))
    ax1.text(8.25, 7, 'Storage', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Bus arrows
    ax1.annotate('', xy=(4, 7), xytext=(3.5, 7),
                 arrowprops=dict(arrowstyle='<->', color='orange', lw=3))
    ax1.annotate('', xy=(7, 7), xytext=(6.5, 7),
                 arrowprops=dict(arrowstyle='<->', color='orange', lw=3))
    
    ax1.text(5.25, 4, 'Data Bus (Bottleneck!)', ha='center', va='center', 
             fontsize=11, color='orange', fontweight='bold')
    ax1.text(5.25, 3, '• Serial processing\n• Data must move\n• Energy inefficient', 
             ha='center', va='center', fontsize=10, color='gray')
    
    # Brain-like
    ax2 = axes[1]
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title('Brain-like Architecture', fontsize=14, fontweight='bold', pad=20)
    
    # Neurons (processing + memory units)
    positions = [(2, 7), (5, 8), (8, 7), (1.5, 5), (4, 5.5), (6, 5.5), (8.5, 5), (3, 3.5), (5, 3), (7, 3.5)]
    for i, (x, y) in enumerate(positions):
        circle = plt.Circle((x, y), 0.6, facecolor='#2ecc71', edgecolor='black', linewidth=2)
        ax2.add_patch(circle)
    
    # Connections
    for i, (x1, y1) in enumerate(positions):
        for j, (x2, y2) in enumerate(positions):
            if i < j and np.random.rand() > 0.5:
                ax2.plot([x1, x2], [y1, y2], 'g-', alpha=0.3, linewidth=1)
    
    ax2.text(5, 1, '• Parallel processing\n• No data movement\n• Energy efficient', 
             ha='center', va='center', fontsize=10, color='gray')
    ax2.text(5, 0, 'Each unit = Processing + Memory', ha='center', va='center', 
             fontsize=11, color='#2ecc71', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fig2_architecture.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig2_architecture.png")


def plot_pattern_matching():
    """Plot: Pattern matching efficiency"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    patterns = [10, 50, 100, 500, 1000]
    von_cycles = [p for p in patterns]  # O(n)
    brain_cycles = [1 for _ in patterns]  # O(1)
    
    x = np.arange(len(patterns))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, von_cycles, width, label='Von Neumann (serial)', 
                   color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, brain_cycles, width, label='Brain-like (parallel)', 
                   color='#2ecc71', edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Number of Patterns to Search', fontsize=12)
    ax.set_ylabel('Cycles Required', fontsize=12)
    ax.set_title('Pattern Matching Efficiency', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(patterns)
    ax.legend(fontsize=11)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add efficiency labels
    for i, (v, b) in enumerate(zip(von_cycles, brain_cycles)):
        ax.text(i, v * 1.2, f'{v}×', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fig3_pattern_matching.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig3_pattern_matching.png")


def plot_fault_tolerance():
    """Plot: Fault tolerance comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fail_rates = [0, 10, 20, 30, 40, 50]
    brain_accuracy = [100, 95, 85, 70, 40, 10]  # Graceful degradation
    von_accuracy = [100, 0, 0, 0, 0, 0]  # Crash on any error
    
    ax.plot(fail_rates, brain_accuracy, 'g-o', linewidth=3, markersize=12, 
            label='Brain-like (graceful degradation)')
    ax.plot(fail_rates, von_accuracy, 'b-s', linewidth=3, markersize=12, 
            label='Von Neumann (1 bit = crash)')
    
    ax.fill_between(fail_rates, brain_accuracy, alpha=0.3, color='green')
    
    ax.set_xlabel('Failure Rate (%)', fontsize=12)
    ax.set_ylabel('System Accuracy (%)', fontsize=12)
    ax.set_title('Fault Tolerance: What Happens When Parts Fail?', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 110)
    ax.set_xlim(-2, 52)
    
    # Annotations
    ax.annotate('Von Neumann crashes\nat first error!', xy=(10, 5), xytext=(20, 30),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='red'),
                color='red', fontweight='bold')
    
    ax.annotate('Brain-like degrades\ngracefully', xy=(30, 70), xytext=(35, 85),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='green'),
                color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fig4_fault_tolerance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig4_fault_tolerance.png")


def plot_energy_efficiency():
    """Plot: Energy efficiency comparison"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Bus\nTransfer', 'CPU\nOperation', 'Spike\n(Neuro)', 'Local\nOp (PIM)']
    energy = [5.0, 1.0, 0.5, 0.1]  # pJ
    colors = ['#e74c3c', '#e74c3c', '#2ecc71', '#2ecc71']
    
    bars = ax.bar(categories, energy, color=colors, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Energy per Operation (pJ)', fontsize=12)
    ax.set_title('Energy Cost Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, e in zip(bars, energy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{e} pJ', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add legend-like annotations
    ax.text(0.25, 4.5, 'Von Neumann', ha='center', fontsize=11, color='#e74c3c', fontweight='bold')
    ax.text(2.75, 0.8, 'Brain-like', ha='center', fontsize=11, color='#2ecc71', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('fig5_energy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig5_energy.png")


def plot_summary():
    """Plot: Summary comparison table as figure"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    
    # Table data
    columns = ['Metric', 'Von Neumann', 'Brain-like', 'Winner']
    data = [
        ['Information Capacity', '2^n', '10^n', '🧠 Brain (9.7M×)'],
        ['Pattern Matching', 'O(n) serial', 'O(1) parallel', '🧠 Brain (100×)'],
        ['Fault Tolerance', '1 bit = crash', 'Graceful', '🧠 Brain'],
        ['Energy Efficiency', '6000 pJ', '150 pJ', '🧠 Brain (40×)'],
    ]
    
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Style winner column
    for i in range(1, len(data) + 1):
        table[(i, 3)].set_facecolor('#d5f4e6')
    
    ax.set_title('Summary: Brain-like Wins in All Categories!', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('fig6_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: fig6_summary.png")


def main():
    print("=" * 60)
    print("  Generating Publication-Quality Figures")
    print("=" * 60)
    
    plot_information_capacity()
    plot_architecture_comparison()
    plot_pattern_matching()
    plot_fault_tolerance()
    plot_energy_efficiency()
    plot_summary()
    
    print("\n" + "=" * 60)
    print("  All figures saved!")
    print("=" * 60)
    print("\n  Generated figures:")
    print("    - fig1_information_capacity.png")
    print("    - fig2_architecture.png")
    print("    - fig3_pattern_matching.png")
    print("    - fig4_fault_tolerance.png")
    print("    - fig5_energy.png")
    print("    - fig6_summary.png")


if __name__ == "__main__":
    main()
