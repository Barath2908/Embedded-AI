import numpy as np
import matplotlib.pyplot as plt
from itertools import product

# ----- Simulation Parameters -----
L = 5  # number of layers
layers = np.arange(1, L + 1)

# For each layer, define:
# Bias update:
bias_improvement = np.ones(L)  # each bias update improves accuracy by 1 unit
bias_cost = np.full(L, 5)  # each bias update costs 5 KB

# Weight update:
# Each layer has a different full-update improvement value (arbitrary example values)
weight_improvement = np.array([4, 3, 5, 2, 4])
weight_cost = np.full(L, 20)  # full weight update costs 20 KB per layer

# Allowed update ratios for weights (0 means skip, 1 means full update)
update_ratios = [0, 0.125, 0.25, 0.5, 1.0]

# Memory budget (in KB)
M_budget = 80

# ----- Optimization: Exhaustive Search -----
# We search for the configuration (bias update decision for each layer and weight update ratio for each layer)
# that maximizes total improvement while keeping total memory cost below the budget.
best_total_improvement = -np.inf
best_config = None

# Iterate over all combinations for bias decisions (0 or 1 for each layer) and weight update ratios
for bias_decisions in product([0, 1], repeat=L):
    for weight_choices in product(update_ratios, repeat=L):
        bias_dec = np.array(bias_decisions)
        weight_ratios = np.array(weight_choices)

        # Total memory cost: sum over layers of (bias_cost if updated) + (weight_cost * update_ratio)
        total_memory = np.sum(bias_dec * bias_cost + weight_ratios * weight_cost)

        # Total improvement: sum over layers of (bias_improvement if updated) + (weight_improvement * update_ratio)
        total_improvement = np.sum(bias_dec * bias_improvement + weight_ratios * weight_improvement)

        # Check the memory constraint and if the current configuration is better than the best found so far
        if total_memory <= M_budget and total_improvement > best_total_improvement:
            best_total_improvement = total_improvement
            best_config = (bias_dec.copy(), weight_ratios.copy(), total_memory)

# Unpack best configuration:
best_bias, best_weight_ratios, best_memory = best_config

print("Best Total Improvement:", best_total_improvement)
print("Total Memory Cost:", best_memory, "KB")
print("Layer-wise Bias Update (1 = update, 0 = skip):", best_bias)
print("Layer-wise Weight Update Ratios:", best_weight_ratios)

# Calculate per-layer memory costs for the selected sparse update configuration
per_layer_bias_cost = best_bias * bias_cost  # cost from bias update per layer
per_layer_weight_cost = best_weight_ratios * weight_cost  # cost from weight update per layer
per_layer_total_cost = per_layer_bias_cost + per_layer_weight_cost

# Calculate per-layer improvement for the configuration
per_layer_bias_improvement = best_bias * bias_improvement
per_layer_weight_improvement = best_weight_ratios * weight_improvement
per_layer_total_improvement = per_layer_bias_improvement + per_layer_weight_improvement

# ----- Enhanced Plots -----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Subplot 1: Stacked Bar Chart of Per-Layer Memory Cost Breakdown
ax1.bar(layers, per_layer_bias_cost, color='skyblue', label='Bias Cost (5 KB if updated)')
ax1.bar(layers, per_layer_weight_cost, bottom=per_layer_bias_cost, color='lightgreen',
        label='Weight Cost (ratio × 20 KB)')
for i in range(L):
    ax1.text(layers[i], per_layer_total_cost[i] + 1, f"{per_layer_total_cost[i]:.1f} KB",
             ha='center', fontsize=10, color='darkblue')
ax1.set_xlabel("Layer Index")
ax1.set_ylabel("Memory Cost (KB)")
ax1.set_title(f"Per-Layer Memory Cost Breakdown\n(Total Memory: {best_memory} KB)")
ax1.legend(loc='upper right')
ax1.grid(axis='y')

# Subplot 2: Cumulative Improvement Across Layers
cumulative_improvement = np.cumsum(per_layer_total_improvement)
ax2.plot(layers, cumulative_improvement, marker='o', linestyle='-', color='purple', label='Cumulative Improvement')
for i in range(L):
    ax2.text(layers[i], cumulative_improvement[i] + 0.2, f"{per_layer_total_improvement[i]:.1f}",
             ha='center', fontsize=10, color='darkred')
ax2.set_xlabel("Layer Index")
ax2.set_ylabel("Cumulative Improvement")
ax2.set_title(f"Cumulative Accuracy Improvement\n(Total Improvement: {best_total_improvement})")
ax2.legend(loc='upper left')
ax2.grid(axis='y')

plt.tight_layout()
plt.show()
