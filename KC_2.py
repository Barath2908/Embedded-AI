import numpy as np
import matplotlib.pyplot as plt

# Simulation: Memory costs (in KB) for a network with 5 layers
# [Layer1, Layer2, Layer3, Layer4, Layer5]
memory_cost = np.array([60, 40, 30, 50, 70])  # in KB
layers = np.arange(1, 6)

# Full update: Sum memory cost of all layers
full_update_memory = np.sum(memory_cost)
print("Full Update Memory Cost: {} KB".format(full_update_memory))

# Sparse update strategy: only update layers where cost is below a threshold
threshold = 50  # in KB
sparse_update_mask = memory_cost < threshold
sparse_update_memory = np.sum(memory_cost[sparse_update_mask])
print("Sparse Update Memory Cost (only layers with cost < {} KB): {} KB".format(threshold, sparse_update_memory))

# Plot the per-layer memory cost and highlight layers selected for sparse update
plt.figure(figsize=(10, 5))
plt.plot(layers, memory_cost, label="Memory Cost per Layer", marker="o", linewidth=2)
plt.scatter(layers[sparse_update_mask], memory_cost[sparse_update_mask], color='red', label="Updated in Sparse Update", s=100)
plt.xlabel("Layer Index")
plt.ylabel("Memory Cost (KB)")
plt.title("Per-Layer Memory Cost: Full Update vs. Sparse Update")
plt.legend()
plt.grid(True)
plt.show()
