import numpy as np
import matplotlib.pyplot as plt

# Network parameters
batch_size = 32
dims = [100, 64, 32, 10]  # Input dim and layer outputs (3 layers)


# Function to compute memory (in KB) for a given layer output size
def memory_kb(output_dim):
    # Each activation: batch_size * output_dim elements, 4 bytes per element, convert to KB.
    return (batch_size * output_dim * 4) / 1024


# Compute activation memory for each layer (ignoring the input layer)
activation_mem = np.array([memory_kb(d) for d in dims[1:]])  # Layers 1, 2, 3
# Assume gradients have the same size as activations.
gradient_mem = activation_mem.copy()

# Number of layers in our network (3 layers)
L = len(activation_mem)

# --- Vanilla Approach ---
# In vanilla back-propagation, all activations and gradients are held concurrently.
# Cumulative memory after each layer is sum_{i=1}^{n} (activation[i] + gradient[i])
cumulative_vanilla = np.cumsum(activation_mem + gradient_mem)

# --- Optimized In-Place Approach ---
# Here, all forward activations are kept, but only one gradient is stored at any time,
# so the peak memory at layer n is sum_{i=1}^{n} (activation[i]) + max_{i=1..n}(gradient[i])
cumulative_activations = np.cumsum(activation_mem)
# For each layer, the peak gradient memory is the maximum gradient seen so far.
peak_gradients = np.maximum.accumulate(gradient_mem)
cumulative_optimized = cumulative_activations + peak_gradients

# Print final peak memory values for comparison:
total_vanilla = cumulative_vanilla[-1]
total_optimized = cumulative_optimized[-1]
print("Vanilla Approach Peak Memory (KB):", total_vanilla)
print("Optimized (In-place) Approach Peak Memory (KB):", total_optimized)

# --- Plotting ---

# Plot 1: Stacked Bar Chart for per-layer memory (Vanilla Approach)
layer_indices = np.arange(1, L + 1)
width = 0.4

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot activations (bottom) and gradients (on top) for each layer
ax1.bar(layer_indices, activation_mem, width, color='lightblue', label='Activation Memory')
ax1.bar(layer_indices, gradient_mem, width, bottom=activation_mem, color='salmon', label='Gradient Memory')

for i in range(L):
    total_layer_mem = activation_mem[i] + gradient_mem[i]
    ax1.text(layer_indices[i], total_layer_mem + 0.5, f"{total_layer_mem:.2f} KB",
             ha='center', fontsize=10, color='navy')

ax1.set_xlabel("Layer Index")
ax1.set_ylabel("Memory (KB)")
ax1.set_title("Per-Layer Memory Cost (Vanilla Approach)")
ax1.legend(loc='upper left')
ax1.grid(axis='y')

# Plot 2: Cumulative Memory Usage Comparison
ax2.plot(layer_indices, cumulative_vanilla, marker='o', linestyle='-', color='red', label='Vanilla Cumulative Memory')
ax2.plot(layer_indices, cumulative_optimized, marker='s', linestyle='--', color='green',
         label='Optimized Cumulative Memory')

# Annotate cumulative values on the plots
for i in range(L):
    ax2.text(layer_indices[i], cumulative_vanilla[i] + 0.5, f"{cumulative_vanilla[i]:.2f} KB",
             ha='center', fontsize=10, color='red')
    ax2.text(layer_indices[i], cumulative_optimized[i] + 0.5, f"{cumulative_optimized[i]:.2f} KB",
             ha='center', fontsize=10, color='green')

ax2.set_xlabel("Layer Index")
ax2.set_ylabel("Cumulative Memory (KB)")
ax2.set_title("Cumulative Memory Usage Across Layers")
ax2.legend(loc='upper left')
ax2.grid(axis='y')

plt.tight_layout()
plt.show()
