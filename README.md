### 1. Gradient Ratio Simulation with Quantization Aware Scaling (QAS)

**Description:**  
This script simulates how quantization affects the gradient-to-weight ratio during training. It compares three scenarios:  
- **FP32 Training:** Standard floating-point computations.  
- **Quantized Training without QAS:** Gradients are scaled down by the square of the quantization scaling factor, simulating reduced magnitude in int8 training.  
- **Quantized Training with QAS:** Applies Quantization Aware Scaling to restore the gradient magnitude to its FP32 level.

**Key Features:**
- Generates 100 iterations of simulated FP32 gradient values.
- Computes the weight-to-gradient ratio for each scenario.
- Visualizes the ratio progression over iterations using line plots.
- Prints average ratios for quick analysis.

**Usage:**  
Run the script to observe the impact of quantization and QAS on the gradient dynamics. It is useful for understanding and debugging quantization effects in neural network training.

---

### 2. Memory Cost Simulation for Sparse vs. Full Updates

**Description:**  
This code estimates and visualizes memory costs across 5 network layers, comparing a full update strategy with a sparse update strategy that only updates layers whose memory cost is below a set threshold.

**Key Features:**
- Defines per-layer memory costs in kilobytes (KB).
- Calculates the total memory for full updates (all layers) versus sparse updates (only layers under a given threshold).
- Plots the per-layer memory cost and highlights the layers chosen for sparse updates.
- Provides printed output for the full and sparse update memory costs.

**Usage:**  
Execute this script to evaluate the benefits of a sparse update strategy in reducing memory usage, especially in resource-constrained environments.

---

### 3. Comparison of Fake vs. Real Quantized Graph Outputs

**Description:**  
This snippet compares the outputs of a linear layer under three conditions:  
- **FP32 Computation:** The standard floating-point operation.  
- **Fake Quantized Graph:** Simulated by rounding the FP32 output to the nearest multiple of the scaling factor.  
- **Real Quantized Graph:** Emulates int8 quantization by quantizing both the weight and the input, performing integer multiplication, and then re-scaling the result.

**Key Features:**
- Simulates a linear layer operation with a single weight and varying inputs.
- Defines a `fake_quantize` function to simulate quantization without altering computation precision.
- Uses clipping to mimic int8 storage and demonstrates how quantized operations differ from FP32.
- Visualizes the output comparisons using a scatter plot.
- Prints sample values for side-by-side comparison of the outputs.

**Usage:**  
Run the script to visually and numerically compare the behavior of fake and real quantization. This is valuable for validating quantization techniques in model deployment.

---

### 4. Optimization of Sparse Update Configuration via Exhaustive Search

**Description:**  
This code performs an exhaustive search to determine the optimal combination of bias and weight updates across 5 network layers under a memory budget constraint. The objective is to maximize overall accuracy improvement while keeping the memory cost within limits.

**Key Features:**
- Considers binary decisions for bias updates (update or skip) and multiple ratios for weight updates.
- Uses predefined costs and improvement values for bias and weight updates.
- Iterates over all possible combinations to find the configuration with the highest total improvement under a specified memory budget.
- Visualizes per-layer memory cost breakdown (stacked bar chart) and cumulative improvement across layers.
- Displays the optimal configuration details along with per-layer annotations.

**Usage:**  
This script is ideal for experiments in resource-constrained training, where selective layer updates can yield the best performance under strict memory limitations.

---

### 5. Memory Usage Comparison: Vanilla vs. Optimized In-Place Backpropagation

**Description:**  
The final script compares two strategies for managing memory during the backpropagation of a neural network:
- **Vanilla Approach:** Stores all activations and gradients simultaneously.
- **Optimized In-Place Approach:** Stores all activations but only keeps one gradient at a time, thus reducing peak memory usage.

**Key Features:**
- Computes the memory requirements for activations and gradients based on network dimensions and batch size.
- Calculates cumulative memory usage across layers for both vanilla and optimized methods.
- Visualizes per-layer memory consumption with a stacked bar chart and shows cumulative memory usage using line plots.
- Annotates plots with memory values to facilitate clear comparisons.

**Usage:**  
Use this script to evaluate and visualize the benefits of an in-place memory optimization strategy during backpropagation, aiding in the design of more memory-efficient neural network training routines.

