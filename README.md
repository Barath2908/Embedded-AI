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

![image](https://github.com/user-attachments/assets/e372f18b-a148-4af8-b1a8-3b01b7759444)
![image](https://github.com/user-attachments/assets/0d59a6e4-eed3-4884-87f2-be75c4cf645b)



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
![image](https://github.com/user-attachments/assets/e4018cbb-762c-4a63-8595-fcf830383b92)




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
![image](https://github.com/user-attachments/assets/0d59a6e4-eed3-4884-87f2-be75c4cf645b)





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
![image](https://github.com/user-attachments/assets/423f6cfb-af7a-4a65-8f46-075de2f1812a)

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

Below is a GitHub-style description for the provided code:
![image](https://github.com/user-attachments/assets/f66760dc-7c1a-4248-ab19-9d9006dc57f0)


---

### Grouped Bar Chart for Optimizer Accuracy Comparison on MCUNet-5FPS

**Description:**  
This Python script generates a grouped bar chart that visualizes the accuracy results across various datasets for different optimization strategies applied during fine-tuning of the MCUNet-5FPS model. The data in this plot is based on the performance of updating real quantized graphs (int8) compared to the floating-point (fp32) baseline. The chart compares the following methods:
- **fp32 SGD-M**
- **int8 SGD-M**
- **Adam**
- **LARS**
- **SGD-M+QAS**

**Key Features:**
- **Data Visualization:**  
  Displays the accuracy percentages for each optimizer across nine different datasets (e.g., "Avg Acc.", "Cars", "CF10", "CF100", "CUB", "Flowers", "Food", "Pets", "VWW").
  
- **Grouped Bar Chart:**  
  Uses a grouped bar chart to facilitate side-by-side comparison of the optimizers, allowing users to quickly identify performance differences.

- **Customization and Clarity:**  
  The script includes labels for the x-axis, y-axis, a chart title, a legend, and grid lines for improved readability.

**Usage:**  
1. Ensure that you have Python installed along with the necessary libraries (`numpy` and `matplotlib`).
2. Run the script in your preferred Python environment. The script will open a window displaying the grouped bar chart.

![image](https://github.com/user-attachments/assets/80d9d19b-95b7-4a42-ad20-f27b7a644bc7)




