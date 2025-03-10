import numpy as np
import matplotlib.pyplot as plt

# --- Simulation Parameters ---
num_iterations = 100
W_fp32 = 1.5                   # Example fp32 weight
s = 0.02                       # Quantization scaling factor

# Simulate 100 iterations of fp32 gradients (absolute values)
grad_fp32 = np.abs(np.random.normal(loc=0.2, scale=0.05, size=num_iterations))

# FP32 weight-to-gradient ratio (ideal)
ratio_fp32 = W_fp32 / grad_fp32

# Scenario 1: Without QAS (assume quantized training yields gradient reduced by factor s^2)
# In our simulation, we assume that the quantized gradient is: G_int8 = G_fp32 * s^2
grad_int8 = grad_fp32 * (s**2)
ratio_int8 = W_fp32 / grad_int8

# Scenario 2: With QAS applied to restore the gradient magnitude
# We assume that QAS multiplies the quantized gradient by (1/s^2) to recover the fp32 gradient.
grad_qas = grad_int8 / (s**2)
ratio_qas = W_fp32 / grad_qas

# Plot the ratios over iterations
plt.figure(figsize=(10,5))
plt.plot(ratio_fp32, label="FP32 Ratio", marker="o")
plt.plot(ratio_int8, label="Quantized Ratio (No QAS)", marker="x")
plt.plot(ratio_qas, label="Quantized Ratio (With QAS)", marker="s")
plt.xlabel("Iteration")
plt.ylabel("Weight / Gradient Ratio")
plt.title("Simulation of Gradient Ratio in Different Settings")
plt.legend()
plt.show()

# Print average ratios for clarity
print("Average FP32 Ratio:", np.mean(ratio_fp32))
print("Average Quantized Ratio (No QAS):", np.mean(ratio_int8))
print("Average Quantized Ratio (With QAS):", np.mean(ratio_qas))
