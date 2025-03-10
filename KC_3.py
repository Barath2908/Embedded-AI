import numpy as np
import matplotlib.pyplot as plt

# Define the quantization scaling factor
s = 0.02

# Create an array of random fp32 weights and inputs for a simple linear layer
num_samples = 50
# For simplicity, assume a single weight and single input per sample
W_fp32 = 1.5
x_fp32 = np.random.uniform(1.0, 3.0, size=num_samples)  # random inputs between 1.0 and 3.0

# Baseline FP32 computation: y = W * x
y_fp32 = W_fp32 * x_fp32

# Fake Quantized Graph:
# We simulate fake quantization by rounding the output to nearest multiple of s.
def fake_quantize(val, scale):
    return np.round(val / scale) * scale

y_fake = fake_quantize(y_fp32, s)

# Real Quantized Graph:
# Quantize the weight and input to int8 (simulate by rounding)
W_int8 = np.clip(np.round(W_fp32 / s), -128, 127)  # simulate int8 storage
x_int8 = np.clip(np.round(x_fp32 / s), -128, 127)    # simulate int8 storage

# Perform integer multiplication (simulate in int32)
y_int8 = W_int8 * x_int8

# Rescale the integer result back to float32 using s^2
y_real = y_int8 * (s ** 2)

# Plot the outputs for comparison
plt.figure(figsize=(12, 6))
plt.plot(x_fp32, y_fp32, 'bo', label='FP32 Output')
plt.plot(x_fp32, y_fake, 'gx', label='Fake Quantized Output')
plt.plot(x_fp32, y_real, 'r^', label='Real Quantized Output')
plt.xlabel('Input (FP32)')
plt.ylabel('Output')
plt.title('Comparison: Fake vs. Real Quantized Graph Outputs')
plt.legend()
plt.grid(True)
plt.show()

# Print sample values for clarity
for i in range(5):
    print(f"Sample {i+1}: x_fp32 = {x_fp32[i]:.3f}, FP32 y = {y_fp32[i]:.3f}, "
          f"Fake Quantized y = {y_fake[i]:.3f}, Real Quantized y = {y_real[i]:.3f}")
