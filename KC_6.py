import numpy as np
import matplotlib.pyplot as plt

# Dataset names corresponding to the columns in the table.
datasets = ["Avg Acc.", "Cars", "CF10", "CF100", "CUB", "Flowers", "Food", "Pets", "VWW"]

# List of techniques/optimizers.
techniques = ["fp32 SGD-M", "int8 SGD-M", "Adam", "LARS", "SGD-M+QAS"]

# Accuracy values for each technique (rows) corresponding to each dataset.
data = np.array([
    [56.7, 86.0, 63.4, 56.2, 88.8, 67.1, 79.5, 88.7, 73.3],  # fp32 SGD-M
    [31.2, 75.4, 54.5, 55.1, 84.5, 52.5, 81.0, 85.4, 64.9],  # int8 SGD-M
    [54.0, 84.5, 61.0, 58.5, 87.2, 62.6, 80.1, 86.5, 71.8],  # Adam
    [5.1, 64.8, 39.5, 9.6, 28.8, 46.5, 39.1, 85.0, 39.8],  # LARS
    [55.2, 86.9, 64.6, 57.8, 89.1, 64.4, 80.9, 89.3, 73.0]  # SGD-M+QAS
])

num_datasets = len(datasets)
num_techniques = len(techniques)

# Setup positions for grouped bars.
x = np.arange(num_datasets)
width = 0.15  # width of each bar

plt.figure(figsize=(12, 6))

# Create a bar for each technique.
for i in range(num_techniques):
    plt.bar(x + (i - num_techniques / 2) * width + width / 2, data[i], width, label=techniques[i])

# Labels, title and grid.
plt.xlabel("Datasets")
plt.ylabel("Accuracy (%)")
plt.title("Comparison of Optimizers on MCUNet-5FPS Model")
plt.xticks(x, datasets)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
