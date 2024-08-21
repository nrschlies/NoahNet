import matplotlib.pyplot as plt
import numpy as np

# Load the loss values from the file
loss_values = []
with open('loss_values.txt', 'r') as file:
    for line in file:
        loss_values.append(float(line.strip()))

# Plot the loss values
plt.figure(figsize=(10, 6))
plt.plot(loss_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Decreasing Over Training Epochs')
plt.show()

# Load the predicted and original values from the files
predicted_values = np.loadtxt('predicted_values.txt')
original_values = np.loadtxt('original_values.txt')

# Plot the original and predicted values for the first dimension
plt.figure(figsize=(10, 6))
plt.plot(original_values[:, 0], label='Original Sine Wave', linewidth=2)
plt.plot(predicted_values[:, 0], linestyle='--', label='Predicted Sine Wave', linewidth=2)

plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('Original vs Predicted Sine Wave (First Dimension)')
plt.legend()
plt.show()
