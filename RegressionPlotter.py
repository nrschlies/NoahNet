import numpy as np
import matplotlib.pyplot as plt
import imageio
import os

# Generate synthetic data
np.random.seed(0)
N = 50  # Total number of data points
X = 2 * np.random.rand(N, 1)
true_slope = 3
true_intercept = 2
y = true_slope * X + true_intercept + np.random.randn(N, 1)

# Prepare for animation
filenames = []

# Compute pseudo-inverse solution using the entire dataset
X_b_full = np.hstack([np.ones((N, 1)), X])
theta_pseudo_inverse = np.linalg.pinv(X_b_full).dot(y)

# Initialize parameters for incremental analytical updates
theta_incremental_list = []

# Iterate over data points incrementally
for i in range(2, N+1):
    # Current subset of data
    X_i = X[:i]
    y_i = y[:i]
    
    # Add a column of ones to X_i for intercept term
    X_b = np.hstack([np.ones((i, 1)), X_i])
    
    # Analytical solution using normal equations (incremental)
    # Use pseudo-inverse in case X_b.T @ X_b is singular
    theta_incremental = np.linalg.pinv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_i)
    theta_incremental_list.append(theta_incremental)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Data Points')
    
    # Plot incremental analytical regression line
    X_plot = np.array([[0], [2]])
    y_plot_incremental = theta_incremental[0] + theta_incremental[1] * X_plot
    plt.plot(X_plot, y_plot_incremental, color='green', label='Incremental Analytical Update')
    
    # Plot pseudo-inverse regression line (constant)
    y_plot_pseudo_inverse = theta_pseudo_inverse[0] + theta_pseudo_inverse[1] * X_plot
    plt.plot(X_plot, y_plot_pseudo_inverse, color='red', linestyle='--', label='Pseudo-Inverse Solution')
    
    plt.title(f'Iteration {i-1}: Convergence of Analytical Updates')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.xlim(0, 2)
    plt.ylim(0, 10)
    
    # Save frame
    filename = f'frame_{i}.png'
    filenames.append(filename)
    plt.savefig(filename)
    plt.close()

# Create GIF
with imageio.get_writer('linear_regression_convergence.gif', mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove files
for filename in set(filenames):
    os.remove(filename)
