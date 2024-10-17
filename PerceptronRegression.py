import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from matplotlib.lines import Line2D

# -----------------------------------
# Hyperparameters for Perceptron and SVM
# -----------------------------------
LEARNING_RATE_PERCEPTRON = 0.01       # Learning rate for Perceptron
LEARNING_RATE_SVM = 0.005            # Learning rate for SVM
MAX_ITER_PERCEPTRON = 10             # Maximum number of iterations for Perceptron
MAX_ITER_SVM = 1000                 # Maximum number of iterations for SVM
REGULARIZATION_C = 100              # Regularization parameter C for SVM
WEIGHT_DECAY = 1e-2                  # Weight decay for SVM (stabilization)
MAX_ALLOWED_FRAMES = 100             # Maximum number of frames for animation

# Logging function for extensive terminal output
def log(message):
    print(message)

# 1. Generate a linearly separable dataset
np.random.seed(42)
N = 20  # Total number of data points per class
X_positive = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], N)
X_negative = np.random.multivariate_normal([0, 0], [[0.5, 0], [0, 0.5]], N)

X = np.vstack((X_positive, X_negative))
y = np.hstack((np.ones(N), -1 * np.ones(N))).reshape(-1, 1)  # Labels: +1 and -1

# Add bias term to X
X_b = np.hstack((np.ones((2*N, 1)), X))  # Shape: (2N, 3)

# 2. Compute the analytical solution using Moore-Penrose pseudo-inverse
theta_analytical = np.linalg.pinv(X_b).dot(y)
log(f"Analytical solution theta: {theta_analytical.flatten()}")

# 3. Implement the Perceptron algorithm with detailed calculations
def perceptron(X, y, learning_rate=LEARNING_RATE_PERCEPTRON, max_iter=MAX_ITER_PERCEPTRON):
    theta = np.zeros((X.shape[1], 1))
    theta_list = [theta.copy()]
    calculations = []
    converged = False  # Track convergence explicitly

    for epoch in range(1, max_iter + 1):
        errors = 0
        log(f"Perceptron - Epoch {epoch}")
        for i in range(len(y)):
            prediction = X[i].dot(theta)
            condition = y[i] * prediction
            log(f"  Iteration {i}, Prediction: {prediction[0]}, Condition: {condition[0]}")
            if condition <= 0:
                update = learning_rate * y[i] * X[i].reshape(-1, 1)
                theta += update
                errors += 1
                theta_list.append(theta.copy())
                log(f"    Update applied. Theta: {theta.flatten()}, Errors: {errors}")
                calculations.append({
                    'epoch': epoch,
                    'iteration': len(theta_list) - 1,
                    'theta': theta.copy(),
                    'y_i': y[i],
                    'X_i': X[i].reshape(-1, 1),
                    'prediction': prediction,
                    'condition': condition,
                    'update': update
                })
        log(f"  End of Epoch {epoch}, Total Errors: {errors}")
        if errors == 0:
            converged = True
            log(f"Perceptron converged after {epoch} epochs.")
            break
    return theta, theta_list, calculations, converged

# 4. Implement a more stable SVM with hinge loss and weight decay
def svm(X, y, learning_rate=LEARNING_RATE_SVM, max_iter=MAX_ITER_SVM, C=REGULARIZATION_C, weight_decay=WEIGHT_DECAY):
    theta = np.zeros((X.shape[1], 1))
    theta_list = [theta.copy()]
    calculations = []
    converged = False  # Track convergence explicitly

    for epoch in range(1, max_iter + 1):
        errors = 0  # Track errors for SVM
        log(f"SVM - Epoch {epoch}")
        for i in range(len(y)):
            prediction = X[i].dot(theta)
            margin = y[i] * prediction
            log(f"  Iteration {i}, Margin: {margin[0]}")
            if margin < 1:
                # Apply hinge loss gradient: penalize margin violation
                update = learning_rate * (y[i] * X[i].reshape(-1, 1) - (2 / C) * theta)
                theta += update
                errors += 1
            else:
                # Apply regularization gradient with weight decay
                update = learning_rate * (-2 / C * theta)
                theta += update
            # Add weight decay to prevent large weights
            theta -= learning_rate * weight_decay * theta
            
            theta_list.append(theta.copy())
            log(f"    Update applied. Theta: {theta.flatten()}, Errors: {errors}")
            calculations.append({
                'epoch': epoch,
                'iteration': len(theta_list) - 1,
                'theta': theta.copy(),
                'y_i': y[i],
                'X_i': X[i].reshape(-1, 1),
                'prediction': prediction,
                'margin': margin,
                'update': update
            })
        log(f"  End of Epoch {epoch}, Total Errors: {errors}")
        if errors == 0:
            converged = True
            log(f"SVM converged after {epoch} epochs.")
            break
    return theta, theta_list, calculations, converged

# Train the Perceptron
log("Training Perceptron...")
theta_perceptron, theta_list_perceptron, calculations_perceptron, perceptron_converged = perceptron(X_b, y)
log(f"Final Perceptron theta: {theta_perceptron.flatten()}")

# Train the SVM
log("Training SVM...")
theta_svm, theta_list_svm, calculations_svm, svm_converged = svm(X_b, y)
log(f"Final SVM theta: {theta_svm.flatten()}")

# Debugging step: Log the sizes of theta lists
log(f"Perceptron theta_list size: {len(theta_list_perceptron)}")
log(f"SVM theta_list size: {len(theta_list_svm)}")

# Cap the maximum number of frames generated for the animation
theta_list_perceptron = theta_list_perceptron[:MAX_ALLOWED_FRAMES]
theta_list_svm = theta_list_svm[:MAX_ALLOWED_FRAMES]

# 5. Animate the Perceptron and SVM learning processes with calculations
filenames = []

# Prepare grid for decision boundary plotting
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

# Create a directory for frames
if not os.path.exists('frames'):
    os.makedirs('frames')

# Determine the total number of frames based on the larger of the two iteration lists
max_iterations = min(max(len(theta_list_perceptron), len(theta_list_svm)), MAX_ALLOWED_FRAMES)
log(f"Generating {max_iterations} frames for the animation.")

# Iterate through each iteration, ensuring we stop based on convergence and max iterations
for idx in range(max_iterations):
    log(f"Rendering frame {idx+1}/{max_iterations}...")
    # Handle cases where the Perceptron has finished but the SVM is still updating
    if idx < len(theta_list_perceptron):
        theta_perceptron = theta_list_perceptron[idx]
    else:
        theta_perceptron = theta_list_perceptron[-1]  # Perceptron remains static if finished

    # Handle cases where the SVM is still updating
    if idx < len(theta_list_svm):
        theta_svm = theta_list_svm[idx]
    else:
        theta_svm = theta_list_svm[-1]  # SVM remains static if finished
    
    plt.figure(figsize=(10, 8))
    # Plot data points
    plt.scatter(X_positive[:, 0], X_positive[:, 1], color='blue', label='Positive Class (+1)')
    plt.scatter(X_negative[:, 0], X_negative[:, 1], color='red', label='Negative Class (-1)')

    # Plot analytical decision boundary
    z_analytical = theta_analytical[0] + theta_analytical[1] * xx + theta_analytical[2] * yy
    plt.contour(xx, yy, z_analytical, levels=[0], colors='green', linestyles='dashed', linewidths=2)

    # Plot Perceptron decision boundary
    z_perceptron = theta_perceptron[0] + theta_perceptron[1] * xx + theta_perceptron[2] * yy
    plt.contour(xx, yy, z_perceptron, levels=[0], colors='purple', linewidths=2)

    # Plot SVM decision boundary
    z_svm = theta_svm[0] + theta_svm[1] * xx + theta_svm[2] * yy
    plt.contour(xx, yy, z_svm, levels=[0], colors='orange', linewidths=2)

    # Display mathematical calculations for Perceptron and SVM
    if idx < len(calculations_perceptron):
        calc_perceptron = calculations_perceptron[idx]
        equation_perceptron = (f'Epoch: {calc_perceptron["epoch"]}, Iteration: {calc_perceptron["iteration"]}\n'
                               f'Perceptron Prediction: {calc_perceptron["prediction"][0]:.2f}\n'
                               f'Condition: {calc_perceptron["condition"][0]:.2f} ≤ 0\n'
                               f'θ_new Perceptron = {calc_perceptron["theta"].flatten()}')

        plt.text(0.05, 0.95, equation_perceptron, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if idx < len(calculations_svm):
        calc_svm = calculations_svm[idx]
        equation_svm = (f'Epoch: {calc_svm["epoch"]}, Iteration: {calc_svm["iteration"]}\n'
                        f'SVM Margin: {calc_svm["margin"][0]:.2f}\n'
                        f'θ_new SVM = {calc_svm["theta"].flatten()}')

        plt.text(0.05, 0.85, equation_svm, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.title(f'Perceptron & SVM Learning Iteration {idx}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Custom legend for Perceptron, SVM, and Analytical solution
    custom_lines = [
        Line2D([0], [0], color='blue', marker='o', linestyle='None', label='Positive Class (+1)'),
        Line2D([0], [0], color='red', marker='o', linestyle='None', label='Negative Class (-1)'),
        Line2D([0], [0], color='green', linestyle='dashed', label='Analytical Solution'),
        Line2D([0], [0], color='purple', label='Perceptron Solution'),
        Line2D([0], [0], color='orange', label='SVM Solution'),
    ]
    
    plt.legend(handles=custom_lines, loc='upper left', ncol=2, bbox_to_anchor=(0.1, 1.1))

    # Save frame
    filename = f'frames/frame_{idx}.png'
    filenames.append(filename)
    plt.savefig(filename)
    plt.close()

# Add the final frame multiple times to signify that the perceptron and SVM have finished
log("Adding final frames to signify end of learning.")
for _ in range(10):  # Repeat the last frame 10 times
    filenames.append(filenames[-1])

# Create GIF with reduced duration per frame
log("Creating GIF...")
with imageio.get_writer('perceptron_vs_svm_vs_analytical.gif', mode='I', duration=0.1) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove frames directory
log("Cleaning up...")
import shutil
shutil.rmtree('frames')

log("Process completed successfully.")
