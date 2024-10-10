"""
The Evolutionary Network: An Adaptive Neural Architectural Optimizer

@author Noah Schliesman nschliesman@sandiego.edu

"""

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Utility
import numpy as np
import random
from copy import deepcopy
from tqdm import tqdm
import time
import logging

# Plotting
import matplotlib.pyplot as plt

# Hyperparameters
input_size = 784             # Example input size for MNIST (28x28 images flattened)
output_size = 10             # Example output size for classification (10 classes)
initial_hidden_units = 1     
learning_rate = 0.001        # Learning rate for optimizer
batch_size = 64         
num_epochs = 100            

# Adaptation/Architecture Evolution Parameters
adaptation_frequency = 1     # Frequency of architecture adaptation (in epochs)
reward_threshold = 0.05      # Minimum reward required to trigger an architectural change
time_complexity_weight = 0.1 # Weight for balancing reward vs. time complexity
max_layers = 10              # Maximum number of layers allowed in the network
min_units_per_layer = 1      # Minimum units allowed in any layer
max_units_per_layer = 1024   # Maximum units allowed in any layer
adaptation_actions = ['add_layer', 'remove_layer', 'change_layer', 'add_units', 'remove_units', 
                      'add_normalization', 'remove_normalization', 'add_residual', 'remove_residual']

# Debugging and Logging Flags
DEBUG_MODE = True            # Toggle for debugging mode
LOG_GRADIENTS = True         # Flag to log gradients
MONITOR_ARCHITECTURE = True  # Flag to monitor architecture changes
DISPLAY_PLOTS = True         # Flag to display visualizations

# Reward Regularization Parameters
alpha = 0.9                  # Smoothing factor for temporal smoothing of rewards
lambda_penalty = 0.001       # Penalty coefficient for gradient magnitude regularization
gamma_sparsity = 0.01        # Coefficient for sparsity-inducing regularization
beta_validation = 0.5        # Weight of validation performance in reward calculation

# Optimizer and Criterion
criterion = nn.CrossEntropyLoss()  # Example loss function for classification tasks
optimizer_type = 'Adam'            # Optimizer type (can be Adam, SGD, etc.)

# Random seed for reproducibility
random_seed = 42
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Device configuration for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_units):
        super(SimpleNN, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_units)
        self.hidden_layer = nn.ReLU()
        self.output_layer = nn.Linear(hidden_units, output_size)
        
        # Dynamic layers storage
        self.dynamic_layers = nn.ModuleList()
        self.is_conv = False  # Flag to indicate if the last added layer is a convolutional layer
        
        # Debugging initialization
        if DEBUG_MODE:
            logging.info("Network Initialized: %s", self)
            for name, param in self.named_parameters():
                logging.info(f"Layer: {name} | Size: {param.size()} | Values : {param[:5]}")

    def forward(self, x):
        # Check if input is a 4D tensor (batch size, channels, height, width) for images
        if len(x.shape) == 4:  # Input from DataLoader might be 4D (batch_size, channels, height, width)
            x = x.view(x.size(0), -1)  # Flatten the input to (batch_size, 784)
        else:
            x = x.view(-1, 28 * 28)  # Explicitly flatten for cases where the input might be 2D

        # Forward pass through the input layer and hidden layer
        x = self.input_layer(x)
        x = self.hidden_layer(x)

        activations = [x]

        # Pass through dynamically added layers
        for i, layer in enumerate(self.dynamic_layers):
            if isinstance(layer, (nn.LSTM, nn.GRU)):
                x, _ = layer(x)  # LSTM and GRU return (output, (hidden_state, cell_state))
            elif isinstance(layer, nn.Conv2d):
                x = layer(x)
                self.is_conv = True
            elif isinstance(layer, nn.Transformer):
                x = layer(x, x)  # For simplicity, use the same input for src and tgt
            else:
                x = layer(x)

            activations.append(x)

            # Apply residual connections if any
            for (layer_a_index, layer_b_index) in self.residual_connections:
                if i == layer_b_index:
                    x += activations[layer_a_index]

        # If the last dynamic layer is convolutional, flatten the output
        if self.is_conv:
            x = torch.flatten(x, 1)  # Flatten all dimensions except batch size

        # Final output layer
        x = self.output_layer(x)
        return x

    def add_layer(self, layer_type, units):
        # Determine input size based on the previous layer
        if len(self.dynamic_layers) > 0:
            if isinstance(self.dynamic_layers[-1], nn.Linear):
                last_layer_output_size = self.dynamic_layers[-1].out_features
            elif isinstance(self.dynamic_layers[-1], nn.Conv2d):
                last_layer_output_size = self.dynamic_layers[-1].out_channels
            elif isinstance(self.dynamic_layers[-1], (nn.LSTM, nn.GRU)):
                last_layer_output_size = self.dynamic_layers[-1].hidden_size
            elif isinstance(self.dynamic_layers[-1], nn.Transformer):
                last_layer_output_size = units  # In Transformers, we often set the output size manually
            else:
                last_layer_output_size = self.output_layer.in_features  # Fall back to input size if no dynamic layers
        else:
            last_layer_output_size = self.output_layer.in_features

        # Adding support for different layer types including fully connected, LSTM, GRU, Transformer
        if layer_type == 'fully_connected':
            new_layer = nn.Linear(last_layer_output_size, units)
            self.dynamic_layers.append(new_layer)
            self.output_layer = nn.Linear(units, self.output_layer.out_features)

        elif layer_type == 'relu':
            new_layer = nn.ReLU()
            self.dynamic_layers.append(new_layer)

        elif layer_type == 'conv':
            input_channels = 1 if len(self.dynamic_layers) == 0 else self.dynamic_layers[-1].out_channels
            new_layer = nn.Conv2d(in_channels=input_channels, out_channels=units, kernel_size=3, stride=1, padding=1)
            self.dynamic_layers.append(new_layer)
            self.is_conv = True  # Update flag for conv layers

        elif layer_type == 'lstm':
            new_layer = nn.LSTM(last_layer_output_size, units, batch_first=True)
            self.dynamic_layers.append(new_layer)

        elif layer_type == 'gru':
            new_layer = nn.GRU(last_layer_output_size, units, batch_first=True)
            self.dynamic_layers.append(new_layer)

        elif layer_type == 'transformer':
            new_layer = nn.Transformer(nhead=8, num_encoder_layers=6, dim_feedforward=units)
            self.dynamic_layers.append(new_layer)

        # Debugging: Log the new layer addition
        if DEBUG_MODE:
            logging.info(f"Added layer: {layer_type} with {units} units.")
            logging.info(f"Updated network structure: {self}")

    def remove_layer(self, layer_index):
        # Ensure the layer index is valid
        if layer_index < 0 or layer_index >= len(self.dynamic_layers):
            raise ValueError(f"Invalid layer index {layer_index}. Must be between 0 and {len(self.dynamic_layers)-1}.")
        
        # Remove the layer from dynamic_layers
        removed_layer = self.dynamic_layers[layer_index]
        del self.dynamic_layers[layer_index]

        # If the removed layer is a convolutional layer, update the `is_conv` flag
        if isinstance(removed_layer, nn.Conv2d):
            # Check if there are any other conv layers left
            self.is_conv = any(isinstance(layer, nn.Conv2d) for layer in self.dynamic_layers)
        
        # Debugging: Log the layer removal
        if DEBUG_MODE:
            logging.info(f"Removed layer at index {layer_index}: {removed_layer}")
            logging.info(f"Updated network structure: {self}")

    def change_layer_type(self, layer_index, new_layer_type):
        # Ensure the layer index is valid
        if layer_index < 0 or layer_index >= len(self.dynamic_layers):
            raise ValueError(f"Invalid layer index {layer_index}. Must be between 0 and {len(self.dynamic_layers)-1}.")

        current_layer = self.dynamic_layers[layer_index]
        input_size = self.input_layer.out_features if layer_index == 0 else self.dynamic_layers[layer_index - 1].out_features

        # Changing to LSTM, GRU, and Transformer layers
        if new_layer_type == 'fully_connected':
            new_layer = nn.Linear(input_size, current_layer.out_features)

        elif new_layer_type == 'relu':
            new_layer = nn.ReLU()

        elif new_layer_type == 'conv':
            input_channels = 1 if layer_index == 0 else self.dynamic_layers[layer_index - 1].out_channels
            new_layer = nn.Conv2d(in_channels=input_channels, out_channels=current_layer.out_features, kernel_size=3, stride=1, padding=1)

        elif new_layer_type == 'lstm':
            new_layer = nn.LSTM(input_size, current_layer.hidden_size, batch_first=True)

        elif new_layer_type == 'gru':
            new_layer = nn.GRU(input_size, current_layer.hidden_size, batch_first=True)

        elif new_layer_type == 'transformer':
            new_layer = nn.Transformer(nhead=8, num_encoder_layers=6, dim_feedforward=current_layer.out_features)

        # Replace the old layer with the new one
        self.dynamic_layers[layer_index] = new_layer

        # Debugging: Log the layer change
        if DEBUG_MODE:
            logging.info(f"Changed layer at index {layer_index} to {new_layer_type}")
            logging.info(f"Updated network structure: {self}")

            
    def add_normalization(self, layer_index, normalization_type):
        # Ensure the layer index is valid
        if layer_index < 0 or layer_index >= len(self.dynamic_layers):
            raise ValueError(f"Invalid layer index {layer_index}. Must be between 0 and {len(self.dynamic_layers)-1}.")

        # Get the layer after which to add normalization
        current_layer = self.dynamic_layers[layer_index]

        # Determine the normalization layer based on the current layer's output size
        if isinstance(current_layer, nn.Linear):
            num_features = current_layer.out_features
        elif isinstance(current_layer, nn.Conv2d):
            num_features = current_layer.out_channels
        else:
            raise ValueError(f"Normalization not supported for the layer type at index {layer_index}")

        # Add the requested normalization
        if normalization_type == 'batch':
            if isinstance(current_layer, nn.Conv2d):
                normalization_layer = nn.BatchNorm2d(num_features)
            else:
                normalization_layer = nn.BatchNorm1d(num_features)
        elif normalization_type == 'layer':
            normalization_layer = nn.LayerNorm(num_features)
        else:
            raise ValueError(f"Unsupported normalization type: {normalization_type}")

        # Insert the normalization layer right after the current layer
        self.dynamic_layers.insert(layer_index + 1, normalization_layer)

        # Debugging: Log the normalization addition
        if DEBUG_MODE:
            logging.info(f"Added {normalization_type} normalization after layer {layer_index}")
            logging.info(f"Updated network structure: {self}")
            
    def remove_normalization(self, layer_index):
        # Ensure the layer index is valid
        if layer_index < 0 or layer_index >= len(self.dynamic_layers):
            raise ValueError(f"Invalid layer index {layer_index}. Must be between 0 and {len(self.dynamic_layers)-1}.")

        # Check if the layer at the specified index is a normalization layer
        layer_to_remove = self.dynamic_layers[layer_index]
        if not isinstance(layer_to_remove, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            raise ValueError(f"Layer at index {layer_index} is not a normalization layer. Unable to remove.")

        # Remove the normalization layer
        del self.dynamic_layers[layer_index]

        # Debugging: Log the normalization removal
        if DEBUG_MODE:
            logging.info(f"Removed normalization layer at index {layer_index}: {layer_to_remove}")
            logging.info(f"Updated network structure: {self}")

    def add_residual_connection(self, layer_a_index, layer_b_index):
        # Ensure both layer indices are valid
        if layer_a_index < 0 or layer_a_index >= len(self.dynamic_layers):
            raise ValueError(f"Invalid layer A index {layer_a_index}. Must be between 0 and {len(self.dynamic_layers)-1}.")
        if layer_b_index < 0 or layer_b_index >= len(self.dynamic_layers):
            raise ValueError(f"Invalid layer B index {layer_b_index}. Must be between 0 and {len(self.dynamic_layers)-1}.")
        if layer_a_index >= layer_b_index:
            raise ValueError(f"Layer A index {layer_a_index} must be less than layer B index {layer_b_index}.")

        # Get the output size of layer A and input size of layer B
        layer_a = self.dynamic_layers[layer_a_index]
        layer_b = self.dynamic_layers[layer_b_index]

        if isinstance(layer_a, nn.Linear):
            layer_a_output_size = layer_a.out_features
        elif isinstance(layer_a, nn.Conv2d):
            layer_a_output_size = layer_a.out_channels
        else:
            raise ValueError(f"Unsupported layer type at index {layer_a_index} for residual connection.")

        if isinstance(layer_b, nn.Linear):
            layer_b_input_size = layer_b.in_features
        elif isinstance(layer_b, nn.Conv2d):
            layer_b_input_size = layer_b.in_channels
        else:
            raise ValueError(f"Unsupported layer type at index {layer_b_index} for residual connection.")

        # Ensure the dimensions of layer A's output match the dimensions of layer B's input
        if layer_a_output_size != layer_b_input_size:
            raise ValueError(f"Mismatch between layer A output size ({layer_a_output_size}) and layer B input size ({layer_b_input_size}).")

        # Store the residual connection as a tuple (layer_a_index, layer_b_index) for use in the forward pass
        self.residual_connections.append((layer_a_index, layer_b_index))

        # Debugging: Log the addition of the residual connection
        if DEBUG_MODE:
            logging.info(f"Added residual connection from layer {layer_a_index} to layer {layer_b_index}.")
            logging.info(f"Updated network structure: {self}")

    def remove_residual_connection(self, layer_a_index, layer_b_index):
        # Ensure both layer indices are valid
        if layer_a_index < 0 or layer_a_index >= len(self.dynamic_layers):
            raise ValueError(f"Invalid layer A index {layer_a_index}. Must be between 0 and {len(self.dynamic_layers)-1}.")
        if layer_b_index < 0 or layer_b_index >= len(self.dynamic_layers):
            raise ValueError(f"Invalid layer B index {layer_b_index}. Must be between 0 and {len(self.dynamic_layers)-1}.")
        if layer_a_index >= layer_b_index:
            raise ValueError(f"Layer A index {layer_a_index} must be less than layer B index {layer_b_index}.")

        # Check if the specified residual connection exists
        connection = (layer_a_index, layer_b_index)
        if connection not in self.residual_connections:
            raise ValueError(f"No residual connection found between layer {layer_a_index} and layer {layer_b_index}.")

        # Remove the residual connection
        self.residual_connections.remove(connection)

        # Debugging: Log the removal of the residual connection
        if DEBUG_MODE:
            logging.info(f"Removed residual connection from layer {layer_a_index} to layer {layer_b_index}.")
            logging.info(f"Updated network structure: {self}")

    def initialize_weights(layer):
        if isinstance(layer, nn.Linear):
            # Initialize fully connected layers using Kaiming initialization
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

        elif isinstance(layer, nn.Conv2d):
            # Initialize convolutional layers using Kaiming initialization
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)

        elif isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.BatchNorm2d):
            # Initialize batch normalization layers
            nn.init.constant_(layer.weight, 1.0)
            nn.init.constant_(layer.bias, 0.0)

        elif isinstance(layer, nn.LayerNorm):
            # Initialize layer normalization layers
            nn.init.constant_(layer.weight, 1.0)
            nn.init.constant_(layer.bias, 0.0)

        # Debugging: Log the initialization of the layer
        if DEBUG_MODE:
            logging.info(f"Initialized weights for layer: {layer}")

    def calculate_gradient(layer_output, loss):
        # Ensure that gradients are being tracked
        if not layer_output.requires_grad:
            raise ValueError("The layer's output must have requires_grad=True to calculate gradients.")

        # Zero out any existing gradients
        if layer_output.grad is not None:
            layer_output.grad.zero_()

        # Perform backpropagation to calculate the gradient of the loss with respect to the layer output
        loss.backward(retain_graph=True)

        # Debugging: Log the gradients if enabled
        if LOG_GRADIENTS:
            logging.info(f"Gradients for layer output: {layer_output.grad}")

        # Return the gradient of the layer's output with respect to the loss
        return layer_output.grad
    
    def evaluate_task_performance(network, task_type, performance_metrics):
        network.eval()  # Set the network to evaluation mode
        results = {metric: 0.0 for metric in performance_metrics}  # Initialize results

        if task_type == "classification":
            correct_predictions = 0
            total_samples = 0

            # Iterate through the dataset to compute accuracy
            for batch, (inputs, targets) in enumerate(performance_metrics['dataloader']):
                inputs, targets = inputs.to(device), targets.to(device)

                with torch.no_grad():  # Disable gradients during evaluation
                    outputs = network(inputs)
                    _, predicted = torch.max(outputs, 1)

                    # Compute accuracy
                    correct_predictions += (predicted == targets).sum().item()
                    total_samples += targets.size(0)

            # Store accuracy in the results
            results['accuracy'] = correct_predictions / total_samples

        elif task_type == "regression":
            total_loss = 0.0

            # Iterate through the dataset to compute mean squared error (MSE) or other metrics
            for batch, (inputs, targets) in enumerate(performance_metrics['dataloader']):
                inputs, targets = inputs.to(device), targets.to(device)

                with torch.no_grad():  # Disable gradients during evaluation
                    outputs = network(inputs)
                    loss = performance_metrics['criterion'](outputs, targets)

                    # Accumulate loss
                    total_loss += loss.item()

            # Store mean loss in the results
            results['mean_loss'] = total_loss / len(performance_metrics['dataloader'])

        # Add other task types or custom metrics as needed

        # Log results if debugging is enabled
        if DEBUG_MODE:
            logging.info(f"Task Type: {task_type}, Performance Metrics: {results}")

        return results

    def calculate_gradient_magnitude(gradient):
        if gradient is None:
            raise ValueError("Gradient tensor is None. Make sure to compute the gradient first.")

        # Calculate the L2 norm (magnitude) of the gradient
        gradient_magnitude = torch.norm(gradient, p=2).item()

        # Debugging: Log the gradient magnitude if enabled
        if LOG_GRADIENTS:
            logging.info(f"Gradient magnitude: {gradient_magnitude}")

        return gradient_magnitude

    def compute_gradient_reward(old_gradient_magnitude, new_gradient_magnitude):
        if old_gradient_magnitude == 0:
            raise ValueError("Old gradient magnitude cannot be zero for comparison.")

        # Calculate the percentage change in gradient magnitude
        gradient_change_ratio = (new_gradient_magnitude - old_gradient_magnitude) / old_gradient_magnitude

        # Apply a threshold to ensure that small changes do not result in unnecessary rewards
        if abs(gradient_change_ratio) < reward_threshold:
            gradient_reward = 0.0
        else:
            # Reward proportional to the magnitude change
            gradient_reward = gradient_change_ratio

        # Debugging: Log the gradient reward if enabled
        if DEBUG_MODE:
            logging.info(f"Computed gradient reward: {gradient_reward} (Old: {old_gradient_magnitude}, New: {new_gradient_magnitude})")

        return gradient_reward

    def calculate_time_complexity_fully_connected(input_size, units):
        time_complexity = input_size * units + units  # Adding units for the bias terms

        # Debugging: Log the time complexity if enabled
        if DEBUG_MODE:
            logging.info(f"Time complexity for fully connected layer (Input Size: {input_size}, Units: {units}): {time_complexity}")

        return time_complexity

    def calculate_time_complexity_convolutional(input_size, filter_size, filters):
        time_complexity = input_size * filter_size * filters
        if DEBUG_MODE:
            logging.info(f"Time complexity for convolutional layer (Input Size: {input_size}, Filter Size: {filter_size}, Filters: {filters}): {time_complexity}")
        return time_complexity

    def calculate_time_complexity_lstm(sequence_length, hidden_size):
        time_complexity = 4 * sequence_length * hidden_size ** 2
        if DEBUG_MODE:
            logging.info(f"Time complexity for LSTM (Sequence Length: {sequence_length}, Hidden Size: {hidden_size}): {time_complexity}")
        return time_complexity

    def calculate_time_complexity_gru(sequence_length, hidden_size):
        time_complexity = 3 * sequence_length * hidden_size ** 2
        if DEBUG_MODE:
            logging.info(f"Time complexity for GRU (Sequence Length: {sequence_length}, Hidden Size: {hidden_size}): {time_complexity}")
        return time_complexity

    def calculate_time_complexity_transformer(sequence_length, hidden_size):
        # Transformers involve multi-head attention, with a typical scaling factor of hidden_size^2
        time_complexity = sequence_length ** 2 * hidden_size
        if DEBUG_MODE:
            logging.info(f"Time complexity for Transformer (Sequence Length: {sequence_length}, Hidden Size: {hidden_size}): {time_complexity}")
        return time_complexity

    def calculate_time_complexity_state_space(sequence_length, system_dim):
        time_complexity = sequence_length * system_dim ** 2
        if DEBUG_MODE:
            logging.info(f"Time complexity for state-space model (Sequence Length: {sequence_length}, System Dimension: {system_dim}): {time_complexity}")
        return time_complexity

    def select_optimal_action(gradient_rewards, time_complexities):
        # Calculate the reward-to-complexity ratio for each action
        reward_to_time_ratios = []
        for reward, complexity in zip(gradient_rewards, time_complexities):
            if complexity > 0:
                ratio = reward / (complexity + time_complexity_weight * complexity)
            else:
                ratio = float('-inf')  # Handle edge case where complexity is zero
            reward_to_time_ratios.append(ratio)

        # Select the action with the highest reward-to-time ratio
        optimal_action_index = reward_to_time_ratios.index(max(reward_to_time_ratios))

        # Debugging: Log the selected action and its reward-to-time ratio
        if DEBUG_MODE:
            logging.info(f"Optimal action selected: {optimal_action_index} with reward-to-time ratio: {reward_to_time_ratios[optimal_action_index]}")

        return optimal_action_index

    def add_units(network, layer_index, num_units):
        # Ensure the layer index is valid
        if layer_index < 0 or layer_index >= len(network.dynamic_layers):
            raise ValueError(f"Invalid layer index {layer_index}. Must be between 0 and {len(network.dynamic_layers)-1}.")

        current_layer = network.dynamic_layers[layer_index]

        if not isinstance(current_layer, nn.Linear):
            raise ValueError(f"Layer at index {layer_index} is not a fully connected layer. Cannot add units.")

        # Get the current number of input and output features of the layer
        current_input_features = current_layer.in_features
        current_output_features = current_layer.out_features

        # Create a new layer with the increased number of output units
        new_output_features = current_output_features + num_units
        new_layer = nn.Linear(current_input_features, new_output_features)

        # Copy over the old weights and biases to the new layer
        with torch.no_grad():
            new_layer.weight[:current_output_features, :] = current_layer.weight
            if current_layer.bias is not None:
                new_layer.bias[:current_output_features] = current_layer.bias

        # Replace the old layer with the new layer in the dynamic layers list
        network.dynamic_layers[layer_index] = new_layer

        # Update the following layer's input size if it's a fully connected layer
        if layer_index + 1 < len(network.dynamic_layers) and isinstance(network.dynamic_layers[layer_index + 1], nn.Linear):
            following_layer = network.dynamic_layers[layer_index + 1]
            new_following_layer = nn.Linear(new_output_features, following_layer.out_features)

            # Copy over the weights and biases
            with torch.no_grad():
                new_following_layer.weight[:, :current_output_features] = following_layer.weight
                new_following_layer.bias = following_layer.bias

            network.dynamic_layers[layer_index + 1] = new_following_layer

        # Debugging: Log the addition of units
        if DEBUG_MODE:
            logging.info(f"Added {num_units} units to layer at index {layer_index}. New output size: {new_output_features}")
            logging.info(f"Updated network structure: {network}")

    def remove_units(network, layer_index, num_units):
        # Ensure the layer index is valid
        if layer_index < 0 or layer_index >= len(network.dynamic_layers):
            raise ValueError(f"Invalid layer index {layer_index}. Must be between 0 and {len(network.dynamic_layers)-1}.")

        current_layer = network.dynamic_layers[layer_index]

        if not isinstance(current_layer, nn.Linear):
            raise ValueError(f"Layer at index {layer_index} is not a fully connected layer. Cannot remove units.")

        # Get the current number of input and output features of the layer
        current_input_features = current_layer.in_features
        current_output_features = current_layer.out_features

        # Ensure the number of units to remove is valid
        if num_units >= current_output_features:
            raise ValueError(f"Cannot remove {num_units} units. Layer only has {current_output_features} units.")

        # Create a new layer with the reduced number of output units
        new_output_features = current_output_features - num_units
        new_layer = nn.Linear(current_input_features, new_output_features)

        # Copy over the old weights and biases to the new layer
        with torch.no_grad():
            new_layer.weight[:, :] = current_layer.weight[:new_output_features, :]
            if current_layer.bias is not None:
                new_layer.bias[:] = current_layer.bias[:new_output_features]

        # Replace the old layer with the new layer in the dynamic layers list
        network.dynamic_layers[layer_index] = new_layer

        # Update the following layer's input size if it's a fully connected layer
        if layer_index + 1 < len(network.dynamic_layers) and isinstance(network.dynamic_layers[layer_index + 1], nn.Linear):
            following_layer = network.dynamic_layers[layer_index + 1]
            new_following_layer = nn.Linear(new_output_features, following_layer.out_features)

            # Copy over the weights and biases
            with torch.no_grad():
                new_following_layer.weight[:new_output_features, :] = following_layer.weight[:new_output_features, :]
                new_following_layer.bias = following_layer.bias

            network.dynamic_layers[layer_index + 1] = new_following_layer

        # Debugging: Log the removal of units
        if DEBUG_MODE:
            logging.info(f"Removed {num_units} units from layer at index {layer_index}. New output size: {new_output_features}")
            logging.info(f"Updated network structure: {network}")

    def evaluate_architecture_change(network, layer_index, dataloader, criterion, optimizer):
        # Store original state
        original_state_dict = deepcopy(network.state_dict())
        original_loss = compute_loss_and_gradients()

        best_gradient_diff = float('-inf')
        best_change = None

        # Define actions and evaluate their impact
        actions = {
            'add_units': lambda: network.add_units(layer_index, num_units=10),
            'remove_units': lambda: network.remove_units(layer_index, num_units=10),
            'add_layer': lambda: network.add_layer('fully_connected', 64),
            'remove_layer': lambda: network.remove_layer(layer_index),
            'add_normalization': lambda: network.add_normalization(layer_index, 'batch'),
            'remove_normalization': lambda: network.remove_normalization(layer_index),
            'add_residual': lambda: network.add_residual_connection(layer_index - 1, layer_index),
            'remove_residual': lambda: network.remove_residual_connection(layer_index - 1, layer_index)
        }

        for action_name, apply_action in actions.items():
            try:
                apply_action()
                new_loss = compute_loss_and_gradients()
                gradient_diff = abs(original_loss - new_loss)

                if gradient_diff > best_gradient_diff:
                    best_gradient_diff = gradient_diff
                    best_change = action_name

            except Exception as e:
                logging.info(f"Failed to apply {action_name}: {str(e)}")

            # Restore original state
            network.load_state_dict(original_state_dict)

        logging.info(f"Best action: {best_change} with gradient difference: {best_gradient_diff}")
        return best_change

    def apply_architecture_change(network, change_type, layer_index, parameters=None):
        if change_type == 'add_units':
            num_units = parameters.get('num_units', 10)
            network.add_units(layer_index, num_units)

        elif change_type == 'remove_units':
            num_units = parameters.get('num_units', 10)
            network.remove_units(layer_index, num_units)

        elif change_type == 'add_layer':
            layer_type = parameters.get('layer_type', 'fully_connected')
            units = parameters.get('units', 64)
            network.add_layer(layer_type, units)

        elif change_type == 'remove_layer':
            network.remove_layer(layer_index)

        elif change_type == 'change_layer_type':
            new_layer_type = parameters.get('new_layer_type', 'fully_connected')
            network.change_layer_type(layer_index, new_layer_type)

        elif change_type == 'add_normalization':
            normalization_type = parameters.get('normalization_type', 'batch')
            network.add_normalization(layer_index, normalization_type)

        elif change_type == 'remove_normalization':
            network.remove_normalization(layer_index)

        elif change_type == 'add_residual':
            layer_a_index = parameters.get('layer_a_index', layer_index - 1)
            layer_b_index = parameters.get('layer_b_index', layer_index)
            network.add_residual_connection(layer_a_index, layer_b_index)

        elif change_type == 'remove_residual':
            layer_a_index = parameters.get('layer_a_index', layer_index - 1)
            layer_b_index = parameters.get('layer_b_index', layer_index)
            network.remove_residual_connection(layer_a_index, layer_b_index)

        else:
            raise ValueError(f"Unknown architecture change type: {change_type}")

        # Debugging: Log the architecture change
        if DEBUG_MODE:
            logging.info(f"Applied architecture change: {change_type} at layer {layer_index} with parameters {parameters}")
            logging.info(f"Updated network structure: {network}")

    def apply_temporal_smoothing(reward, previous_reward, alpha):
        smoothed_reward = alpha * reward + (1 - alpha) * previous_reward
        return smoothed_reward

    def apply_validation_based_regularization(reward, validation_performance, training_performance):
        performance_diff = validation_performance - training_performance
        regularized_reward = reward - beta_validation * max(0, performance_diff)

        return regularized_reward

    def apply_sparsity_regularization(reward, num_units, gamma):
        sparsity_penalty = gamma * num_units
        regularized_reward = reward - sparsity_penalty

        if DEBUG_MODE:
            logging.info(f"Sparsity Regularization Applied: {sparsity_penalty} penalty for {num_units} units. Updated reward: {regularized_reward}")

        return regularized_reward

    def apply_multi_task_regularization(reward_list, weights):
        # Ensure the lengths of reward_list and weights match
        if len(reward_list) != len(weights):
            raise ValueError("The length of reward_list and weights must be the same.")

        # Compute the weighted sum of rewards
        combined_reward = sum(r * w for r, w in zip(reward_list, weights))

        if DEBUG_MODE:
            logging.info(f"Multi-task Regularization Applied: Combined reward: {combined_reward}")

        return combined_reward

    def log_gradient_information(gradient):
        if gradient is None:
            logging.warning("Gradient is None. No information to log.")
            return

        # Log basic gradient information
        logging.info(f"Gradient shape: {gradient.shape}")
        logging.info(f"Gradient mean: {gradient.mean().item()}")
        logging.info(f"Gradient standard deviation: {gradient.std().item()}")

        # Optionally log more detailed gradient statistics
        if LOG_GRADIENTS:
            logging.info(f"Gradient min value: {gradient.min().item()}")
            logging.info(f"Gradient max value: {gradient.max().item()}")
            logging.info(f"First few gradient values: {gradient.flatten()[:5].tolist()}")

    def debug_gradient_magnitude(layer_index, gradient_magnitude):
        if gradient_magnitude is None:
            logging.warning(f"Layer {layer_index}: Gradient magnitude is None. Unable to log.")
            return

        # Log the gradient magnitude and layer index
        logging.info(f"Layer {layer_index}: Gradient magnitude = {gradient_magnitude}")

        # Provide further debugging details if enabled
        if DEBUG_MODE:
            if gradient_magnitude == 0:
                logging.warning(f"Layer {layer_index}: Gradient magnitude is zero. Check for potential issues in backpropagation.")
            elif gradient_magnitude > 1e6:
                logging.warning(f"Layer {layer_index}: Gradient magnitude is unusually large (>1e6). Consider gradient clipping.")
            elif gradient_magnitude < 1e-6:
                logging.warning(f"Layer {layer_index}: Gradient magnitude is very small (<1e-6). Check for vanishing gradients.")

    def monitor_computational_cost(units, time_complexity):
        # Log the number of units and time complexity
        logging.info(f"Monitoring computational cost: Units = {units}, Time Complexity = {time_complexity}")

        # Provide further insights based on computational cost
        if DEBUG_MODE:
            if time_complexity > 1e6:
                logging.warning(f"High computational cost detected: Time Complexity = {time_complexity}. Consider optimizing the architecture.")
            elif time_complexity < 1e3:
                logging.info(f"Low computational cost: Time Complexity = {time_complexity}. Architecture is efficient.")

        # Track time complexity relative to units for future decisions
        logging.info(f"Computational cost per unit: {time_complexity / units:.4f}")

    def debug_architecture_change(network, layer_index, change_type):
        """
        Logs detailed information when an architectural change is made in the network.

        Args:
            network (nn.Module): The network in which the change is occurring.
            layer_index (int): The index of the layer where the change is applied.
            change_type (str): The type of architectural change being made.
        """
        # Check for valid layer index
        if layer_index < 0 or layer_index >= len(network.dynamic_layers):
            logging.error(f"Invalid layer index: {layer_index}. Must be between 0 and {len(network.dynamic_layers) - 1}.")
            return

        # Log the architectural change type and layer information
        logging.info(f"Applying architectural change: {change_type} at layer index {layer_index}")

        # Log the layer's current structure before change
        if DEBUG_MODE:
            current_layer = network.dynamic_layers[layer_index]
            logging.info(f"Current layer structure before change: {current_layer}")
            logging.info(f"Layer parameters: {list(current_layer.parameters()) if hasattr(current_layer, 'parameters') else 'None'}")

        # After the change is applied, log the new structure
        if DEBUG_MODE:
            logging.info(f"Updated network structure after {change_type}: {network}")

        # Provide further debugging insights if necessary
        if change_type == 'add_layer':
            logging.info(f"New layer added at index {layer_index}. Total layers: {len(network.dynamic_layers)}")
        elif change_type == 'remove_layer':
            logging.info(f"Layer removed at index {layer_index}. Total layers: {len(network.dynamic_layers)}")
        elif change_type == 'change_layer_type':
            logging.info(f"Layer at index {layer_index} changed to a new type.")
        elif change_type == 'add_units' or change_type == 'remove_units':
            logging.info(f"Units adjusted in layer at index {layer_index}.")

    def check_for_vanishing_or_exploding_gradients(network):
        """
        Checks for vanishing or exploding gradients in the network by monitoring the gradients of each layer.

        Args:
            network (nn.Module): The neural network model to check.
        """
        # Iterate through each layer in the network
        for layer_index, layer in enumerate(network.children()):
            # Skip layers that do not have gradients
            if not hasattr(layer, 'weight') or layer.weight.grad is None:
                continue

            # Calculate the L2 norm (magnitude) of the gradients
            grad_magnitude = torch.norm(layer.weight.grad, p=2).item()

            # Log gradient magnitude for debugging purposes
            logging.info(f"Layer {layer_index}: Gradient magnitude = {grad_magnitude}")

            # Check for vanishing gradients (very small magnitude)
            if grad_magnitude < 1e-6:
                logging.warning(f"Vanishing gradient detected in layer {layer_index}. Gradient magnitude = {grad_magnitude}")

            # Check for exploding gradients (very large magnitude)
            elif grad_magnitude > 1e6:
                logging.warning(f"Exploding gradient detected in layer {layer_index}. Gradient magnitude = {grad_magnitude}")

    def display_reward_to_time_ratio(ratio):
        # Ensure the ratio list is not empty
        if not ratio:
            logging.warning("The reward-to-time ratio list is empty.")
            return

        # Plot the reward-to-time ratios
        plt.figure(figsize=(8, 6))
        plt.bar(range(len(ratio)), ratio, color='skyblue')

        # Set plot title and labels
        plt.title('Reward-to-Time Ratio for Architectural Actions')
        plt.xlabel('Action Index')
        plt.ylabel('Reward-to-Time Ratio')

        # Display grid for better readability
        plt.grid(True)

        # Show the plot
        plt.show()

        # Log the display action
        if DEBUG_MODE:
            logging.info("Displayed reward-to-time ratio plot.")

    def visualize_architecture(network_structure):
        # Initialize an empty list to store layer descriptions
        layers = []

        # Traverse through each layer in the network
        for layer_index, layer in enumerate(network_structure.children()):
            layer_type = layer.__class__.__name__

            # Check if the layer is fully connected
            if isinstance(layer, nn.Linear):
                description = f"Layer {layer_index}: {layer_type} (Input: {layer.in_features}, Output: {layer.out_features})"

            # Check if the layer is a convolutional layer
            elif isinstance(layer, nn.Conv2d):
                description = f"Layer {layer_index}: {layer_type} (In Channels: {layer.in_channels}, Out Channels: {layer.out_channels}, Kernel: {layer.kernel_size})"

            # Check for recurrent layers like LSTM and GRU
            elif isinstance(layer, (nn.LSTM, nn.GRU)):
                description = f"Layer {layer_index}: {layer_type} (Input Size: {layer.input_size}, Hidden Size: {layer.hidden_size})"

            # Check for other types of layers
            else:
                description = f"Layer {layer_index}: {layer_type}"

            layers.append(description)

        # Plot the architecture visualization using matplotlib
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(layers)), [1]*len(layers), tick_label=layers, color='lightgreen')

        # Set plot title and labels
        plt.title('Neural Network Architecture Visualization')
        plt.xlabel('Layer Order')

        # Show the plot
        plt.tight_layout()
        plt.show()

        # Log the architecture visualization
        if DEBUG_MODE:
            logging.info("Visualized network architecture.")

    def print_debugging_info(epoch, loss, reward, architecture_changes):
        # Print epoch information
        print(f"Epoch: {epoch}")

        # Print loss and reward information
        print(f"Loss: {loss:.4f}, Reward: {reward:.4f}")

        # Print architecture changes
        if architecture_changes:
            print("Architecture Changes Applied:")
            for change in architecture_changes:
                print(f" - {change}")
        else:
            print("No architecture changes applied.")

        # Log the debugging information
        if DEBUG_MODE:
            logging.info(f"Epoch {epoch}: Loss = {loss}, Reward = {reward}")
            if architecture_changes:
                logging.info(f"Architecture changes in Epoch {epoch}: {architecture_changes}")
            else:
                logging.info(f"No architecture changes in Epoch {epoch}.")

    def train_network(network, dataloader, criterion, optimizer):
        # Initialize variables to track the reward and previous reward
        previous_reward = 0.0
        architecture_changes = []

        # Loop through each epoch
        for epoch in range(num_epochs):
            network.train()
            total_loss = 0.0

            for batch, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                # Zero the gradients for each batch
                optimizer.zero_grad()

                # Forward pass
                outputs = network(inputs)
                loss = criterion(outputs, targets)

                # Backward pass and optimization step
                loss.backward()
                optimizer.step()

                # Accumulate total loss
                total_loss += loss.item()

            # Average loss over the batches
            avg_loss = total_loss / len(dataloader)

            # Compute rewards for architectural adaptation
            if epoch % adaptation_frequency == 0:
                current_reward = compute_gradient_reward(previous_reward, avg_loss)

                # Apply regularizations (temporal smoothing, sparsity, etc.)
                smoothed_reward = apply_temporal_smoothing(current_reward, previous_reward, alpha)
                regularized_reward = apply_sparsity_regularization(smoothed_reward, network.output_layer.out_features, gamma_sparsity)
                previous_reward = regularized_reward

                # If reward exceeds threshold, trigger architecture changes
                if regularized_reward > reward_threshold:
                    architecture_changes.append(evaluate_architecture_change(network, layer_index=0, dataloader=dataloader, criterion=criterion, optimizer=optimizer))

                    # Apply the best architectural change
                    for change in architecture_changes:
                        apply_architecture_change(network, change, layer_index=0)

            # Print debugging information
            print_debugging_info(epoch, avg_loss, regularized_reward, architecture_changes)

            # Optionally, display plots for architecture evolution or loss curves
            if DISPLAY_PLOTS:
                visualize_architecture(network)

        # Return the trained model
        return network

    def evaluate_network(network, validation_loader, criterion):
        network.eval()  # Set the network to evaluation mode
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():  # Disable gradient calculation for evaluation
            for batch, (inputs, targets) in enumerate(validation_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                outputs = network(inputs)
                loss = criterion(outputs, targets)

                # Accumulate total loss
                total_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == targets).sum().item()
                total_samples += targets.size(0)

        # Calculate average loss and accuracy
        avg_loss = total_loss / len(validation_loader)
        accuracy = correct_predictions / total_samples

        # Log the results if debugging is enabled
        if DEBUG_MODE:
            logging.info(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        return avg_loss, accuracy

    def apply_sgnn_optimization(network, dataloader, criterion, optimizer, epochs):
        for epoch in range(epochs):
            network.train()  # Set the network to training mode
            total_loss = 0.0

            for batch, (inputs, targets) in enumerate(dataloader):
                inputs, targets = inputs.to(device), targets.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = network(inputs)
                loss = criterion(outputs, targets)

                # Backward pass
                loss.backward()

                # Apply sparse gradient pruning (SGNN)
                for name, param in network.named_parameters():
                    if param.grad is not None:
                        # Sparse gradient optimization: Set gradients below a threshold to zero
                        sparse_grad_threshold = 1e-5  # Example threshold value, can be tuned
                        param.grad = torch.where(torch.abs(param.grad) < sparse_grad_threshold, torch.zeros_like(param.grad), param.grad)

                        # Log the sparse gradient optimization if debugging is enabled
                        if DEBUG_MODE:
                            logging.info(f"Sparse Gradient Optimization applied to layer {name}")

                # Update weights
                optimizer.step()

                # Accumulate total loss
                total_loss += loss.item()

            # Calculate average loss for the epoch
            avg_loss = total_loss / len(dataloader)

            # Optionally log epoch information
            if DEBUG_MODE:
                logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        return network

if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    # Define hyperparameters
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 100

    # Define transformations for the dataset
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the network, criterion, and optimizer
    network = SimpleNN(input_size, output_size, initial_hidden_units).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

    # Initialize variables to track training and validation accuracy
    training_accuracy = []
    validation_accuracy = []
    architecture_changes = []  # Store any architecture changes made

    # Training loop
    for epoch in range(num_epochs):
        network.train()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Use tqdm to wrap the train_loader for a progress bar
        for batch, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

        avg_loss = total_loss / len(train_loader)
        train_accuracy = correct_predictions / total_samples
        training_accuracy.append(train_accuracy)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.4f}')

        # Evaluate on validation dataset using the instance method
        val_metrics = network.evaluate_task_performance("classification", {
            'dataloader': DataLoader(datasets.MNIST(root='./data', train=False, transform=transform), batch_size=batch_size, shuffle=False),
            'criterion': criterion
        })
        val_accuracy = val_metrics['accuracy']
        validation_accuracy.append(val_accuracy)

        print(f'Validation Accuracy: {val_accuracy:.4f}')

        # Track architecture changes if any
        if architecture_changes:
            print(f'Architecture Changes in Epoch {epoch + 1}: {architecture_changes}')


    # Evaluate the model on the test dataset
    performance_metrics = {
        'dataloader': test_loader,
        'criterion': criterion
    }
    test_results = network.evaluate_task_performance("classification", performance_metrics)
    test_accuracy = test_results['accuracy']
    
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
"""
CURRENT ISSUES:
    + .train() appears to make the model fail to use developed methods.
"""
