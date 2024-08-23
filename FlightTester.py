from FlightPathCalculator import FlightPathCalculator
from KalmanFilter import KalmanFilter  
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel


def t_test(orig_noise_diffs, orig_kalman_diffs):
    # Ensure the arrays have equal length
    min_length = min(len(orig_noise_diffs), len(orig_kalman_diffs))
    orig_noise_diffs = orig_noise_diffs[:min_length]
    orig_kalman_diffs = orig_kalman_diffs[:min_length]
    #print("Performing t-test")
    #print(f"Original_noise_length {(orig_noise_diffs)}")
    #print(f"Original_kalman_length {(orig_kalman_diffs)}")
    # Perform a paired t-test
    t_stat, p_value = ttest_rel(orig_noise_diffs, orig_kalman_diffs)
    return p_value


def load_city_coordinates(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def inject_noise(coordinates, variance, std_dev):
    # Inject noise into each coordinate
    noisy_coordinates = []
    for lat, lon in coordinates:
        # Generate noise
        noise_lat = np.random.normal(0, std_dev * variance)
        noise_lon = np.random.normal(0, std_dev * variance)
        
        # Apply noise to the original coordinates
        noisy_lat = lat + noise_lat
        noisy_lon = lon + noise_lon
        
        # Create a new tuple with the noisy coordinates and add to the list
        noisy_coordinates.append((noisy_lat, noisy_lon))
    
    return noisy_coordinates

def optimize_kalman(noisy_path, original_path, std_dev, process_gain_range, observation_gain_range, finer_search=False):
    min_distance = float('inf')
    optimal_process_gain = None
    optimal_observation_gain = None

    expected_length = len(original_path)

    # Perform a grid search over the process and observation gain ranges
    for process_gain in process_gain_range:
        for observation_gain in observation_gain_range:
            # Apply the Kalman filter with the current gains
            denoised_path = apply_kalman_filter(noisy_path, std_dev, process_gain, observation_gain)
            
            # Ensure the length of denoised_path matches the expected length
            if len(denoised_path) != expected_length:
                raise ValueError(f"Expected {expected_length} denoised coordinates, but got {len(denoised_path)}.")

            # Calculate Orig-Kalman and Orig-Noise errors
            orig_kalman_diffs = []
            orig_noise_diffs = []
            for j in range(len(original_path)):
                lat_diff_orig_kalman = abs(original_path[j][0] - denoised_path[j][0])
                lon_diff_orig_kalman = abs(original_path[j][1] - denoised_path[j][1])
                mean_diff_orig_kalman = (lat_diff_orig_kalman + lon_diff_orig_kalman) / 2
                orig_kalman_diffs.append(mean_diff_orig_kalman.item())  # Convert to scalar

                lat_diff_orig_noise = abs(original_path[j][0] - noisy_path[j][0])
                lon_diff_orig_noise = abs(original_path[j][1] - noisy_path[j][1])
                mean_diff_orig_noise = (lat_diff_orig_noise + lon_diff_orig_noise) / 2
                orig_noise_diffs.append(mean_diff_orig_noise)

            # Calculate the mean of the differences
            mean_diff_orig_kalman = np.mean(orig_kalman_diffs)
            # mean_diff_orig_noise = np.mean(orig_noise_diffs)

            # Update the optimal gains if a new minimum distance is found
            if mean_diff_orig_kalman < min_distance:
                min_distance = mean_diff_orig_kalman
                optimal_process_gain = process_gain
                optimal_observation_gain = observation_gain

    return optimal_process_gain, optimal_observation_gain, min_distance


def apply_kalman_filter(noisy_path, std_dev, process_gain, observation_gain):
    # Initialize Kalman Filter parameters
    initial_state = np.array([noisy_path[0][0], noisy_path[0][1]])  # Start with the first noisy coordinate
    initial_covariance = np.eye(2) * std_dev**2  # Initial covariance

    # Transition matrix - Identity matrix assuming constant velocity model
    transition_matrix = np.eye(2)

    # Observation matrix - Direct observation of position
    observation_matrix = np.eye(2)

    # Process noise: This is adjusted to allow the filter to be more responsive
    process_noise = np.eye(2) * std_dev**2 * process_gain  # Increase process noise to make the filter more responsive

    # Observation noise: Reducing this will increase the trust in the noisy measurements
    observation_noise = np.eye(2) * std_dev**2 / observation_gain  # Decrease observation noise to trust the noisy measurements more

    # Initialize the Kalman Filter
    kalman_filter = KalmanFilter(
        initial_state, initial_covariance, 
        transition_matrix, observation_matrix, 
        process_noise, observation_noise
    )
    
    # Convert noisy path to numpy array
    noisy_measurements = [np.array([lat, lon]) for lat, lon in noisy_path]
    
    # Apply the Kalman filter to the noisy path
    denoised_path = kalman_filter.filter(noisy_measurements)
    
    return denoised_path

def compare_noise_and_denoise(coordinates, std_dev, variances, sampling_rate):
    fig, axs = plt.subplots(2, 4, figsize=(15, 5))
    axs = axs.flatten()

    expected_length = sampling_rate + 1

    # Ensure the length of coordinates matches the expected length
    if len(coordinates) != expected_length:
        raise ValueError(f"Expected {expected_length} coordinates, but got {len(coordinates)}.")

    # Define process and observation gain ranges for high and low variances
    process_gain_range = np.linspace(-2000, 2000, 100)
    observation_gain_range = np.linspace(-2000, 2000, 100)
    
    optimal_process_gains = []
    optimal_observation_gains = []
    
    for i, variance in enumerate(variances):
        # Inject noise into the flight path
        noisy_path = inject_noise(coordinates, variance, std_dev)

        # Ensure the length of noisy_path matches the expected length
        if len(noisy_path) != expected_length:
            raise ValueError(f"Expected {expected_length} noisy coordinates, but got {len(noisy_path)}.")
        
        # Find optimal process and observation gains using the original path
        optimal_process_gain, optimal_observation_gain, min_distance = optimize_kalman(
            noisy_path, coordinates, std_dev, 
            process_gain_range, observation_gain_range
        )
                
        # Apply Kalman filter with the optimal gains
        denoised_path = apply_kalman_filter(noisy_path, std_dev, optimal_process_gain, optimal_observation_gain)
        
        # Ensure the length of denoised_path matches the expected length
        if len(denoised_path) != expected_length:
            raise ValueError(f"Expected {expected_length} denoised coordinates, but got {len(denoised_path)}.")

        # Convert denoised path back to list of tuples
        denoised_path = [(state[0], state[1]) for state in denoised_path]

        # Calculate Orig-Kalman and Orig-Noise errors
        orig_kalman_diffs = []
        orig_noise_diffs = []
        for j in range(len(coordinates)):
            lat_diff_orig_kalman = abs(coordinates[j][0] - denoised_path[j][0])
            lon_diff_orig_kalman = abs(coordinates[j][1] - denoised_path[j][1])
            mean_diff_orig_kalman = (lat_diff_orig_kalman + lon_diff_orig_kalman) / 2
            orig_kalman_diffs.append(mean_diff_orig_kalman.item())  # Convert to scalar

            lat_diff_orig_noise = abs(coordinates[j][0] - noisy_path[j][0])
            lon_diff_orig_noise = abs(coordinates[j][1] - noisy_path[j][1])
            mean_diff_orig_noise = (lat_diff_orig_noise + lon_diff_orig_noise) / 2
            orig_noise_diffs.append(mean_diff_orig_noise)

        mean_diff_orig_kalman = np.mean(orig_kalman_diffs)
        mean_diff_orig_noise = np.mean(orig_noise_diffs)

        # Perform the t-test
        p_value = t_test(orig_noise_diffs, orig_kalman_diffs)
        
        # Plot the original, noisy, and denoised flight path on the current subplot
        axs[i].plot([coord[1] for coord in coordinates], [coord[0] for coord in coordinates], 
                    marker='o', color='green', label='Original')
        
        axs[i].plot([coord[1] for coord in noisy_path], [coord[0] for coord in noisy_path], 
                    marker='o', color='blue', label='Noisy')
        
        axs[i].plot([state[1] for state in denoised_path], [state[0] for state in denoised_path], 
                    marker='o', color='red', label='Denoised')
        
        axs[i].set_xlabel('Longitude')
        axs[i].set_ylabel('Latitude')
        axs[i].legend()
        
        # Add the optimal gains, Orig-Kalman, and Orig-Noise errors to the subplot title
        axs[i].set_title(f'Variance = {variance}\n'
                         f'Opt Process Gain: {optimal_process_gain:.4f}\n'
                         f'Opt Observation Gain: {optimal_observation_gain:.4f}\n'
                         f'Orig-Kalman Error: {mean_diff_orig_kalman:.4f}\n'
                         f'Orig-Noise Error: {mean_diff_orig_noise:.4f}\n'
                         f'p-value: {p_value:.3f}')

    plt.tight_layout()
    plt.show()
    
    # Plot the optimal process and observation gains against variances
    plt.figure()
    plt.plot(variances, optimal_process_gains, label='Optimal Process Gain', marker='o')
    plt.plot(variances, optimal_observation_gains, label='Optimal Observation Gain', marker='o')
    plt.xlabel('Variance')
    plt.ylabel('Gain')
    plt.title('Optimal Process and Observation Gains')
    plt.legend()
    plt.show()




def apply_kalman_filter(noisy_path, std_dev, process_gain, observation_gain):
    # Initialize Kalman Filter parameters
    initial_state = np.array([noisy_path[0][0], noisy_path[0][1]])  # Start with the first noisy coordinate
    initial_covariance = np.eye(2) * std_dev**2  # Initial covariance

    # Transition matrix - Identity matrix assuming constant velocity model
    transition_matrix = np.eye(2)

    # Observation matrix - Direct observation of position
    observation_matrix = np.eye(2)

    # Process noise: This is adjusted to allow the filter to be more responsive
    process_noise = np.eye(2) * std_dev**2 * process_gain  # Increase process noise to make the filter more responsive

    # Observation noise: Reducing this will increase the trust in the noisy measurements
    observation_noise = np.eye(2) * std_dev**2 * observation_gain  # Decrease observation noise to trust the noisy measurements more

    # Initialize the Kalman Filter
    kalman_filter = KalmanFilter(
        initial_state, initial_covariance, 
        transition_matrix, observation_matrix, 
        process_noise, observation_noise
    )
    
    # Convert noisy path to numpy array
    noisy_measurements = [np.array([lat, lon]) for lat, lon in noisy_path]
    
    # Apply the Kalman filter to the noisy path
    denoised_path = kalman_filter.filter(noisy_measurements)
    
    return denoised_path

def compare_noise_and_denoise(coordinates, std_dev, variances, sampling_rate):
    fig, axs = plt.subplots(2, 4, figsize=(15, 5))
    axs = axs.flatten()

    expected_length = sampling_rate + 1

    # Ensure the length of coordinates matches the expected length
    if len(coordinates) != expected_length:
        raise ValueError(f"Expected {expected_length} coordinates, but got {len(coordinates)}.")

    # Define process and observation gain ranges for high and low variances
    process_gain_range = np.linspace(-200, 200, 10)
    observation_gain_range = np.linspace(-200, 200, 10)
    
    optimal_process_gains = []
    optimal_observation_gains = []
    
    for i, variance in enumerate(variances):
        # Inject noise into the flight path
        noisy_path = inject_noise(coordinates, variance, std_dev)

        # Ensure the length of noisy_path matches the expected length
        if len(noisy_path) != expected_length:
            raise ValueError(f"Expected {expected_length} noisy coordinates, but got {len(noisy_path)}.")
        
        # Find optimal process and observation gains using the original path
        optimal_process_gain, optimal_observation_gain, min_distance = optimize_kalman(
            noisy_path, coordinates, std_dev, 
            process_gain_range, observation_gain_range
        )
                
        # Apply Kalman filter with the optimal gains
        denoised_path = apply_kalman_filter(noisy_path, std_dev, optimal_process_gain, optimal_observation_gain)
        
        # Ensure the length of denoised_path matches the expected length
        if len(denoised_path) != expected_length:
            raise ValueError(f"Expected {expected_length} denoised coordinates, but got {len(denoised_path)}.")

        # Convert denoised path back to list of tuples
        denoised_path = [(state[0], state[1]) for state in denoised_path]

        # Calculate Orig-Kalman and Orig-Noise errors
        orig_kalman_diffs = []
        orig_noise_diffs = []
        for j in range(len(coordinates)):
            lat_diff_orig_kalman = abs(coordinates[j][0] - denoised_path[j][0])
            lon_diff_orig_kalman = abs(coordinates[j][1] - denoised_path[j][1])
            mean_diff_orig_kalman = (lat_diff_orig_kalman + lon_diff_orig_kalman) / 2
            orig_kalman_diffs.append(mean_diff_orig_kalman.item())  # Convert to scalar

            lat_diff_orig_noise = abs(coordinates[j][0] - noisy_path[j][0])
            lon_diff_orig_noise = abs(coordinates[j][1] - noisy_path[j][1])
            mean_diff_orig_noise = (lat_diff_orig_noise + lon_diff_orig_noise) / 2
            orig_noise_diffs.append(mean_diff_orig_noise)

        mean_diff_orig_kalman = np.mean(orig_kalman_diffs)
        mean_diff_orig_noise = np.mean(orig_noise_diffs)

        # Perform the t-test
        p_value = t_test(orig_noise_diffs, orig_kalman_diffs)
        
        # Plot the original, noisy, and denoised flight path on the current subplot
        axs[i].plot([coord[1] for coord in coordinates], [coord[0] for coord in coordinates], 
                    marker='o', color='green', label='Original')
        
        axs[i].plot([coord[1] for coord in noisy_path], [coord[0] for coord in noisy_path], 
                    marker='o', color='blue', label='Noisy')
        
        axs[i].plot([state[1] for state in denoised_path], [state[0] for state in denoised_path], 
                    marker='o', color='red', label='Denoised')
        
        axs[i].set_xlabel('Longitude')
        axs[i].set_ylabel('Latitude')
        axs[i].legend()
        
        # Add the optimal gains, Orig-Kalman, and Orig-Noise errors to the subplot title
        axs[i].set_title(f'Variance = {variance}\n'
                         f'Opt Process Gain: {optimal_process_gain:.4f}\n'
                         f'Opt Observation Gain: {optimal_observation_gain:.4f}\n'
                         f'Orig-Kalman Error: {mean_diff_orig_kalman:.4f}\n'
                         f'Orig-Noise Error: {mean_diff_orig_noise:.4f}\n'
                         f'p-value: {p_value:.3f}')

    plt.tight_layout()
    plt.show()
    
    # Plot the optimal process and observation gains against variances
    plt.figure()
    plt.plot(variances, optimal_process_gains, label='Optimal Process Gain', marker='o')
    plt.plot(variances, optimal_observation_gains, label='Optimal Observation Gain', marker='o')
    plt.xlabel('Variance')
    plt.ylabel('Gain')
    plt.title('Optimal Process and Observation Gains')
    plt.legend()
    plt.show()




def main():
    # Initialize the FlightPathCalculator
    calculator = FlightPathCalculator()

    city_coordinates = load_city_coordinates('city_coordinates.json')

    # Define the initial and target coordinates (latitude and longitude in degrees)
    lat1, lon1 = city_coordinates['Seattle']
    lat2, lon2 = city_coordinates['Los Angeles']

    # Define the sampling rate (number of points along the path)
    sampling_rate = 64

    # Calculate the optimized flight path
    try:
        optimized_path = calculator.optimize_flight_path(lat1, lon1, lat2, lon2, sampling_rate)
        calculator.plot_flight_path(optimized_path)

        std_dev = 1

        # Check the length of the generated path
        path_length = len(optimized_path)
        # print(f"Optimized path length: {path_length}")

        # Ensure your expectations match the generated path length
        # if path_length != sampling_rate:
            # print(f"Warning: Path length {path_length} does not match expected sampling rate {sampling_rate}")

        # Define variance levels to test (logarithmic scale)
        variances = [0.2, 0.1, 0.05, 0.02, 0.01] 

        # Compare the effect of different noise levels and apply Kalman filter
        compare_noise_and_denoise(optimized_path, std_dev, variances, sampling_rate)

    except Exception as e:
        print(f"An error occurred during the flight path calculation: {e}")

if __name__ == "__main__":
    main()
