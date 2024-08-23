from FlightPathCalculator import FlightPathCalculator
from KalmanFilter import KalmanFilter  
import json
import numpy as np
import matplotlib.pyplot as plt

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

def apply_kalman_filter(noisy_path, std_dev):
    # Initialize Kalman Filter parameters
    initial_state = np.array([noisy_path[0][0], noisy_path[0][1]])  # Start with the first noisy coordinate
    initial_covariance = np.eye(2) * std_dev**2
    transition_matrix = np.eye(2)
    observation_matrix = np.eye(2)
    process_noise = np.eye(2) * std_dev**2
    observation_noise = np.eye(2) * std_dev**2

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

def compare_noise_and_denoise(coordinates, std_dev, variances):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    
    original_latitudes = np.array([coord[0] for coord in coordinates])
    original_longitudes = np.array([coord[1] for coord in coordinates])
    
    for i, variance in enumerate(variances):
        # Inject noise into the flight path
        noisy_path = inject_noise(coordinates, variance, std_dev)
        
        # Apply Kalman filter to denoise the path
        denoised_path = apply_kalman_filter(noisy_path, std_dev)
        
        # Convert denoised path back to list of tuples
        denoised_path = [(state[0], state[1]) for state in denoised_path]
        
        # Calculate empirical differences
        noisy_latitudes = np.array([coord[0] for coord in noisy_path])
        noisy_longitudes = np.array([coord[1] for coord in noisy_path])
        
        denoised_latitudes = np.array([coord[0] for coord in denoised_path])
        denoised_longitudes = np.array([coord[1] for coord in denoised_path])
        
        original_noise_diff = np.sqrt((original_latitudes - noisy_latitudes)**2 + (original_longitudes - noisy_longitudes)**2)
        original_denoised_diff = np.sqrt((original_latitudes - denoised_latitudes)**2 + (original_longitudes - denoised_longitudes)**2)
        noisy_denoised_diff = np.sqrt((noisy_latitudes - denoised_latitudes)**2 + (noisy_longitudes - denoised_longitudes)**2)
        
        print(f"Variance {variance}:")
        print(f"  Mean difference between original and noisy signal: {np.mean(original_noise_diff)}")
        print(f"  Mean difference between original and Kalman-denoised signal: {np.mean(original_denoised_diff)}")
        print(f"  Mean difference between noisy and Kalman-denoised signal: {np.mean(noisy_denoised_diff)}")
        
        # Plot the original, noisy, and denoised flight path
        axs[i].set_title(f'Variance = {variance}')
        
        # Original path (in green)
        axs[i].plot(original_longitudes, original_latitudes, 
                    marker='o', color='green', label='Original')
        
        # Noisy path (in blue)
        axs[i].plot(noisy_longitudes, noisy_latitudes, 
                    marker='o', color='blue', label='Noisy')
        
        # Denoised path (in red)
        axs[i].plot(denoised_longitudes, denoised_latitudes, 
                    marker='o', color='red', label='Denoised')
        
        axs[i].set_xlabel('Longitude')
        axs[i].set_ylabel('Latitude')
        axs[i].legend()
        
    plt.tight_layout()
    plt.show()


def main():
    # Initialize the FlightPathCalculator
    calculator = FlightPathCalculator()

    city_coordinates = load_city_coordinates('city_coordinates.json')

    # Define the initial and target coordinates (latitude and longitude in degrees)
    lat1, lon1 = city_coordinates['Seattle']
    lat2, lon2 = city_coordinates['London']

    # Define the sampling rate (number of points along the path)
    sampling_rate = 16

    # Calculate the optimized flight path
    try:
        optimized_path = calculator.optimize_flight_path(lat1, lon1, lat2, lon2, sampling_rate)
        std_dev = 1

        # Define variance levels to test (logarithmic scale)
        variances = [10, 5, 1, 0.5, 0.1, 0.05, 0.01, 0.005] 

        # Compare the effect of different noise levels and apply Kalman filter
        compare_noise_and_denoise(optimized_path, std_dev, variances)

    except Exception as e:
        print(f"An error occurred during the flight path calculation: {e}")

if __name__ == "__main__":
    main()
