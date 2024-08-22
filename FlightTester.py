# FlightTester.py

from FlightPathCalculator import FlightPathCalculator
import json
import numpy as np
import matplotlib.pyplot as plt

def load_city_coordinates(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)
    
def inject_noise(coordinates, variance, std_dev):
    # Ensure the variance is within the valid range
    # if not 0 < variance < 1:
        # raise ValueError("Variance must be between 0 and 1")
    
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

def compare_noise(coordinates, std_dev, variances):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    axs = axs.flatten()
    
    for i, variance in enumerate(variances):
        # Inject noise into the flight path
        noisy_path = inject_noise(coordinates, variance, std_dev)
        
        # Plot the noisy flight path
        axs[i].set_title(f'Variance = {variance}')
        axs[i].plot([coord[1] for coord in noisy_path], [coord[0] for coord in noisy_path], marker='o')
        axs[i].set_xlabel('Longitude')
        axs[i].set_ylabel('Latitude')
        
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
    sampling_rate = 8

    # Calculate the optimized flight path
    try:
        optimized_path = calculator.optimize_flight_path(lat1, lon1, lat2, lon2, sampling_rate)
        std_dev = 1

        # Define variance levels to test (logarithmic scale)
        variances = [1, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005] 

        # Compare the effect of different noise levels
        compare_noise(optimized_path, std_dev, variances)

    except Exception as e:
        print(f"An error occurred during the flight path calculation: {e}")

if __name__ == "__main__":
    main()
