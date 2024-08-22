# FlightTester.py

from FlightPathCalculator import FlightPathCalculator
import json

def load_city_coordinates(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)
    
def main():
    # Initialize the FlightPathCalculator
    calculator = FlightPathCalculator()

    city_coordinates = load_city_coordinates('city_coordinates.json')

    # Define the initial and target coordinates (latitude and longitude in degrees)
    lat1, lon1 = city_coordinates['Seattle']
    lat2, lon2 =  city_coordinates['New York']

    # Define the sampling rate (number of points along the path)
    sampling_rate = 160

    # Calculate the optimized flight path
    try:
        optimized_path = calculator.optimize_flight_path(lat1, lon1, lat2, lon2, sampling_rate)

        # Plot the optimized flight path
        calculator.plot_flight_path(optimized_path)
    except Exception as e:
        print(f"An error occurred during the flight path calculation: {e}")

if __name__ == "__main__":
    main()
