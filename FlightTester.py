# FlightTester.py

from FlightPathCalculator import FlightPathCalculator

def main():
    # Initialize the FlightPathCalculator
    calculator = FlightPathCalculator()

    # Define the initial and target coordinates (latitude and longitude in degrees)
    lat1, lon1 = 34.0522, -118.2437  # Los Angeles, CA
    lat2, lon2 = 40.7128, -74.0060   # New York, NY

    # Define the sampling rate (number of points along the path)
    sampling_rate = 50

    # Calculate the optimized flight path
    try:
        optimized_path = calculator.optimize_flight_path(lat1, lon1, lat2, lon2, sampling_rate)

        # Plot the optimized flight path
        calculator.plot_flight_path(optimized_path)
    except Exception as e:
        print(f"An error occurred during the flight path calculation: {e}")

if __name__ == "__main__":
    main()
