import math
import random
import requests
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import json
from config import API_EMAIL, API_TOKEN

class FlightPathCalculator:
    """
    A class to handle the calculation and optimization of flight paths considering Earth's curvature and wind vector fields.
    """

    @staticmethod
    def validate_coordinates(lat: float, lon: float):
        """
        Validates the latitude and longitude coordinates.

        :param lat: Latitude in degrees.
        :param lon: Longitude in degrees.
        :raises ValueError: If the coordinates are out of range.
        """
        if not (-90 <= lat <= 90):
            raise ValueError("Latitude value must be in the range of -90 to 90 degrees.")
        if not (-180 <= lon <= 180):
            raise ValueError("Longitude value must be in the range of -180 to 180 degrees.")

    @staticmethod
    def radians_from_degrees(lat1: float, lon1: float, lat2: float, lon2: float):
        """
        Converts latitude and longitude from degrees to radians for two points.

        :param lat1: Latitude of the first point in degrees.
        :param lon1: Longitude of the first point in degrees.
        :param lat2: Latitude of the second point in degrees.
        :param lon2: Longitude of the second point in degrees.
        :return: Tuple containing lat1, lon1, lat2, lon2 in radians.
        """
        return math.radians(lat1), math.radians(lon1), math.radians(lat2), math.radians(lon2)

    @staticmethod
    def haversine(lat1_rad: float, lon1_rad: float, lat2_rad: float, lon2_rad: float) -> float:
        """
        Applies the Haversine formula to calculate the great-circle distance.

        :param lat1_rad: Latitude of the first point in radians.
        :param lon1_rad: Longitude of the first point in radians.
        :param lat2_rad: Latitude of the second point in radians.
        :param lon2_rad: Longitude of the second point in radians.
        :return: Great-circle distance in kilometers.
        """
        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad

        a = math.sin(delta_lat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        radius_of_earth_km = 6371.0
        return radius_of_earth_km * c

    def calculate_initial_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculates the initial bearing between two geographic points on Earth.

        :param lat1: Latitude of the initial coordinate in degrees.
        :param lon1: Longitude of the initial coordinate in degrees.
        :param lat2: Latitude of the target coordinate in degrees.
        :param lon2: Longitude of the target coordinate in degrees.
        :return: Initial bearing in degrees.
        """
        self.validate_coordinates(lat1, lon1)
        self.validate_coordinates(lat2, lon2)

        try:
            lat1_rad, lon1_rad, lat2_rad, lon2_rad = self.radians_from_degrees(lat1, lon1, lat2, lon2)
            delta_lon = lon2_rad - lon1_rad

            x = math.sin(delta_lon) * math.cos(lat2_rad)
            y = math.cos(lat1_rad) * math.sin(lat2_rad) - (math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))

            if x == 0 and y == 0:
                raise ZeroDivisionError("Division by zero encountered in the calculation of initial bearing.")

            initial_bearing_rad = math.atan2(x, y)
            initial_bearing_deg = (math.degrees(initial_bearing_rad) + 360) % 360

            return initial_bearing_deg

        except ZeroDivisionError as e:
            print("Error in calculate_initial_bearing: Division by zero detected.")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred in calculate_initial_bearing: {e}")
            raise e

    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculates the great-circle distance between two points on Earth using the Haversine formula.

        :param lat1: Latitude of the initial coordinate in degrees.
        :param lon1: Longitude of the initial coordinate in degrees.
        :param lat2: Latitude of the target coordinate in degrees.
        :param lon2: Longitude of the target coordinate in degrees.
        :return: Distance between the points in kilometers.
        """
        self.validate_coordinates(lat1, lon1)
        self.validate_coordinates(lat2, lon2)

        try:
            lat1_rad, lon1_rad, lat2_rad, lon2_rad = self.radians_from_degrees(lat1, lon1, lat2, lon2)
            return self.haversine(lat1_rad, lon1_rad, lat2_rad, lon2_rad)

        except ZeroDivisionError as e:
            print("Error in calculate_distance: Division by zero detected.")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred in calculate_distance: {e}")
            raise e

    def adjust_for_wind(self, lat: float, lon: float, bearing: float, wind_vector: tuple) -> float:
        """
        Adjusts the flight path to account for wind vector fields at a given location.

        :param lat: Latitude of the current coordinate in degrees.
        :param lon: Longitude of the current coordinate in degrees.
        :param bearing: Current bearing in degrees.
        :param wind_vector: Wind vector as a tuple (speed in m/s, direction in degrees).
        :return: Adjusted bearing in degrees considering wind effects.
        """
        self.validate_coordinates(lat, lon)
        if not (0 <= bearing < 360):
            raise ValueError("Bearing must be in the range of 0 to 360 degrees.")
        if len(wind_vector) != 2:
            raise ValueError("Wind vector must be a tuple of two elements: (speed in m/s, direction in degrees).")

        wind_speed, wind_direction = wind_vector
        if not isinstance(wind_speed, (int, float)) or not isinstance(wind_direction, (int, float)):
            raise ValueError("Wind vector components must be numerical values.")
        if not (0 <= wind_direction < 360):
            raise ValueError("Wind direction must be in the range of 0 to 360 degrees.")
        if wind_speed < 0:
            raise ValueError("Wind speed cannot be negative.")

        try:
            bearing_rad = math.radians(bearing)
            wind_direction_rad = math.radians(wind_direction)

            wind_effect = wind_speed * math.sin(wind_direction_rad - bearing_rad)
            adjusted_bearing_rad = bearing_rad + wind_effect / 1000

            adjusted_bearing_deg = (math.degrees(adjusted_bearing_rad) + 360) % 360
            return adjusted_bearing_deg

        except ZeroDivisionError as e:
            print("Error in adjust_for_wind: Division by zero detected.")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred in adjust_for_wind: {e}")
            raise e

    def calculate_next_point(self, lat: float, lon: float, bearing: float, distance: float) -> tuple:
        """
        Calculates the next point (lat, lon) given the current position, bearing, and distance traveled.

        :param lat: Latitude of the current coordinate in degrees.
        :param lon: Longitude of the current coordinate in degrees.
        :param bearing: Current bearing in degrees.
        :param distance: Distance to travel from the current point in kilometers.
        :return: A tuple containing the next coordinates (latitude, longitude) in degrees.
        """
        self.validate_coordinates(lat, lon)
        if not (0 <= bearing < 360):
            raise ValueError("Bearing must be in the range of 0 to 360 degrees.")
        if distance < 0:
            raise ValueError("Distance cannot be negative.")

        try:
            lat_rad, lon_rad, bearing_rad = math.radians(lat), math.radians(lon), math.radians(bearing)
            radius_of_earth_km = 6371.0

            new_lat_rad = math.asin(math.sin(lat_rad) * math.cos(distance / radius_of_earth_km) +
                                    math.cos(lat_rad) * math.sin(distance / radius_of_earth_km) * math.cos(bearing_rad))

            new_lon_rad = lon_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance / radius_of_earth_km) * math.cos(lat_rad),
                                               math.cos(distance / radius_of_earth_km) - math.sin(lat_rad) * math.sin(new_lat_rad))

            new_lat = math.degrees(new_lat_rad)
            new_lon = math.degrees(new_lon_rad)

            new_lon = (new_lon + 180) % 360 - 180
            return new_lat, new_lon

        except ZeroDivisionError as e:
            print("Error in calculate_next_point: Division by zero detected.")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred in calculate_next_point: {e}")
            raise e

    def get_wind_vector(self, lat: float, lon: float) -> tuple:
        """
        Retrieves the wind vector (speed and direction) at a specific latitude and longitude.

        :param lat: Latitude of the location in degrees.
        :param lon: Longitude of the location in degrees.
        :return: Wind vector as a tuple (speed in m/s, direction in degrees).
        """
        self.validate_coordinates(lat, lon)

        try:
            # First, retrieve the metadata for the given latitude and longitude
            api_endpoint = f"https://api.weather.gov/points/{lat},{lon}"
            headers = {
                "User-Agent": "Your App Name",
                "Authorization": f"Bearer {API_TOKEN}",
                "From": API_EMAIL
            }
            response = requests.get(api_endpoint, headers=headers)
            response.raise_for_status()

            metadata = response.json()
            forecast_grid_url = metadata['properties']['forecastGridData']

            # Now retrieve the grid data
            grid_response = requests.get(forecast_grid_url, headers=headers)
            grid_response.raise_for_status()
            grid_data = grid_response.json()

            # Extract wind speed and direction from the grid data
            wind_speed = grid_data['properties']['windSpeed']['values'][0]['value']
            wind_direction = grid_data['properties']['windDirection']['values'][0]['value']

            # Print wind speed and direction to terminal
            print(f"Wind speed: {wind_speed} m/s")
            print(f"Wind direction: {wind_direction} degrees")

            return wind_speed, wind_direction

        except requests.exceptions.RequestException as e:
            print(f"Network error occurred while fetching wind data: {e}")
            raise RuntimeError("Unable to retrieve wind vector data due to a network issue.")
        except KeyError as e:
            print(f"Unexpected data format received: {e}")
            raise RuntimeError("Unexpected data format received from the wind data source.")
        except Exception as e:
            print(f"An unexpected error occurred in get_wind_vector: {e}")
            raise e

    @staticmethod
    def _convert_wind_speed_to_mps(wind_speed_str: str) -> float:
        """
        Converts wind speed from a string format (e.g., '10 mph') to meters per second.

        :param wind_speed_str: Wind speed as a string.
        :return: Wind speed in meters per second.
        """
        try:
            speed_mph = float(wind_speed_str.split()[0])
            return speed_mph * 0.44704
        except (ValueError, IndexError) as e:
            print(f"Error converting wind speed: {e}")
            raise ValueError("Invalid wind speed format encountered.")

    @staticmethod
    def _convert_wind_direction_to_degrees(wind_direction_str: str) -> float:
        """
        Converts wind direction from a string format (e.g., 'NE' for Northeast) to degrees.

        :param wind_direction_str: Wind direction as a string.
        :return: Wind direction in degrees.
        """
        directions = {
            "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
            "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
            "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
            "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5
        }
        try:
            return directions[wind_direction_str.upper()]
        except KeyError as e:
            print(f"Error converting wind direction: {e}")
            raise ValueError("Invalid wind direction format encountered.")

    def calculate_flight_path(self, lat1: float, lon1: float, lat2: float, lon2: float, sampling_rate: int) -> list:
        """
        Computes the flight path as a list of lat/long coordinates from the initial to the target coordinates,
        considering Earth's curvature and wind fields.

        :param lat1: Latitude of the initial coordinate in degrees.
        :param lon1: Longitude of the initial coordinate in degrees.
        :param lat2: Latitude of the target coordinate in degrees.
        :param lon2: Longitude of the target coordinate in degrees.
        :param sampling_rate: Number of points to sample along the path.
        :return: List of tuples containing lat/long coordinates representing the flight path.
        """
        self.validate_coordinates(lat1, lon1)
        self.validate_coordinates(lat2, lon2)
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be a positive integer.")

        try:
            initial_bearing = self.calculate_initial_bearing(lat1, lon1, lat2, lon2)
            total_distance = self.calculate_distance(lat1, lon1, lat2, lon2)
            step_distance = total_distance / sampling_rate

            flight_path = [(lat1, lon1)]
            current_lat, current_lon = lat1, lon1
            current_bearing = initial_bearing

            for i in range(1, sampling_rate):
                wind_speed, wind_direction = self.get_wind_vector(current_lat, current_lon)
                adjusted_bearing = self.adjust_for_wind(current_lat, current_lon, current_bearing, (wind_speed, wind_direction))
                next_lat, next_lon = self.calculate_next_point(current_lat, current_lon, adjusted_bearing, step_distance)

                flight_path.append((next_lat, next_lon))
                current_lat, current_lon = next_lat, next_lon
                current_bearing = adjusted_bearing

            flight_path.append((lat2, lon2))
            return flight_path

        except ZeroDivisionError as e:
            print("Error in calculate_flight_path: Division by zero detected.")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred in calculate_flight_path: {e}")
            raise e

    def optimize_flight_path(self, lat1: float, lon1: float, lat2: float, lon2: float, sampling_rate: int) -> list:
        """
        Optimizes the flight path to find the most efficient route considering distance and wind effects.

        :param lat1: Latitude of the initial coordinate in degrees.
        :param lon1: Longitude of the initial coordinate in degrees.
        :param lat2: Latitude of the target coordinate in degrees.
        :param lon2: Longitude of the target coordinate in degrees.
        :param sampling_rate: Number of points to sample along the path.
        :return: Optimized list of tuples containing lat/long coordinates.
        """
        self.validate_coordinates(lat1, lon1)
        self.validate_coordinates(lat2, lon2)
        if sampling_rate <= 0:
            raise ValueError("Sampling rate must be a positive integer.")

        try:
            best_path = None
            best_cost = float('inf')
            initial_flight_path = self.calculate_flight_path(lat1, lon1, lat2, lon2, sampling_rate)
            optimization_iterations = 100

            for _ in range(optimization_iterations):
                perturbed_path = self._perturb_path(initial_flight_path)
                total_cost = self._calculate_path_cost(perturbed_path)

                if total_cost < best_cost:
                    best_path = perturbed_path
                    best_cost = total_cost

            if best_path is None:
                best_path = initial_flight_path

            return best_path

        except ZeroDivisionError as e:
            print("Error in optimize_flight_path: Division by zero detected.")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred in optimize_flight_path: {e}")
            raise e

    def _perturb_path(self, flight_path: list) -> list:
        """
        Slightly perturbs a given flight path to explore different routes.

        :param flight_path: List of tuples containing lat/long coordinates.
        :return: A perturbed list of tuples containing lat/long coordinates.
        """
        perturbed_path = []
        for lat, lon in flight_path:
            delta_lat = random.uniform(-0.01, 0.01)
            delta_lon = random.uniform(-0.01, 0.01)
            new_lat = min(max(lat + delta_lat, -90), 90)
            new_lon = min(max(lon + delta_lon, -180), 180)

            perturbed_path.append((new_lat, new_lon))
        return perturbed_path

    def _calculate_path_cost(self, flight_path: list) -> float:
        """
        Calculates the total cost of a flight path considering distance and wind effects.

        :param flight_path: List of tuples containing lat/long coordinates.
        :return: Total cost as a float.
        """
        total_cost = 0.0
        for i in range(1, len(flight_path)):
            lat1, lon1 = flight_path[i - 1]
            lat2, lon2 = flight_path[i]

            distance = self.calculate_distance(lat1, lon1, lat2, lon2)
            wind_speed, wind_direction = self.get_wind_vector(lat1, lon1)
            bearing = self.calculate_initial_bearing(lat1, lon1, lat2, lon2)
            adjusted_bearing = self.adjust_for_wind(lat1, lon1, bearing, (wind_speed, wind_direction))

            wind_effect = abs(adjusted_bearing - bearing)
            total_cost += distance + wind_effect

        return total_cost

    def plot_flight_path(self, flight_path: list):
        """
        Plots the computed flight path on a map for visualization.

        :param flight_path: List of tuples containing lat/long coordinates representing the flight path.
        """
        if not flight_path:
            raise ValueError("Flight path is empty. Please provide a valid list of coordinates.")

        try:
            lats, lons = zip(*flight_path)
            self.validate_coordinates(min(lats), min(lons))
            self.validate_coordinates(max(lats), max(lons))

            plt.figure(figsize=(10, 7))
            m = Basemap(projection='merc', llcrnrlat=min(lats) - 10, urcrnrlat=max(lats) + 10,
                        llcrnrlon=min(lons) - 10, urcrnrlon=max(lons) + 10, resolution='i')

            m.drawcoastlines()
            m.drawcountries()
            x, y = m(lons, lats)
            m.plot(x, y, marker='o', color='r', markersize=5, linewidth=2, label='Flight Path')

            m.drawmapboundary(fill_color='aqua')
            m.fillcontinents(color='coral', lake_color='aqua')
            m.drawparallels(range(-90, 90, 10), labels=[1, 0, 0, 0])
            m.drawmeridians(range(-180, 180, 10), labels=[0, 0, 0, 1])

            plt.title('Flight Path Visualization')
            plt.legend()
            plt.show()

        except ZeroDivisionError as e:
            print("Error in plot_flight_path: Division by zero detected.")
            raise e
        except ValueError as e:
            print(f"Value error in plot_flight_path: {e}")
            raise e
        except Exception as e:
            print(f"An unexpected error occurred in plot_flight_path: {e}")
            raise e
