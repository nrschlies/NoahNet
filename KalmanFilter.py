import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, initial_state: np.ndarray, initial_covariance: np.ndarray, 
                 transition_matrix: np.ndarray, observation_matrix: np.ndarray, 
                 process_noise: np.ndarray, observation_noise: np.ndarray):
        self._validate_matrices(initial_state, initial_covariance, transition_matrix, 
                                observation_matrix, process_noise, observation_noise)

        self._initial_state = initial_state.copy()
        self._initial_covariance = initial_covariance.copy()
        self._initial_transition_matrix = transition_matrix.copy()
        self._initial_observation_matrix = observation_matrix.copy()
        self._initial_process_noise = process_noise.copy()
        self._initial_observation_noise = observation_noise.copy()

        self.state = initial_state.copy().reshape(-1, 1)  # Ensure state is a column vector
        self.covariance = initial_covariance.copy()
        self.transition_matrix = transition_matrix.copy()
        self.observation_matrix = observation_matrix.copy()
        self.process_noise = process_noise.copy()
        self.observation_noise = observation_noise.copy()

    def _validate_matrices(self, initial_state, initial_covariance, transition_matrix, 
                           observation_matrix, process_noise, observation_noise):
        if not isinstance(initial_state, np.ndarray) or initial_state.ndim != 1:
            raise ValueError("initial_state must be a 1D numpy array.")
        if not isinstance(initial_covariance, np.ndarray) or initial_covariance.shape[0] != initial_covariance.shape[1]:
            raise ValueError("initial_covariance must be a square numpy array.")
        if not isinstance(transition_matrix, np.ndarray) or transition_matrix.shape[0] != transition_matrix.shape[1]:
            raise ValueError("transition_matrix must be a square numpy array.")
        if not isinstance(observation_matrix, np.ndarray) or observation_matrix.shape[0] <= 0 or observation_matrix.shape[1] <= 0:
            raise ValueError("observation_matrix must be a 2D numpy array with positive dimensions.")
        if not isinstance(process_noise, np.ndarray) or process_noise.shape[0] != process_noise.shape[1]:
            raise ValueError("process_noise must be a square numpy array.")
        if not isinstance(observation_noise, np.ndarray) or observation_noise.shape[0] != observation_noise.shape[1]:
            raise ValueError("observation_noise must be a square numpy array.")
        
        self._check_for_zeros_in_diagonal(initial_covariance, "initial_covariance")
        self._check_for_zeros_in_diagonal(process_noise, "process_noise")
        self._check_for_zeros_in_diagonal(observation_noise, "observation_noise")
        
        state_dim = initial_state.shape[0]
        if (initial_covariance.shape[0] != state_dim or 
            transition_matrix.shape[0] != state_dim or 
            transition_matrix.shape[1] != state_dim or 
            process_noise.shape[0] != state_dim or 
            process_noise.shape[1] != state_dim):
            raise ValueError("Mismatch in matrix dimensions related to state dimension.")

        obs_dim = observation_matrix.shape[0]
        if observation_matrix.shape[1] != state_dim:
            raise ValueError("Mismatch between observation matrix dimensions and state dimension.")
        if observation_noise.shape[0] != obs_dim or observation_noise.shape[1] != obs_dim:
            raise ValueError("Mismatch in observation_noise dimensions related to observation dimension.")

    def _check_for_zeros_in_diagonal(self, matrix: np.ndarray, matrix_name: str):
        if np.any(np.diag(matrix) == 0):
            raise ValueError(f"{matrix_name} matrix contains zeros on the diagonal, which may lead to division by zero.")

    def _validate_dimensions(self, matrix_a: np.ndarray, matrix_b: np.ndarray, operation: str):
        if matrix_a.shape != matrix_b.shape:
            raise ValueError(f"Dimension mismatch during {operation}: {matrix_a.shape} vs {matrix_b.shape}")

    def _safe_matrix_multiply(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
        try:
            return np.dot(matrix_a, matrix_b)
        except Exception as e:
            raise RuntimeError(f"Matrix multiplication error: {str(e)}")

    def _safe_inverse_multiply(self, matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
        try:
            return np.dot(matrix_a, np.linalg.inv(matrix_b))
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Matrix inversion error: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during matrix inversion: {str(e)}")

    def predict(self):
        self.state = self._safe_matrix_multiply(self.transition_matrix, self.state)
        self.covariance = self._safe_matrix_multiply(self.transition_matrix, self.covariance)
        self.covariance = self._safe_matrix_multiply(self.covariance, self.transition_matrix.T) + self.process_noise
        self._check_for_zeros_in_diagonal(self.covariance, "Predicted covariance")

    def update(self, measurement: np.ndarray):
        measurement = measurement.reshape(-1, 1)  # Ensure measurement is a column vector
        S = self._safe_matrix_multiply(self.observation_matrix, self.covariance)
        S = self._safe_matrix_multiply(S, self.observation_matrix.T) + self.observation_noise
        
        K = self._safe_matrix_multiply(self.covariance, self.observation_matrix.T)
        K = self._safe_inverse_multiply(K, S)
        
        y = measurement - self._safe_matrix_multiply(self.observation_matrix, self.state)
        self.state = self.state + self._safe_matrix_multiply(K, y)

        I = np.eye(self.covariance.shape[0])
        self.covariance = self._safe_matrix_multiply(I - self._safe_matrix_multiply(K, self.observation_matrix), self.covariance)
        self._check_for_zeros_in_diagonal(self.covariance, "Updated covariance")

    def filter(self, measurements: list[np.ndarray]) -> list[np.ndarray]:
        filtered_states = []
        for measurement in measurements:
            self.predict()
            self.update(measurement)
            filtered_states.append(self.state.copy())
        return filtered_states

    def reset(self):
        self.state = self._initial_state.copy().reshape(-1, 1)
        self.covariance = self._initial_covariance.copy()
        self.transition_matrix = self._initial_transition_matrix.copy()
        self.observation_matrix = self._initial_observation_matrix.copy()
        self.process_noise = self._initial_process_noise.copy()
        self.observation_noise = self._initial_observation_noise.copy()

    def set_transition_matrix(self, transition_matrix: np.ndarray):
        self._check_for_zeros_in_diagonal(transition_matrix, "Transition")
        self.transition_matrix = transition_matrix.copy()

    def set_observation_matrix(self, observation_matrix: np.ndarray):
        if observation_matrix.shape[1] != self.state.shape[0]:
            raise ValueError("Observation matrix dimensions must match the state dimension.")
        self.observation_matrix = observation_matrix.copy()

    def set_process_noise(self, process_noise: np.ndarray):
        self._check_for_zeros_in_diagonal(process_noise, "Process noise")
        self.process_noise = process_noise.copy()

    def set_observation_noise(self, observation_noise: np.ndarray):
        self._check_for_zeros_in_diagonal(observation_noise, "Observation noise")
        self.observation_noise = observation_noise.copy()

    def get_state_estimate(self) -> np.ndarray:
        if not np.isfinite(self.state).all():
            raise ValueError("State estimate contains invalid values (NaN or infinity).")
        return self.state.flatten()

    def get_covariance(self) -> np.ndarray:
        if not np.isfinite(self.covariance).all():
            raise ValueError("Covariance matrix contains invalid values (NaN or infinity).")
        return self.covariance

    def _convert_coordinates_to_state_vector(self, coordinates: list[tuple]) -> np.ndarray:
        if not isinstance(coordinates, list) or len(coordinates) == 0:
            raise ValueError("Coordinates must be a non-empty list.")
        
        if not all(isinstance(coord, tuple) and len(coord) == 2 for coord in coordinates):
            raise ValueError("Each coordinate must be a tuple of (latitude, longitude).")
        
        state_vector = np.array(coordinates).flatten()

        if not np.isfinite(state_vector).all():
            raise ValueError("State vector contains invalid values (NaN or infinity).")

        return state_vector.reshape(-1, 1)

    def _convert_state_vector_to_coordinates(self, state_vector: np.ndarray) -> list[tuple]:
        if state_vector.ndim != 2 or state_vector.shape[1] != 1:
            raise ValueError("State vector must be a 2D numpy array with one column.")
        
        if not np.isfinite(state_vector).all():
            raise ValueError("State vector contains invalid values (NaN or infinity).")

        coordinates = [(state_vector[i, 0], state_vector[i+1, 0]) for i in range(0, len(state_vector), 2)]
        return coordinates

    def plot_results(self, original_coordinates: list[tuple], filtered_coordinates: list[tuple]):
        if len(original_coordinates) != len(filtered_coordinates):
            raise ValueError("Original and filtered coordinates lists must be of the same length.")
        
        original_latitudes, original_longitudes = zip(*original_coordinates)
        filtered_latitudes, filtered_longitudes = zip(*filtered_coordinates)
        
        plt.figure(figsize=(10, 6))
        plt.plot(original_longitudes, original_latitudes, 'bo-', label='Original Coordinates')
        plt.plot(filtered_longitudes, filtered_latitudes, 'ro-', label='Filtered Coordinates')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Original vs Filtered Coordinates')
        plt.legend()
        plt.grid(True)
        plt.show()
