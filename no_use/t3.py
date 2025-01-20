import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import resample, firwin, lfilter

class SignalReconstructor:
    def __init__(self, velocity_profile):
        """
        Initialize the SignalReconstructor with a velocity profile.
        :param velocity_profile: A callable that converts depth to time (e.g., a linear or interpolated function).
        """
        self.velocity_profile = velocity_profile

    def depth_to_time_conversion(self, well_log_data, depth_points):
        """
        Convert well log data from the depth domain to the time domain using the velocity profile.
        :param well_log_data: The well log data in the depth domain.
        :param depth_points: Corresponding depth points for the well log data.
        :return: Well log data in the time domain and time points.
        """
        time_points = self.velocity_profile(depth_points)
        return time_points, well_log_data

    def adjust_sampling_interval(self, seismic_data, seismic_time, target_time_points):
        """
        Upsample seismic data to match the finer sampling rate of well log data.
        :param seismic_data: The seismic data.
        :param seismic_time: Time points for the seismic data.
        :param target_time_points: Target time points for upsampling.
        :return: Upsampled seismic data.
        """
        interpolator = interp1d(seismic_time, seismic_data, kind='linear', fill_value="extrapolate")
        upsampled_data = interpolator(target_time_points)
        return upsampled_data

    def reconstruct_band_limited_signal(self, signal_data, cutoff_frequency, sampling_rate):
        """
        Reconstruct a band-limited signal using a low-pass filter based on the Nyquist theorem.
        :param signal_data: The signal data to be reconstructed.
        :param cutoff_frequency: The cutoff frequency for the low-pass filter.
        :param sampling_rate: The sampling rate of the signal.
        :return: Reconstructed band-limited signal.
        """
        nyquist_rate = sampling_rate / 2
        normalized_cutoff = cutoff_frequency / nyquist_rate
        filter_coeffs = firwin(101, normalized_cutoff)
        reconstructed_signal = lfilter(filter_coeffs, [1.0], signal_data)
        return reconstructed_signal

    def integrate_data(self, seismic_data, well_log_data):
        """
        Combine seismic and well log data for further analysis.
        :param seismic_data: The reconstructed seismic data.
        :param well_log_data: The well log data.
        :return: Integrated data for analysis.
        """
        return np.column_stack((seismic_data, well_log_data))

# --- Example Usage ---
if __name__ == "__main__":
    # Example velocity profile (depth to time conversion)
    def velocity_profile(depth):
        return depth / 2000  # Simple linear conversion (depth in meters to time in seconds)

    # Synthetic well log and seismic data
    np.random.seed(42)
    depth_points = np.linspace(1000, 2000, 500)  # Depth in meters
    well_log_data = np.sin(2 * np.pi * 0.01 * depth_points) + 0.1 * np.random.normal(size=500)

    seismic_time = np.linspace(0, 1, 100)  # Coarse time points (seconds)
    seismic_data = np.sin(2 * np.pi * 10 * seismic_time) + 0.2 * np.random.normal(size=100)

    # Target time points for well log data (finer sampling rate)
    target_time_points = np.linspace(0, 1, 500)

    # Initialize reconstructor
    reconstructor = SignalReconstructor(velocity_profile)

    # Depth-to-time conversion
    time_points, well_log_time_data = reconstructor.depth_to_time_conversion(well_log_data, depth_points)

    # Upsample seismic data
    upsampled_seismic = reconstructor.adjust_sampling_interval(seismic_data, seismic_time, target_time_points)

    # Reconstruct band-limited seismic signal
    reconstructed_seismic = reconstructor.reconstruct_band_limited_signal(upsampled_seismic, cutoff_frequency=20, sampling_rate=500)

    # Integrate seismic and well log data
    integrated_data = reconstructor.integrate_data(reconstructed_seismic, well_log_time_data)

    # Print results for verification
    print("Integrated Data Sample:", integrated_data[:5])
