import numpy as np
import scipy.signal as signal
import pywt
from scipy.fft import fft, ifft
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class DataPreprocessor:
    def __init__(self):
        self.scalers = {}

    def z_score_normalization(self, data, variable_name):
        """
        Applies Z-score normalization to the data.
        """
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        self.scalers[variable_name] = scaler
        return normalized_data

    def min_max_normalization(self, data, variable_name, feature_range=(0.1, 0.9)):
        """
        Applies Min-Max normalization to the data.
        """
        scaler = MinMaxScaler(feature_range=feature_range)
        normalized_data = scaler.fit_transform(data.reshape(-1, 1)).flatten()
        self.scalers[variable_name] = scaler
        return normalized_data

    def fourier_regularization(self, target_signal, predictor_signal, cutoff_ratio=0.8):
        """
        Applies Fourier Transform (FT) based regularization.
        """
        target_fft = fft(target_signal)
        predictor_fft = fft(predictor_signal)
        cutoff_index = int(cutoff_ratio * len(predictor_fft) // 2)
        regularized_fft = np.zeros_like(target_fft)
        regularized_fft[:cutoff_index] = target_fft[:cutoff_index]
        regularized_fft[-cutoff_index:] = target_fft[-cutoff_index:]
        return ifft(regularized_fft).real

    def wavelet_regularization(self, signal_data, wavelet='db4', level=6):
        """
        Applies Wavelet Decomposition (WD) based regularization.
        """
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]  # Zero out detail coefficients
        return pywt.waverec(coeffs, wavelet)

    def emd_regularization(self, target_signal, max_imfs_to_retain=3):
        """
        Applies Empirical Mode Decomposition (EMD) based regularization.
        """
        # Placeholder implementation since PyEMD is unavailable
        print("EMD regularization is not available as PyEMD is not installed.")
        return target_signal  # Return the original signal as a fallback

# --- Example Usage ---
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    time = np.linspace(0, 1, 500)
    seismic_data = np.sin(2 * np.pi * 10 * time) + 0.5 * np.random.normal(size=500)
    sand_fraction = np.sin(2 * np.pi * 20 * time) + np.random.normal(size=500)

    # Create preprocessor object
    preprocessor = DataPreprocessor()

    # Z-score normalization
    seismic_normalized = preprocessor.z_score_normalization(seismic_data, "seismic")
    sand_normalized = preprocessor.min_max_normalization(sand_fraction, "sand_fraction")

    # Fourier regularization
    sand_fourier_regularized = preprocessor.fourier_regularization(sand_normalized, seismic_normalized)

    # Wavelet regularization
    sand_wavelet_regularized = preprocessor.wavelet_regularization(sand_normalized)

    # EMD regularization (fallback behavior)
    sand_emd_regularized = preprocessor.emd_regularization(sand_normalized)

    # Print results for verification
    print("Fourier Regularized Signal:", sand_fourier_regularized[:10])
    print("Wavelet Regularized Signal:", sand_wavelet_regularized[:10])
    print("EMD Regularized Signal (fallback):", sand_emd_regularized[:10])
