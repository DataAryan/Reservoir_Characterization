import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.fftpack import fft, ifft
from pywt import wavedec, waverec
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Step 1: Generate Synthetic Seismic and Lithological Data
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    time = np.linspace(0, 10, num_samples)
    seismic_impedance = np.sin(2 * np.pi * 0.5 * time) + np.random.normal(0, 0.1, num_samples)
    seismic_amplitude = np.cos(2 * np.pi * 0.5 * time) + np.random.normal(0, 0.1, num_samples)
    instantaneous_frequency = np.abs(np.sin(2 * np.pi * 0.25 * time)) + np.random.normal(0, 0.05, num_samples)

    # Target lithological property (Sand Fraction)
    sand_fraction = 0.5 * seismic_impedance + 0.3 * seismic_amplitude + 0.2 * instantaneous_frequency
    sand_fraction += np.random.normal(0, 0.05, num_samples)  # Add noise

    return time, seismic_impedance, seismic_amplitude, instantaneous_frequency, sand_fraction

# Step 2: Add Noise
def add_noise(data, noise_level=0.1):
    return data + np.random.normal(0, noise_level, len(data))

# Step 3: Normalize Data
def normalize_data(data, method="zscore"):
    if method == "zscore":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler((0.1, 0.9))
    else:
        raise ValueError("Unknown normalization method")
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()

# Step 4: Regularization using Fourier Transform
def regularize_data_fourier(data, cutoff_freq=0.2):
    fft_coeffs = fft(data)
    freqs = np.fft.fftfreq(len(data))
    fft_coeffs[np.abs(freqs) > cutoff_freq] = 0
    return np.real(ifft(fft_coeffs))

# Step 5: Regularization using Wavelet Decomposition
def regularize_data_wavelet(data, wavelet='db4', level=6):
    coeffs = wavedec(data, wavelet, level=level)
    coeffs[-1] = np.zeros_like(coeffs[-1])  # Remove high-frequency noise
    return waverec(coeffs, wavelet)

# Main Workflow
time, imp, amp, freq, sand = generate_synthetic_data()

# Add noise
imp_noisy = add_noise(imp, 0.1)
amp_noisy = add_noise(amp, 0.1)
freq_noisy = add_noise(freq, 0.05)
sand_noisy = add_noise(sand, 0.05)

# Normalize Data
imp_norm = normalize_data(imp_noisy, "zscore")
amp_norm = normalize_data(amp_noisy, "zscore")
freq_norm = normalize_data(freq_noisy, "zscore")
sand_norm = normalize_data(sand_noisy, "minmax")

# Regularize Data
sand_reg_fourier = regularize_data_fourier(sand_norm, cutoff_freq=0.2)
sand_reg_wavelet = regularize_data_wavelet(sand_norm)

# Plot Results
plt.figure(figsize=(10, 8))
plt.subplot(3, 1, 1)
plt.plot(time, sand_norm, label='Normalized Sand Fraction')
plt.plot(time, sand_reg_fourier, label='Fourier Regularized')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, sand_norm, label='Normalized Sand Fraction')
plt.plot(time, sand_reg_wavelet, label='Wavelet Regularized')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(time, imp_norm, label='Normalized Impedance')
plt.plot(time, amp_norm, label='Normalized Amplitude')
plt.plot(time, freq_norm, label='Normalized Frequency')
plt.legend()

plt.tight_layout()
plt.show()
