import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load realistic seismic and well log data
predictor_data = pd.read_csv(r'C:\Users\aryan\OneDrive\Desktop\Reservoir Classification\seismic_data.csv')
target_data = pd.read_csv(r'C:\Users\aryan\OneDrive\Desktop\Reservoir Classification\well_logs.csv')

# Extract depths for seismic and well log data
seismic_depth = predictor_data['Depth (m)'].values
log_depth = target_data['Depth (m)'].values

# 1. Visualize Depth Comparison
plt.figure(figsize=(10, 6))
plt.plot(seismic_depth, np.zeros_like(seismic_depth), 'o', label="Seismic Depth", color='orange')
plt.plot(log_depth, np.ones_like(log_depth), 'x', label="Log Depth", color='blue')
plt.legend()
plt.xlabel("Depth (m)")
plt.title("Depth Comparison")
plt.show()

# Define the sinc interpolation function
def sinc_interp(x, s, u):
    """Interpolate the values at u using the samples at s."""
    if len(s) != len(x):
        raise ValueError("Input arrays should have the same length")
    return np.dot(x, np.sinc((u[:, None] - s) / (s[1] - s[0])))

# Upsample seismic attributes to well log depth using sinc interpolation
interpolated_impedance = sinc_interp(predictor_data['Impedance'].values, seismic_depth, log_depth)
interpolated_amplitude = sinc_interp(predictor_data['Seismic Amplitude'].values, seismic_depth, log_depth)
interpolated_frequency = sinc_interp(predictor_data['Instantaneous Frequency'].values, seismic_depth, log_depth)

# Combine interpolated seismic data with original well log data
interpolated_data = pd.DataFrame({
    'Depth (m)': log_depth,
    'Interpolated Impedance': interpolated_impedance,
    'Interpolated Amplitude': interpolated_amplitude,
    'Interpolated Frequency': interpolated_frequency,
    'Sand Fraction': target_data['Sand Fraction'].values,
    'Water Saturation': target_data['Water Saturation'].values,
    
})

# Save interpolated data to a new CSV file
interpolated_data.to_csv('sinc_interpolated_data_visual_2.csv', index=False)

# 2. Correlation Check
corr_impedance = np.corrcoef(interpolated_impedance, target_data['Sand Fraction'])[0, 1]
corr_amplitude = np.corrcoef(interpolated_amplitude, target_data['Sand Fraction'])[0, 1]
corr_frequency = np.corrcoef(interpolated_frequency, target_data['Sand Fraction'])[0, 1]

print(f"Correlation with Sand Fraction:\n - Impedance: {corr_impedance:.4f}\n - Amplitude: {corr_amplitude:.4f}\n - Frequency: {corr_frequency:.4f}")

# 3. Error Metrics Calculation at Common Points
# Find the nearest indices in log_depth for original seismic depth points
nearest_indices = np.searchsorted(log_depth, seismic_depth)

# Remove indices that are out of bounds (e.g., if seismic depth points exceed log depth range)
valid_indices = nearest_indices[nearest_indices < len(log_depth)]

# Calculate MAE and RMSE using the valid overlapping points
mae_impedance = mean_absolute_error(predictor_data['Impedance'].values, interpolated_impedance[valid_indices])
mae_amplitude = mean_absolute_error(predictor_data['Seismic Amplitude'].values, interpolated_amplitude[valid_indices])
mae_frequency = mean_absolute_error(predictor_data['Instantaneous Frequency'].values, interpolated_frequency[valid_indices])

rmse_impedance = np.sqrt(mean_squared_error(predictor_data['Impedance'].values, interpolated_impedance[valid_indices]))
rmse_amplitude = np.sqrt(mean_squared_error(predictor_data['Seismic Amplitude'].values, interpolated_amplitude[valid_indices]))
rmse_frequency = np.sqrt(mean_squared_error(predictor_data['Instantaneous Frequency'].values, interpolated_frequency[valid_indices]))

print(f"Mean Absolute Error (MAE):\n - Impedance: {mae_impedance:.4f}\n - Amplitude: {mae_amplitude:.4f}\n - Frequency: {mae_frequency:.4f}")
print(f"Root Mean Square Error (RMSE):\n - Impedance: {rmse_impedance:.4f}\n - Amplitude: {rmse_amplitude:.4f}\n - Frequency: {rmse_frequency:.4f}")

# 4. Distribution Comparison
original_means = predictor_data[['Impedance', 'Seismic Amplitude', 'Instantaneous Frequency']].mean()
interpolated_means = [interpolated_impedance.mean(), interpolated_amplitude.mean(), interpolated_frequency.mean()]

original_stds = predictor_data[['Impedance', 'Seismic Amplitude', 'Instantaneous Frequency']].std()
interpolated_stds = [interpolated_impedance.std(), interpolated_amplitude.std(), interpolated_frequency.std()]

print("\nOriginal Means:")
print(original_means)
print("\nOriginal Stds:")
print(original_stds)
print("\nInterpolated Means and Standard Deviations:")
print(f"Impedance Mean: {interpolated_means[0]:.4f}, Std: {interpolated_stds[0]:.4f}")
print(f"Amplitude Mean: {interpolated_means[1]:.4f}, Std: {interpolated_stds[1]:.4f}")
print(f"Frequency Mean: {interpolated_means[2]:.4f}, Std: {interpolated_stds[2]:.4f}")

# 5. Small Tolerance for Floating-Point Comparison (for RMSE/MSE validation)
tolerance = 1e-10  # Adjust this based on the scale of your data

diff_impedance = np.abs(predictor_data['Impedance'].values - interpolated_impedance[valid_indices])
if np.all(diff_impedance < tolerance):
    print("Impedance values are considered equal within tolerance")

# 6. Visual Inspection: Plot original vs interpolated data for alignment check
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(log_depth, interpolated_impedance, label='Interpolated Impedance', color='brown')
plt.plot(seismic_depth, predictor_data['Impedance'], 'o', markersize=2, color='orange', alpha=0.6, label='Original Impedance')
plt.xlabel("Depth (m)")
plt.ylabel("Impedance")
plt.legend()
plt.title("Impedance - Original vs Interpolated")

plt.subplot(3, 1, 2)
plt.plot(log_depth, interpolated_amplitude, label='Interpolated Amplitude', color='purple')
plt.plot(seismic_depth, predictor_data['Seismic Amplitude'], 'o', markersize=2, color='pink', alpha=0.6, label='Original Amplitude')
plt.xlabel("Depth (m)")
plt.ylabel("Amplitude")
plt.legend()
plt.title("Amplitude - Original vs Interpolated")

plt.subplot(3, 1, 3)
plt.plot(log_depth, interpolated_frequency, label='Interpolated Frequency', color='blue')
plt.plot(seismic_depth, predictor_data['Instantaneous Frequency'], 'o', markersize=2, color='cyan', alpha=0.6, label='Original Frequency')
plt.xlabel("Depth (m)")
plt.ylabel("Frequency")
plt.legend()
plt.title("Frequency - Original vs Interpolated")

plt.tight_layout()
plt.show()