import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters for synthetic data
depth_range = np.linspace(0, 3000, 1000)  # Depth in meters
sampling_interval = 0.15  # Sampling interval for logs (milliseconds)

# Simulate Well Log Data
def generate_well_log(depth_range, noise_level=0.05):
    np.random.seed(42)
    gamma_ray = np.sin(depth_range / 300) + np.random.normal(0, noise_level, size=len(depth_range))
    resistivity = np.cos(depth_range / 400) + np.random.normal(0, noise_level, size=len(depth_range))
    density = 2.5 + 0.1 * np.sin(depth_range / 500) + np.random.normal(0, noise_level, size=len(depth_range))
    sand_fraction = np.abs(np.sin(depth_range / 200)) + np.random.normal(0, noise_level, size=len(depth_range))
    water_saturation = np.clip(1 - sand_fraction, 0, 1)
    return pd.DataFrame({
        'Depth': depth_range,
        'GammaRay': gamma_ray,
        'Resistivity': resistivity,
        'Density': density,
        'SandFraction': sand_fraction,
        'WaterSaturation': water_saturation
    })

# Simulate Seismic Attributes
def generate_seismic_data(depth_range, noise_level=0.05):
    np.random.seed(24)
    seismic_impedance = np.abs(np.cos(depth_range / 300)) + np.random.normal(0, noise_level, size=len(depth_range))
    instantaneous_frequency = np.sin(depth_range / 400) + np.random.normal(0, noise_level, size=len(depth_range))
    amplitude = np.abs(np.cos(depth_range / 500)) + np.random.normal(0, noise_level, size=len(depth_range))
    return pd.DataFrame({
        'Depth': depth_range,
        'SeismicImpedance': seismic_impedance,
        'InstantaneousFrequency': instantaneous_frequency,
        'Amplitude': amplitude
    })

# Generate Synthetic Data
well_logs = generate_well_log(depth_range)
seismic_data = generate_seismic_data(depth_range)

# Save data to CSV
well_logs.to_csv("well_logs.csv", index=False)
seismic_data.to_csv("seismic_data.csv", index=False)

# Visualization
def plot_data(well_logs, seismic_data):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(well_logs['Depth'], well_logs['SandFraction'], label='Sand Fraction', color='blue')
    axs[0].plot(well_logs['Depth'], well_logs['WaterSaturation'], label='Water Saturation', color='red')
    axs[0].set_title('Well Logs')
    axs[0].set_xlabel('Depth (m)')
    axs[0].set_ylabel('Values')
    axs[0].legend()

    axs[1].plot(seismic_data['Depth'], seismic_data['SeismicImpedance'], label='Seismic Impedance', color='green')
    axs[1].plot(seismic_data['Depth'], seismic_data['Amplitude'], label='Amplitude', color='orange')
    axs[1].set_title('Seismic Attributes')
    axs[1].set_xlabel('Depth (m)')
    axs[1].set_ylabel('Values')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

plot_data(well_logs, seismic_data)

print("Synthetic data generation complete. Saved to CSV files.")
