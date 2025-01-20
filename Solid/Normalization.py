import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load data
interpolated_data = pd.read_csv(r"C:\Users\aryan\OneDrive\Desktop\Reservoir Classification\sinc_interpolated_data_visual_2.csv")

# Extract seismic and well log attributes
seismic_interpolated_data = interpolated_data[['Interpolated Impedance', 'Interpolated Amplitude', 'Interpolated Frequency']]
wellog_interpolated_data = interpolated_data[['Sand Fraction', 'Water Saturation']]

# Normalize seismic data using StandardScaler
scaler = StandardScaler()
seismic_data_normalized = scaler.fit_transform(seismic_interpolated_data)

# Normalize well log attributes using MinMaxScaler
scaler_minmax = MinMaxScaler(feature_range=(0.0, 1.0))
well_log_data_normalized = scaler_minmax.fit_transform(wellog_interpolated_data)

# Convert to DataFrame with column names
well_df = pd.DataFrame(well_log_data_normalized, columns=['Sand Fraction', 'Water Saturation'])
seismic_df = pd.DataFrame(seismic_data_normalized, columns=['Interpolated Impedance', 'Interpolated Amplitude', 'Interpolated Frequency'])

# Include depth in the well log DataFrame
df1 = interpolated_data[['Depth (m)']]
well_df = pd.concat([df1, well_df], axis=1)

# Include depth in the seismic DataFrame
df1 = interpolated_data[['Depth (m)']]
seismic_df = pd.concat([df1, seismic_df], axis=1)

# Save the normalized data to CSV files
well_df.to_csv(r"C:\Users\aryan\OneDrive\Desktop\Reservoir Classification\well_normalised.csv", index=False)
seismic_df.to_csv(r"C:\Users\aryan\OneDrive\Desktop\Reservoir Classification\seismic_normalised.csv", index=False)

combined_df = pd.merge(well_df, seismic_df, on='Depth (m)', how='inner')
combined_df.to_csv("normalized_all_data.csv", index=False)