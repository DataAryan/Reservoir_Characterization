import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_absolute_percentage_error

# Load normalized predictor data
predictor_df = pd.read_csv(r'C:\Users\aryan\OneDrive\Desktop\Reservoir Classification\seismic_normalised.csv')
X = predictor_df[['Interpolated Impedance', 'Interpolated Amplitude', 'Interpolated Frequency']].values

# Load normalized target data
target_df = pd.read_csv(r'C:\Users\aryan\OneDrive\Desktop\Reservoir Classification\well_normalised.csv')
y = target_df['Sand Fraction'].values

# Initialize the linear regression model
model = LinearRegression()

# Define cross-validation configuration
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform cross-validation using MAPE
cross_val_scores = cross_val_score(
    model, X, y, cv=kf, scoring=make_scorer(mean_absolute_percentage_error)
)

# Print cross-validation MAPE scores
print(f"Cross-validation MAPE scores: {cross_val_scores}")
avg_mape = np.mean(cross_val_scores)
print(f"Average MAPE across folds: {avg_mape}")

# Train the model on the entire dataset
model.fit(X, y)

# Get the learned weights (coefficients) from the model
weights = model.coef_
print(f"Learned weights (coefficients): {weights}")

# Predict using the trained model
predicted_signal = model.predict(X)

# Plot normalized predictions
plt.plot(np.arange(len(y)), predicted_signal, label='Predicted Signal', color='red')
plt.title("Predicted Signal Using Model Weights (Normalized Data)")
plt.xlabel("Time Step")
plt.ylabel("Normalized Amplitude")
plt.show()

# Save the normalized predicted signal to CSV
predicted_signal_df = pd.DataFrame(predicted_signal, columns=['Predicted Signal'])
predicted_signal_df.to_csv(r'C:\Users\aryan\OneDrive\Desktop\Reservoir Classification\predicted_signal.csv', index=False)

# Save the weights and predictor variable names to CSV
weights_df = pd.DataFrame({
    'Variable': ['Interpolated Impedance', 'Interpolated Amplitude', 'Interpolated Frequency'],
    'Weight': weights
})
weights_df.to_csv(r'C:\Users\aryan\OneDrive\Desktop\Reservoir Classification\weights_and_variables.csv', index=False)

# Visualizing normalized performance
plt.plot(np.arange(len(y)), y, label='Target Signal (Sand Fraction)', color='blue')
plt.plot(np.arange(len(y)), predicted_signal, label='Predicted Signal', linestyle='--', color='red')
plt.legend()
plt.xlabel('Time Step')
plt.ylabel('Normalized Signal Value')
plt.title('Target Signal vs Predicted Signal (Normalized Data)')
plt.show()
