import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# Load and preprocess data
def load_data(signal_path, target_path):
    # Load datasets
    try:
        # Try reading with header first
        signals = pd.read_csv(signal_path)
        # If first column contains non-numeric data, try reading without header
        if not np.issubdtype(signals.iloc[:, 0].dtype, np.number):
            signals = pd.read_csv(signal_path, header=None)
            print("Read signal data without header")
        
        targets = pd.read_csv(target_path)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
    
    # Verify data shapes
    if len(signals) != len(targets):
        print(f"Warning: Mismatch in sample sizes - signals: {len(signals)}, targets: {len(targets)}")
        # Align datasets by taking the minimum length
        min_length = min(len(signals), len(targets))
        signals = signals.iloc[:min_length]
        targets = targets.iloc[:min_length]
    
    # Convert to numpy arrays
    X = signals.values
    # Ensure input has at least 3 time steps for Conv1D
    if X.shape[1] < 3:
        # Pad sequences with zeros if too short
        padding_size = 3 - X.shape[1]
        X = np.pad(X, ((0, 0), (0, padding_size)), mode='constant')
    
    X = X.reshape((X.shape[0], X.shape[1], 1))
    y = targets['Sand Fraction'].values
    
    print(f"Processed input shape: {X.shape}")
    
    print(f"Final dataset shapes - X: {X.shape}, y: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    return X_train, X_test, y_train, y_test

# Build CNN-LSTM model
def build_model(input_shape):
    model = Sequential()
    
    # CNN layers with adjusted kernel size
    kernel_size = min(3, input_shape[0])  # Ensure kernel size <= sequence length
    model.add(Conv1D(filters=32, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    if input_shape[0] > 1:  # Only add pooling if sequence length > 1
        model.add(MaxPooling1D(pool_size=1))  # Use pool_size=1 to maintain sequence length
    model.add(Dropout(0.2))
    
    # LSTM layer
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    
    # Dense layers
    model.add(Dense(25, activation='relu'))
    model.add(Dense(1))  # Output layer for regression
    
    # Compile model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model

def main():
    # File paths
    signal_path = 'combined_signal.csv'
    target_path = 'regularized_emd_target_signals.csv'
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_data(signal_path, target_path)
    
    # Build model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    
    # Early stopping callback
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f'\nTest MAE: {mae:.4f}')
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot actual vs predicted with regression line
    plt.subplot(1, 2, 2)
    
    # Calculate regression line
    coef = np.polyfit(y_test.flatten(), y_pred.flatten(), 1)
    poly1d_fn = np.poly1d(coef)
    
    # Plot points and regression line
    plt.scatter(y_test, y_pred, alpha=0.5, label='Data Points')
    plt.plot(y_test, poly1d_fn(y_test), 'r-', label='Regression Line')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', label='Perfect Prediction')
    
    # Add plot details
    plt.title('Actual vs Predicted Sand Fraction')
    plt.xlabel('Actual Sand Fraction')
    plt.ylabel('Predicted Sand Fraction')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.show()
    
    # Save model
    model.save('sand_fraction_model.h5')
    
    # Plot model architecture
    plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)

if __name__ == '__main__':
    main()
