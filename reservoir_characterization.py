import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv1D, LSTM, Dense, Dropout, 
                                   BatchNormalization, Flatten, concatenate,
                                   Attention, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from feature_utils import SeismicFeatureExtractor

# Load predictor and target data
file_path = 'combined_signal.csv'
target_path = 'regularized_emd_target_signals.csv'

# Read data
data = pd.read_csv(file_path)
target = pd.read_csv(target_path)

# Feature extraction
X = data['Combined Signal'].values.reshape(-1, 1)
y = target['Sand Fraction'].values

from sklearn.impute import SimpleImputer

# Enhanced feature engineering with feature selection
feature_extractor = SeismicFeatureExtractor(n_features=30)

# Handle NaN values in features
imputer = SimpleImputer(strategy='mean')
X_features = feature_extractor.transform(X)
X_features = imputer.fit_transform(X_features)

# Combine raw signal and engineered features
X_combined = np.concatenate([X, X_features], axis=1)

# Remove any remaining NaN values
nan_mask = np.isnan(X_combined).any(axis=1)
X_combined = X_combined[~nan_mask]
y = y[~nan_mask]

# Remove outliers using IQR
q1 = np.percentile(X_combined, 25, axis=0)
q3 = np.percentile(X_combined, 75, axis=0)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
mask = ((X_combined >= lower_bound) & (X_combined <= upper_bound)).all(axis=1)
X_combined = X_combined[mask]
y = y[mask]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Data augmentation functions
def augment_data(X, y, noise_level=0.01, scaling_range=(0.9, 1.1)):
    augmented_X = []
    augmented_y = []
    
    for signal, target in zip(X, y):
        # Add Gaussian noise
        noisy_signal = signal + noise_level * np.random.normal(size=signal.shape)
        augmented_X.append(noisy_signal)
        augmented_y.append(target)
        
        # Apply random scaling
        scale_factor = np.random.uniform(scaling_range[0], scaling_range[1])
        scaled_signal = signal * scale_factor
        augmented_X.append(scaled_signal)
        augmented_y.append(target)
        
    return np.array(augmented_X), np.array(augmented_y)

# Apply data augmentation
X_train_aug, y_train_aug = augment_data(X_train, y_train)

# Normalize features
scaler = StandardScaler()
X_train_aug = scaler.fit_transform(X_train_aug)
X_test = scaler.transform(X_test)

# Expand dimensions for Conv1D
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

from tensorflow.keras.layers import MultiHeadAttention, Add, LayerNormalization

# Build enhanced hybrid CNN-LSTM model with multi-head attention and residual connections
def build_hybrid_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # CNN branch with residual connections
    conv1 = Conv1D(64, kernel_size=5, activation='relu', padding='same')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Dropout(0.2)(conv1)
    
    # First residual block
    conv2 = Conv1D(64, kernel_size=5, activation='relu', padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Dropout(0.3)(conv2)
    res1 = Add()([conv1, conv2])
    
    # Second residual block with increased filters
    conv3 = Conv1D(128, kernel_size=5, activation='relu', padding='same')(res1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Dropout(0.3)(conv3)
    
    # Match dimensions for second residual connection
    conv4 = Conv1D(128, kernel_size=5, activation='relu', padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Dropout(0.3)(conv4)
    res2 = Add()([conv3, conv4])
    
    # LSTM branch with residual connections
    lstm1 = LSTM(128, return_sequences=True)(inputs)
    lstm1 = BatchNormalization()(lstm1)
    lstm1 = Dropout(0.3)(lstm1)
    
    # Multi-head attention mechanism
    attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(lstm1, lstm1)
    attention_output = LayerNormalization()(attention_output + lstm1)  # Add residual connection
    
    # Combine CNN and LSTM branches
    combined = concatenate([res1, attention_output])
    pooled = GlobalAveragePooling1D()(combined)
    
    # Dense layers with residual connections
    dense1 = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(pooled)
    dense1 = Dropout(0.4)(dense1)
    
    dense2 = Dense(128, activation='relu')(dense1)
    dense2 = Dropout(0.3)(dense2)
    
    # Add residual connection
    res2 = Add()([dense1, dense2])
    
    output = Dense(1, activation='linear')(res2)
    
    model = Model(inputs, output)
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
    return model

model = build_hybrid_model((X_train.shape[1], X_train.shape[2]))

from tensorflow.keras.callbacks import LearningRateScheduler

# Learning rate warmup schedule
def lr_warmup(epoch, lr):
    if epoch < 10:
        return lr * (epoch + 1) / 10
    return lr

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
lr_scheduler = LearningRateScheduler(lr_warmup)

# Compile model with gradient clipping
optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])

# Expand dimensions for augmented data
X_train_aug = np.expand_dims(X_train_aug, axis=-1)

# Train the model with augmented data
history = model.fit(
    X_train_aug, y_train_aug,
    validation_split=0.1,
    epochs=150,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr, lr_scheduler],
    verbose=1
)

# Evaluate the model
loss, mae, mape = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest Loss: {loss:.4f}, Test MAE: {mae:.4f}, Test MAPE: {mape:.2f}%")

# Predict on test data
predictions = model.predict(X_test).flatten()

# Enhanced visualization
def plot_results(y_test, predictions):
    plt.figure(figsize=(18, 12))
    
    # Actual vs Predicted
    plt.subplot(2, 2, 1)
    plt.plot(y_test, 'g-', label='Actual Sand Fraction')
    plt.plot(predictions, 'r--', label='Predicted Sand Fraction')
    plt.title('Actual vs Predicted Sand Fraction')
    plt.xlabel('Sample Index')
    plt.ylabel('Sand Fraction')
    plt.legend()
    plt.grid(True)
    
    # Residuals plot
    residuals = y_test - predictions
    plt.subplot(2, 2, 2)
    plt.scatter(range(len(residuals)), residuals, color='blue', alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.title('Residuals (Actual - Predicted)')
    plt.xlabel('Sample Index')
    plt.ylabel('Residuals')
    plt.grid(True)
    
    # Training and validation loss
    plt.subplot(2, 2, 3)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Error distribution
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=30, color='purple', alpha=0.7)
    plt.title('Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_results(y_test, predictions)
