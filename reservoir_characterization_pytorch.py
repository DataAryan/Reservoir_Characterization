import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from feature_utils import SeismicFeatureExtractor

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load predictor and target data
file_path = 'combined_signal.csv'
target_path = 'regularized_emd_target_signals.csv'

# Read data
data = pd.read_csv(file_path)
target = pd.read_csv(target_path)

# Feature extraction
X = data['Combined Signal'].values.reshape(-1, 1)
y = target['Sand Fraction'].values

# Enhanced feature engineering
feature_extractor = SeismicFeatureExtractor()
X_features = feature_extractor.transform(X)

# Combine raw signal and engineered features
X_combined = np.concatenate([X, X_features], axis=1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train).unsqueeze(-1).to(device)
X_test = torch.FloatTensor(X_test).unsqueeze(-1).to(device)
y_train = torch.FloatTensor(y_train).to(device)
y_test = torch.FloatTensor(y_test).to(device)

# Create DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Define the model
class HybridModel(nn.Module):
    def __init__(self, input_dim):
        super(HybridModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, padding='same')
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding='same')
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
        
        self.fc1 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(128, 1)
        
    def forward(self, x):
        # CNN branch
        x_conv = self.conv1(x)
        x_conv = self.bn1(x_conv)
        x_conv = torch.relu(x_conv)
        x_conv = self.dropout1(x_conv)
        
        x_conv = self.conv2(x_conv)
        x_conv = self.bn2(x_conv)
        x_conv = torch.relu(x_conv)
        x_conv = self.dropout2(x_conv)
        
        # LSTM branch
        x_lstm, _ = self.lstm(x.transpose(1, 2))
        
        # Attention
        attn_output, _ = self.attention(x_lstm, x_lstm, x_lstm)
        
        # Combine features
        x_conv = x_conv.mean(dim=-1)
        x_lstm = attn_output.mean(dim=1)
        x = torch.cat([x_conv, x_lstm], dim=1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        return x

# Initialize model, loss function and optimizer
model = HybridModel(X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

# Training loop
num_epochs = 150
best_loss = float('inf')
patience = 15
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs.squeeze(), y_test)
    
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

# Load best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluation
model.eval()
with torch.no_grad():
    predictions = model(X_test).cpu().numpy().flatten()
    y_test_np = y_test.cpu().numpy()
    
    mae = np.mean(np.abs(y_test_np - predictions))
    mape = np.mean(np.abs((y_test_np - predictions) / y_test_np)) * 100
    
    print(f"\nTest MAE: {mae:.4f}, Test MAPE: {mape:.2f}%")

# Enhanced visualization
def plot_results(y_test, predictions):
    plt.figure(figsize=(18, 12))
    
    # Actual vs Predicted
    plt.subplot(2, 2, 1)
    plt.plot(y_test_np, 'g-', label='Actual Sand Fraction')
    plt.plot(predictions, 'r--', label='Predicted Sand Fraction')
    plt.title('Actual vs Predicted Sand Fraction')
    plt.xlabel('Sample Index')
    plt.ylabel('Sand Fraction')
    plt.legend()
    plt.grid(True)
    
    # Residuals plot
    residuals = y_test_np - predictions
    plt.subplot(2, 2, 2)
    plt.scatter(range(len(residuals)), residuals, color='blue', alpha=0.6)
    plt.axhline(0, color='red', linestyle='--', linewidth=2)
    plt.title('Residuals (Actual - Predicted)')
    plt.xlabel('Sample Index')
    plt.ylabel('Residuals')
    plt.grid(True)
    
    # Error distribution
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, color='purple', alpha=0.7)
    plt.title('Error Distribution')
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_results(y_test_np, predictions)
