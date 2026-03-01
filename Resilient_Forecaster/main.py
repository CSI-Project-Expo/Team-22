import numpy as np  # Numerical operations (arrays, math functions)
import matplotlib.pyplot as plt  # For plotting graphs
from src.defender import detect_and_restore  # Defense mechanism module
from src.data_loader import download_stock_data  # Function to download stock data
from src.utils import load_data, scale_data, create_sequences, calculate_mse  # Utility helper functions
from src.model_lstm import build_model, train_model  # LSTM model creation and training
from src.attacker import poison_training_labels  # Function to simulate data poisoning attack


# ==============================
# 🔒 Fix randomness for stable demo
# ==============================
np.random.seed(42)  
# Fixes random values so results remain the same every time we run the code


# ==============================
# 1️⃣ Download Data
# ==============================
download_stock_data("AAPL")  
# Downloads historical stock data of Apple (AAPL) and saves it locally


# ==============================
# 2️⃣ Load & Scale
# ==============================
data = load_data("data/raw_stocks.csv")  
# Loads stock data from CSV file

scaled_data, scaler = scale_data(data)  
# Normalizes the data (usually between 0 and 1) for better LSTM performance
# 'scaler' is saved to convert predictions back to original values later


# ==============================
# 3️⃣ Create Sequences
# ==============================
window_size = 90  
# Model will look at past 90 days to predict next day

X, y = create_sequences(scaled_data, window_size)  
# Converts time-series data into input-output sequences for LSTM
# X = past 90 days data
# y = next day's price


# ==============================
# 4️⃣ Train/Test Split
# ==============================
split = int(len(X) * 0.8)  
# 80% data for training, 20% for testing

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
# Splitting inputs and labels into training and testing sets


# ==============================
# 5️⃣ Poison Training Labels (Stronger Attack)
# ==============================
y_train_poisoned = poison_training_labels(
    y_train,
    poison_rate=0.4,     # 40% of training labels will be corrupted
    spike_factor=1.2     # Corrupted values increased by 20%
)
# This simulates a malicious attack by altering some training labels


# ==============================
# 6️⃣ Train CLEAN Model
# ==============================
clean_model = build_model((X_train.shape[1], 1))  
# Build LSTM model with input shape (time_steps, features)

train_model(clean_model, X_train, y_train)  
# Train model on clean (original) data

clean_predictions = clean_model.predict(X_test)  
# Predict on test data using clean model


# ==============================
# 7️⃣ Train POISONED Model
# ==============================
poisoned_model = build_model((X_train.shape[1], 1))  
# Build another LSTM model

train_model(poisoned_model, X_train, y_train_poisoned)  
# Train model on poisoned (corrupted) data

poisoned_predictions = poisoned_model.predict(X_test)  
# Predictions from poisoned model


# ==============================
# 8️⃣ DEFENSE (Deviation-based Restore)
# ==============================

y_train_defended = detect_and_restore(
    y_clean=y_train,
    y_poisoned=y_train_poisoned,
    threshold=0.25
)
# Defense mechanism:
# If difference between clean and poisoned label > 25%, restore original value

defended_model = build_model((X_train.shape[1], 1))  
# Build new model for defended training

train_model(defended_model, X_train, y_train_defended)  
# Train model on restored (defended) labels

defended_predictions = defended_model.predict(X_test)  
# Predictions after defense mechanism


# ==============================
# 9️⃣ Inverse Scaling
# ==============================
clean_predictions = scaler.inverse_transform(clean_predictions)
poisoned_predictions = scaler.inverse_transform(poisoned_predictions)
defended_predictions = scaler.inverse_transform(defended_predictions)
# Convert predictions back to original stock price scale

y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
# Convert actual test values back to original scale


# ==============================
# 🔟 Calculate MSE
# ==============================
clean_mse = calculate_mse(y_test_actual, clean_predictions)
poisoned_mse = calculate_mse(y_test_actual, poisoned_predictions)
defended_mse = calculate_mse(y_test_actual, defended_predictions)
# Calculate Mean Squared Error for all three models

print("\n===== MODEL PERFORMANCE =====")
print("Clean LSTM MSE:", clean_mse)
print("Poisoned LSTM MSE:", poisoned_mse)
print("Defended LSTM MSE:", defended_mse)
# Displays performance comparison

print("\n===== IMPACT ANALYSIS =====")
print("Poison Impact (%):", 
      ((poisoned_mse - clean_mse) / clean_mse) * 100)
# Shows how much performance degraded due to poisoning

print("Defense Recovery (%):", 
      ((poisoned_mse - defended_mse) / poisoned_mse) * 100)
# Shows how much performance improved after defense


# ==============================
# 1️⃣1️⃣ Plot Results
# ==============================
plt.figure(figsize=(14, 6))
# Create large figure for better visibility

plt.plot(y_test_actual, label="Actual", linewidth=2)
# Actual stock prices

plt.plot(clean_predictions, label="Clean Model")
# Predictions from clean model

plt.plot(poisoned_predictions, label="Poisoned Model")
# Predictions from poisoned model

plt.plot(defended_predictions, label="Defended Model")
# Predictions from defended model

plt.legend()
plt.grid(True)
plt.title("Impact of Data Poisoning and Defense on LSTM Stock Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.show()
# Displays comparison graph