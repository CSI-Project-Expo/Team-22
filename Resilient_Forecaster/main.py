import numpy as np
import matplotlib.pyplot as plt
from src.defender import detect_and_restore
from src.data_loader import download_stock_data
from src.utils import load_data, scale_data, create_sequences, calculate_mse
from src.model_lstm import build_model, train_model
from src.attacker import poison_training_labels


# ==============================
# 🔒 Fix randomness for stable demo
# ==============================
np.random.seed(42)

# ==============================
# 1️⃣ Download Data
# ==============================
download_stock_data("AAPL")

# ==============================
# 2️⃣ Load & Scale
# ==============================
data = load_data("data/raw_stocks.csv")
scaled_data, scaler = scale_data(data)

# ==============================
# 3️⃣ Create Sequences
# ==============================
window_size = 90
X, y = create_sequences(scaled_data, window_size)

# ==============================
# 4️⃣ Train/Test Split
# ==============================
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ==============================
# 5️⃣ Poison Training Labels (Stronger Attack)
# ==============================
y_train_poisoned = poison_training_labels(
    y_train,
    poison_rate=0.4,     # Increased
    spike_factor=1.2     # Stronger corruption
)

# ==============================
# 6️⃣ Train CLEAN Model
# ==============================
clean_model = build_model((X_train.shape[1], 1))
train_model(clean_model, X_train, y_train)
clean_predictions = clean_model.predict(X_test)

# ==============================
# 7️⃣ Train POISONED Model
# ==============================
poisoned_model = build_model((X_train.shape[1], 1))
train_model(poisoned_model, X_train, y_train_poisoned)
poisoned_predictions = poisoned_model.predict(X_test)

# ==============================
# 8️⃣ DEFENSE (Deviation-based Restore)
# ==============================

y_train_defended = detect_and_restore(
    y_clean=y_train,
    y_poisoned=y_train_poisoned,
    threshold=0.25
)

defended_model = build_model((X_train.shape[1], 1))
train_model(defended_model, X_train, y_train_defended)
defended_predictions = defended_model.predict(X_test)

# ==============================
# 9️⃣ Inverse Scaling
# ==============================
clean_predictions = scaler.inverse_transform(clean_predictions)
poisoned_predictions = scaler.inverse_transform(poisoned_predictions)
defended_predictions = scaler.inverse_transform(defended_predictions)

y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# ==============================
# 🔟 Calculate MSE
# ==============================
clean_mse = calculate_mse(y_test_actual, clean_predictions)
poisoned_mse = calculate_mse(y_test_actual, poisoned_predictions)
defended_mse = calculate_mse(y_test_actual, defended_predictions)

print("\n===== MODEL PERFORMANCE =====")
print("Clean LSTM MSE:", clean_mse)
print("Poisoned LSTM MSE:", poisoned_mse)
print("Defended LSTM MSE:", defended_mse)

print("\n===== IMPACT ANALYSIS =====")
print("Poison Impact (%):", 
      ((poisoned_mse - clean_mse) / clean_mse) * 100)

print("Defense Recovery (%):", 
      ((poisoned_mse - defended_mse) / poisoned_mse) * 100)

# ==============================
# 1️⃣1️⃣ Plot Results
# ==============================
plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, label="Actual", linewidth=2)
plt.plot(clean_predictions, label="Clean Model")
plt.plot(poisoned_predictions, label="Poisoned Model")
plt.plot(defended_predictions, label="Defended Model")

plt.legend()
plt.grid(True)
plt.title("Impact of Data Poisoning and Defense on LSTM Stock Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.show()