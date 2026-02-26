# 🔐 Resilient Forecaster
### Defending LSTM Models Against Data Poisoning Attacks

---

## 📌 Overview
**Resilient Forecaster** demonstrates how machine learning models can be vulnerable to **data poisoning attacks** and how defensive mechanisms can restore model performance. 

We simulate an attack on an LSTM-based stock price prediction system and measure:
* 📉 **Performance degradation** due to poisoning.
* 🛡 **Recovery** after applying statistical defense.

---

## 🎯 Problem Statement
Machine learning models rely heavily on clean training data. If an attacker manipulates training labels, the model learns incorrect patterns.

> **Can we detect and recover from data poisoning using automated defense?**

---

## 🧠 Project Architecture
The workflow follows a modular pipeline:
1. **Raw Stock Data:** Ingested via `yFinance`.
2. **Preprocessing:** Scaling and sequence creation (Time-Series Windows).
3. **Execution:** Clean Training → Poisoned Training → Defended Training.
4. **Comparison:** Evaluation via Mean Squared Error (MSE).

---

## ⚙️ Technologies Used
* **Python** (Core Language)
* **TensorFlow / Keras** (Deep Learning)
* **NumPy & Pandas** (Data Manipulation)
* **Scikit-learn** (Preprocessing)
* **Matplotlib** (Visualization)
* **yFinance** (Stock Data)

---

## 🗂 Project Structure
```text
Resilient_Forecaster/
├── src/
│   ├── attacker.py        # Poisoning logic
│   ├── defender.py        # Restoration logic
│   ├── model_lstm.py      # LSTM architecture
│   └── data_loader.py     # Data fetching
├── main.py                # Main pipeline
└── README.md              # Documentation
