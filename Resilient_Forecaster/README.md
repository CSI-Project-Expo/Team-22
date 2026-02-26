# 🔐 Resilient Forecaster
### Defending LSTM Models Against Data Poisoning Attacks

---

## 📌 Overview
**Resilient Forecaster** demonstrates how machine learning models can be vulnerable to **data poisoning attacks** and how defensive mechanisms can restore model performance. 

We simulate an attack on an LSTM-based stock price prediction system and measure:
* 📉 **Performance degradation** due to poisoning.
* 🛡 **Recovery** after applying statistical defense.

This project highlights the critical importance of **data integrity in AI systems**.

---

## 🎯 Problem Statement
Machine learning models rely heavily on clean training data. If an attacker manipulates training labels, the model learns incorrect patterns, leading to biased or failed predictions.

**This project answers:**
> What happens when training data is poisoned?  
> Can we detect and recover from it using automated defense?

---

## 🧠 Project Architecture



The workflow follows a modular pipeline:
1. **Raw Stock Data:** Ingested via `yFinance`.
2. **Preprocessing:** Scaling and sequence creation (Time-Series Windows).
3. **Experimental Execution:**
    * 🔹 **Clean Model Training:** Baseline performance.
    * 🔹 **Poisoned Model Training:** Performance under attack.
    * 🔹 **Defended Model Training:** Performance after restoration.
4. **Performance Comparison:** Evaluation via Mean Squared Error (MSE).

---

## ⚙️ Technologies Used
* **Python** (Core Language)
* **TensorFlow / Keras** (Deep Learning - LSTM/GRU)
* **NumPy & Pandas** (Data Manipulation)
* **Scikit-learn** (Preprocessing & Metrics)
* **Matplotlib** (Visualization)
* **yFinance** (Real-time Stock Data)

---

## 🗂 Project Structure
```text
Resilient_Forecaster/
│
├── src/
│   ├── attacker.py        # Data poisoning & spike injection logic
│   ├── defender.py        # Detection & restoration logic
│   ├── model_lstm.py      # LSTM model architecture
│   ├── model_gru.py       # GRU comparison model
│   ├── data_loader.py     # Stock data download & windowing
│   └── utils.py           # Helper functions for plotting
│
├── data/                  # Local storage for datasets
├── main.py                # Complete execution pipeline
├── requirements.txt       # Dependency list
└── README.md              # Documentation