🛡️ The Resilient Forecaster
Adversarial Machine Learning: Data Poisoning Attack & Defense in Stock Forecasting
📌 Project Overview
In the world of FinTech, AI models make split-second decisions involving millions of dollars. But what happens if an attacker "poisons" the data the AI learns from?

This project demonstrates a Data Poisoning Attack on a Stock Price Forecasting LSTM model and implements a robust Defense Mechanism using an Isolation Forest to detect and sanitize malicious data.

The Three-Pillar Architecture
The Brain (Baseline LSTM): A Long Short-Term Memory network trained on 5 years of historical stock data to predict future prices.

The Attack (Data Poisoning): An adversarial script that injects "Label Flipping" and "Feature Spikes" into the training set to misguide the model.

The Shield (Isolation Forest): An unsupervised anomaly detection system that identifies and removes poisoned data points before they reach the model.

📂 Project Structure

Resilient_Forecaster/
├── data/                   # Raw and poisoned stock datasets (CSV)
├── models/                 # Saved .keras model files
├── src/                    # Core Logic
│   ├── data_loader.py      # yfinance API integration
│   ├── model_lstm.py       # Neural network architecture
│   ├── attacker.py         # Poisoning (Label Flipping & Injection)
│   ├── defender.py         # Isolation Forest defense logic
│   └── utils.py            # Plotting and scaling helpers
├── main.py                 # Master execution script
├── requirements.txt        # Dependencies
└── README.md               # You are here!

🚀 Getting Started
1. Prerequisites
Ensure you have Python 3.10+ installed. We recommend using a virtual environment.

2. Installation

# Clone the repository
git clone https://github.com/your-username/resilient-forecaster.git
cd resilient-forecaster

# Install dependencies
pip install -r requirements.txt

3. Running the Pipeline
You can run the entire experiment—from data collection to defense—with one command:

python main.py

🛠️ Methodology
The Attack: Label Flipping
We simulate a sophisticated attacker who gains access to the data pipeline. The attacker modifies 10% of the target prices, introducing 20-30% volatility spikes. This forces the LSTM to "learn" incorrect patterns, leading to disastrous prediction errors.

The Defense: Isolation Forest
Because stock data is sequential, we use an Isolation Forest (an ensemble of decision trees) to isolate observations. Poisoned data points appear as "outliers" in the feature space.

Contamination Factor: 0.1 (10%)

Action: Flagged points are removed or replaced using a linear interpolation of the previous 48 hours of data.

⚖️ Disclaimer
This project is for educational purposes only. It is designed to highlight vulnerabilities in machine learning pipelines and test defensive strategies. Do not use this for actual financial trading.

