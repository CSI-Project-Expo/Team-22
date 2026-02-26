from src.utils import load_data, scale_data

data = load_data("data/raw_stocks.csv")
scaled, scaler = scale_data(data)

print("Shape:", scaled.shape)
print("First 5 values:", scaled[:5])