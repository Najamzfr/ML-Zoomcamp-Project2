# Step 4: Feature Engineering
def feature_engineering():
    processed_data_path = "data/processed"
    for file in os.listdir(processed_data_path):
        if file.endswith(".csv"):
            print(f"Engineering features for {file}...")
            file_path = os.path.join(processed_data_path, file)
            data = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

            # Generate lag features
            data["Lag_1"] = data["Close"].shift(1)
            data["Lag_2"] = data["Close"].shift(2)

            # Calculate moving averages
            data["MA_5"] = data["Close"].rolling(window=5).mean()
            data["MA_10"] = data["Close"].rolling(window=10).mean()

            # Drop rows with NaN values introduced by lagging/rolling
            data = data.dropna()

            # Save engineered data
            feature_file_path = os.path.join(processed_data_path, f"features_{file}")
            data.to_csv(feature_file_path)
            print(f"Saved feature-engineered data to {feature_file_path}")