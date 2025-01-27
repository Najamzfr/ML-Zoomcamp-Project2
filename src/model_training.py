# Step 5: Model Training
def train_models():
    processed_data_path = "data/processed"
    models_path = "models/saved_models"
    analysis_path = "models/model_analysis"
    figures_path = "reports/figures"

    os.makedirs(models_path, exist_ok=True)
    os.makedirs(analysis_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    predictions = {}

    for file in os.listdir(processed_data_path):
        if file.startswith("features_") and file.endswith(".csv"):
            print(f"Training models for {file}...")
            file_path = os.path.join(processed_data_path, file)
            data = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")

            # Prepare data for Linear Regression
            X = data[["Lag_1", "Lag_2", "MA_5", "MA_10"]]
            y = data["Close"]

            # Split into training and testing sets
            train_size = int(len(data) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            # Linear Regression Model
            lr_model = LinearRegression()
            lr_model.fit(X_train, y_train)
            lr_predictions = lr_model.predict(X_test)

            # Save Linear Regression Model
            lr_model_path = os.path.join(models_path, f"linear_regression_{file}.pkl")
            with open(lr_model_path, "wb") as f:
                pickle.dump(lr_model, f)
            print(f"Saved Linear Regression model to {lr_model_path}")

            # Evaluate Linear Regression Model
            lr_mse = mean_squared_error(y_test, lr_predictions)
            lr_rmse = np.sqrt(lr_mse)
            print(f"Linear Regression RMSE for {file}: {lr_rmse}")

            # Align predictions by index (dates)
            predictions[file] = pd.Series(lr_predictions, index=y_test.index)

    return predictions