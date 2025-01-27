from src.data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from model_training import train_models
from src.portfolio_optimization import optimize_portfolio

def main():
    # Step 1: Create necessary directories
    print("Step 1: Creating directories...")
    create_directories()

    # Step 2: Define stock tickers and date range
    print("Step 2: Downloading stock data...")
    tickers = ["AAPL", "MSFT", "TSLA", "GOOGL", "AMZN"]  # Example stock symbols
    start_date = "2018-01-01"
    end_date = "2024-12-31"
    download_stock_data(tickers, start_date, end_date)

    # Step 3: Preprocess downloaded data
    print("Step 3: Preprocessing data...")
    preprocess_data()

    # Step 4: Perform feature engineering
    print("Step 4: Engineering features...")
    feature_engineering()

    # Step 5: Train models and generate predictions
    print("Step 5: Training models...")
    predictions = train_models()

    # Step 6: Optimize portfolio based on predictions
    print("Step 6: Optimizing portfolio...")
    optimize_portfolio(predictions)


if __name__ == "__main__":
    main()
