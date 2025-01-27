def preprocess_data():
    raw_data_path = "data/raw"
    processed_data_path = "data/processed"

    for file in os.listdir(raw_data_path):
        if file.endswith(".csv"):
            print(f"Processing {file}...")
            file_path = os.path.join(raw_data_path, file)
            data = pd.read_csv(file_path, header=2, parse_dates=["Date"], index_col="Date")  # Start from row 3
            data = df.rename(columns={'Unnamed: 1': 'Close',
                                      'Unnamed: 2': 'High',
                                      'Unnamed: 3': 'Low',
                                      'Unnamed: 4': 'Open',
                                      'Unnamed: 5': 'Volume'})

            # Retain only relevant columns
            data = data[["Close"]]

            # Remove NaN rows (if any)
            data = data.dropna()

            # Save the processed data
            processed_file_path = os.path.join(processed_data_path, file)
            data.to_csv(processed_file_path)
            print(f"Saved processed data to {processed_file_path}")