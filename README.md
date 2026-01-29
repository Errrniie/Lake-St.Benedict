# Lake St. Benedict - Temperature Prediction Application

A predictive modeling application for forecasting water temperature in Lake St. Benedict using deep learning (LSTM neural networks).

## Overview

This application uses Long Short-Term Memory (LSTM) neural networks to predict lake water temperatures based on historical data. It can:
- Train a predictive model on historical temperature data
- Make predictions on recent data
- Forecast temperatures for the next 7 days
- Visualize actual vs predicted temperatures

## Features

- **Time Series Prediction**: LSTM-based model for accurate temperature forecasting
- **Historical Analysis**: Evaluate model performance on historical data
- **Future Forecasting**: Predict temperatures up to 7 days ahead
- **Visualization**: Generate plots of predictions and forecasts
- **Easy to Use**: Simple command-line interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Errrniie/Lake-St.Benedict.git
cd Lake-St.Benedict
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Train the predictive model on historical data:

```bash
python app.py train
```

This will:
- Load temperature data from `data/sample_data.csv`
- Train an LSTM model for 50 epochs
- Save the trained model to `models/lake_temp_model.h5`
- Generate training history plots in `plots/training_history.png`

### Making Predictions

Generate predictions and forecasts:

```bash
python app.py predict
```

This will:
- Load the trained model
- Make predictions on recent historical data
- Generate a 7-day temperature forecast
- Create visualization in `plots/predictions.png`

### Using Custom Scripts

You can also run the scripts directly:

```bash
# Train the model
python train.py

# Make predictions
python predict.py
```

## Project Structure

```
Lake-St.Benedict/
├── app.py                  # Main application entry point
├── train.py                # Training script
├── predict.py              # Prediction and forecasting script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── src/                   # Source code modules
│   ├── data_loader.py     # Data loading and preprocessing
│   └── model.py           # LSTM model architecture
├── data/                  # Data directory
│   └── sample_data.csv    # Sample temperature data
├── models/                # Saved models directory
│   └── lake_temp_model.h5 # Trained model (created after training)
└── plots/                 # Visualization outputs
    ├── training_history.png
    └── predictions.png
```

## Data Format

The application expects CSV data with the following format:

```csv
date,temperature
2024-01-01,5.83
2024-01-02,4.86
2024-01-03,6.02
```

- `date`: Date in YYYY-MM-DD format
- `temperature`: Water temperature in degrees Celsius

## Model Architecture

The application uses an LSTM (Long Short-Term Memory) neural network:
- Input: 30 days of historical temperatures
- Architecture: 2 LSTM layers (50 units each) with dropout
- Output: Next day's temperature prediction
- Training: Adam optimizer with MSE loss

## Requirements

- Python 3.8+
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.4.0
- tensorflow >= 2.8.0

## Example Output

After training, the model typically achieves:
- Mean Absolute Error: ~1-2°C on test data
- Captures seasonal trends and patterns
- Provides 7-day forecasts with confidence

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.