"""
Prediction script for forecasting lake temperatures.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import LakeDataLoader
from model import LakeTemperatureModel


def plot_predictions(dates, actual, predicted, forecast_dates=None, 
                     forecast_temps=None, save_path='plots/predictions.png'):
    """Plot actual vs predicted temperatures with optional forecast."""
    plt.figure(figsize=(14, 6))
    
    plt.plot(dates, actual, label='Actual Temperature', color='blue', linewidth=2)
    plt.plot(dates, predicted, label='Predicted Temperature', 
             color='red', linestyle='--', linewidth=2)
    
    if forecast_dates is not None and forecast_temps is not None:
        plt.plot(forecast_dates, forecast_temps, 
                label='7-Day Forecast', color='green', 
                linestyle=':', linewidth=2, marker='o')
    
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.title('Lake St. Benedict Temperature Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Prediction plot saved to {save_path}")
    plt.close()


def forecast_future(model, loader, last_sequence, days=7):
    """
    Forecast temperatures for future days.
    
    Args:
        model: Trained model
        loader: Data loader with fitted scaler
        last_sequence: Last sequence of temperatures
        days: Number of days to forecast
        
    Returns:
        Forecasted temperatures in Celsius
    """
    forecasts = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days):
        # Predict next value
        prediction = model.predict(current_sequence.reshape(1, -1, 1))
        forecasts.append(prediction[0, 0])
        
        # Update sequence (roll and append new prediction)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = prediction[0, 0]
    
    # Convert back to Celsius
    forecasts_array = np.array(forecasts).reshape(-1, 1)
    forecasts_celsius = loader.inverse_transform(forecasts_array)
    
    return forecasts_celsius.flatten()


def main():
    """Main prediction function."""
    print("=" * 60)
    print("Lake St. Benedict Temperature Prediction")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = 'data/sample_data.csv'
    MODEL_PATH = 'models/lake_temp_model.h5'
    SEQUENCE_LENGTH = 30
    FORECAST_DAYS = 7
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\nError: Model not found at {MODEL_PATH}")
        print("Please run train.py first to train the model.")
        return
    
    # Load data
    print("\n1. Loading data...")
    loader = LakeDataLoader(DATA_PATH)
    data = loader.load_data()
    print(f"   Loaded {len(data)} days of temperature data")
    
    # Prepare sequences
    print("\n2. Preparing sequences...")
    X, y = loader.prepare_sequences(sequence_length=SEQUENCE_LENGTH)
    print(f"   Created {len(X)} sequences")
    
    # Load trained model
    print("\n3. Loading trained model...")
    model = LakeTemperatureModel(sequence_length=SEQUENCE_LENGTH)
    model.load_model(MODEL_PATH)
    print(f"   Model loaded from {MODEL_PATH}")
    
    # Make predictions on the last 60 days
    print("\n4. Making predictions on recent data...")
    recent_X = X[-60:]
    recent_y = y[-60:]
    predictions = model.predict(recent_X)
    
    # Convert back to Celsius
    predictions_celsius = loader.inverse_transform(predictions)
    actual_celsius = loader.inverse_transform(recent_y)
    
    # Calculate error
    mae = np.mean(np.abs(predictions_celsius - actual_celsius))
    print(f"   Mean Absolute Error: {mae:.2f}°C")
    
    # Get dates for recent predictions
    recent_dates = data['date'].iloc[-60:].values
    
    # Forecast future temperatures
    print(f"\n5. Forecasting next {FORECAST_DAYS} days...")
    last_sequence = X[-1].flatten()
    forecast = forecast_future(model, loader, last_sequence, days=FORECAST_DAYS)
    
    # Generate forecast dates
    last_date = pd.to_datetime(data['date'].iloc[-1])
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(FORECAST_DAYS)]
    
    print(f"\n   7-Day Temperature Forecast:")
    print("   " + "-" * 40)
    for date, temp in zip(forecast_dates, forecast):
        print(f"   {date.strftime('%Y-%m-%d')}: {temp:.2f}°C")
    print("   " + "-" * 40)
    
    # Plot results
    print("\n6. Generating visualization...")
    plot_predictions(
        recent_dates, 
        actual_celsius.flatten(), 
        predictions_celsius.flatten(),
        forecast_dates,
        forecast
    )
    
    print("\n" + "=" * 60)
    print("Prediction completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
