"""
Data module for loading and preprocessing lake temperature data.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class LakeDataLoader:
    """Handles loading and preprocessing of lake temperature data."""
    
    def __init__(self, filepath):
        """
        Initialize the data loader.
        
        Args:
            filepath: Path to the CSV file containing lake data
        """
        self.filepath = filepath
        self.data = None
        self.scaler = MinMaxScaler()
        
    def load_data(self):
        """Load data from CSV file."""
        self.data = pd.read_csv(self.filepath)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data = self.data.sort_values('date')
        return self.data
    
    def prepare_sequences(self, sequence_length=30):
        """
        Prepare sequences for time series prediction.
        
        Args:
            sequence_length: Number of past days to use for prediction
            
        Returns:
            X: Input sequences
            y: Target values
        """
        if self.data is None:
            self.load_data()
        
        # Extract temperature values
        temperatures = self.data['temperature'].values.reshape(-1, 1)
        
        # Normalize the data
        scaled_temps = self.scaler.fit_transform(temperatures)
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_temps) - sequence_length):
            X.append(scaled_temps[i:i + sequence_length])
            y.append(scaled_temps[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def inverse_transform(self, scaled_data):
        """Convert normalized data back to original scale."""
        return self.scaler.inverse_transform(scaled_data)
