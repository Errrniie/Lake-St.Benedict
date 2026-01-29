"""
Predictive model module for lake temperature forecasting.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


class LakeTemperatureModel:
    """LSTM-based model for predicting lake temperature."""
    
    def __init__(self, sequence_length=30):
        """
        Initialize the temperature prediction model.
        
        Args:
            sequence_length: Number of past days to use for prediction
        """
        self.sequence_length = sequence_length
        self.model = None
        
    def build_model(self):
        """Build the LSTM model architecture."""
        self.model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, 
                 input_shape=(self.sequence_length, 1)),
            Dropout(0.2),
            LSTM(50, activation='relu'),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return self.model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the model.
        
        Args:
            X_train: Training input sequences
            y_train: Training target values
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history
        """
        if self.model is None:
            self.build_model()
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input sequences
            
        Returns:
            Predicted temperatures
        """
        return self.model.predict(X, verbose=0)
    
    def save_model(self, filepath):
        """Save the trained model."""
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load a pre-trained model."""
        self.model = keras.models.load_model(filepath)
