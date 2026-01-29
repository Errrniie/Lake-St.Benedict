"""
Training script for the lake temperature prediction model.
"""
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import LakeDataLoader
from model import LakeTemperatureModel


def plot_training_history(history, save_path='plots/training_history.png'):
    """Plot training history."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss During Training')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.title('Model MAE During Training')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()


def main():
    """Main training function."""
    print("=" * 60)
    print("Lake St. Benedict Temperature Prediction Model Training")
    print("=" * 60)
    
    # Configuration
    DATA_PATH = 'data/sample_data.csv'
    SEQUENCE_LENGTH = 30
    MODEL_SAVE_PATH = 'models/lake_temp_model.h5'
    EPOCHS = 50
    BATCH_SIZE = 32
    
    # Load and prepare data
    print("\n1. Loading data...")
    loader = LakeDataLoader(DATA_PATH)
    loader.load_data()
    print(f"   Loaded {len(loader.data)} days of temperature data")
    
    # Prepare sequences
    print("\n2. Preparing sequences...")
    X, y = loader.prepare_sequences(sequence_length=SEQUENCE_LENGTH)
    print(f"   Created {len(X)} sequences")
    print(f"   Sequence shape: {X.shape}")
    
    # Split data
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    
    # Build and train model
    print("\n4. Building model...")
    model = LakeTemperatureModel(sequence_length=SEQUENCE_LENGTH)
    model.build_model()
    print(f"   Model built with {model.model.count_params()} parameters")
    
    print("\n5. Training model...")
    print("-" * 60)
    history = model.train(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2
    )
    print("-" * 60)
    
    # Evaluate on test set
    print("\n6. Evaluating on test set...")
    test_loss, test_mae = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test MAE: {test_mae:.4f}")
    
    # Calculate temperature MAE in Celsius
    predictions = model.predict(X_test)
    predictions_celsius = loader.inverse_transform(predictions)
    y_test_celsius = loader.inverse_transform(y_test)
    celsius_mae = np.mean(np.abs(predictions_celsius - y_test_celsius))
    print(f"   Test MAE (°C): {celsius_mae:.2f}°C")
    
    # Save model
    print("\n7. Saving model...")
    os.makedirs('models', exist_ok=True)
    model.save_model(MODEL_SAVE_PATH)
    print(f"   Model saved to {MODEL_SAVE_PATH}")
    
    # Plot training history
    print("\n8. Plotting training history...")
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
