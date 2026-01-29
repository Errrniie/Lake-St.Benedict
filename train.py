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
    
    # Load data
    print("\n1. Loading data...")
    loader = LakeDataLoader(DATA_PATH)
    data = loader.load_data()
    print(f"   Loaded {len(data)} days of temperature data")
    
    # Split data before creating sequences to avoid data leakage
    print("\n2. Splitting data...")
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    print(f"   Training data: {len(train_data)} days")
    print(f"   Test data: {len(test_data)} days")
    
    # Prepare training sequences
    print("\n3. Preparing training sequences...")
    train_loader = LakeDataLoader(DATA_PATH)
    train_loader.data = train_data
    X_train_full, y_train_full = train_loader.prepare_sequences(
        sequence_length=SEQUENCE_LENGTH, fit_scaler=True
    )
    print(f"   Created {len(X_train_full)} training sequences")
    
    # Prepare test sequences using the same scaler
    print("\n4. Preparing test sequences...")
    test_loader = LakeDataLoader(DATA_PATH)
    test_loader.data = test_data
    test_loader.scaler = train_loader.scaler  # Use training scaler
    X_test, y_test = test_loader.prepare_sequences(
        sequence_length=SEQUENCE_LENGTH, fit_scaler=False
    )
    print(f"   Created {len(X_test)} test sequences")
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Build and train model
    print("\n5. Building model...")
    model = LakeTemperatureModel(sequence_length=SEQUENCE_LENGTH)
    model.build_model()
    print(f"   Model built with {model.model.count_params()} parameters")
    
    print("\n6. Training model...")
    print("-" * 60)
    history = model.model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        verbose=1
    )
    print("-" * 60)
    
    # Evaluate on test set
    print("\n7. Evaluating on test set...")
    test_loss, test_mae = model.model.evaluate(X_test, y_test, verbose=0)
    print(f"   Test Loss: {test_loss:.4f}")
    print(f"   Test MAE: {test_mae:.4f}")
    
    # Calculate temperature MAE in Celsius
    predictions = model.predict(X_test)
    predictions_celsius = train_loader.inverse_transform(predictions)
    y_test_celsius = train_loader.inverse_transform(y_test)
    celsius_mae = np.mean(np.abs(predictions_celsius - y_test_celsius))
    print(f"   Test MAE (°C): {celsius_mae:.2f}°C")
    
    # Save model and scaler
    print("\n8. Saving model and scaler...")
    os.makedirs('models', exist_ok=True)
    model.save_model(MODEL_SAVE_PATH)
    # Save scaler for use during prediction
    import pickle
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(train_loader.scaler, f)
    print(f"   Model saved to {MODEL_SAVE_PATH}")
    print(f"   Scaler saved to models/scaler.pkl")
    
    # Plot training history
    print("\n9. Plotting training history...")
    os.makedirs('plots', exist_ok=True)
    plot_training_history(history)
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
