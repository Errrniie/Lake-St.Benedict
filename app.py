"""
Main application entry point for Lake St. Benedict predictive model.
"""
import argparse
import sys
import os

# Add scripts to path
sys.path.append(os.path.dirname(__file__))


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='Lake St. Benedict Temperature Prediction Application',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train the model
  python app.py train
  
  # Make predictions and forecasts
  python app.py predict
  
  # Get help on a specific command
  python app.py train --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the prediction model')
    train_parser.add_argument('--epochs', type=int, default=50,
                             help='Number of training epochs (default: 50)')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Batch size for training (default: 32)')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', 
                                          help='Make predictions and forecasts')
    predict_parser.add_argument('--days', type=int, default=7,
                               help='Number of days to forecast (default: 7)')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("Starting model training...")
        import train
        train.main()
    elif args.command == 'predict':
        print("Starting predictions...")
        import predict
        predict.main()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
