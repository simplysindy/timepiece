"""
Neural network models for watch price prediction.

Simple LSTM implementation for sequential learning.
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LSTMModel:
    """Simple LSTM model for time series forecasting."""
    
    def __init__(self, sequence_length: int = 30, lstm_units: int = 50,
                 dropout_rate: float = 0.2, epochs: int = 100,
                 batch_size: int = 32):
        """
        Initialize LSTM model.
        
        Parameters:
        ----------
        sequence_length : int
            Length of input sequences
        lstm_units : int
            Number of LSTM units
        dropout_rate : float
            Dropout rate for regularization
        epochs : int
            Training epochs
        batch_size : int
            Training batch size
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = None
        self.is_fitted = False
        self.scaler = None
        
    def _prepare_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training."""
        X_seq, y_seq = [], []
        
        if len(X) <= self.sequence_length:
            raise ValueError(f"Need more than {self.sequence_length} samples for sequence creation")
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:(i + self.sequence_length)])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit LSTM model."""
        try:
            # Configure TensorFlow to suppress warnings
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            
            import tensorflow as tf
            tf.get_logger().setLevel('ERROR')
            
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            from sklearn.preprocessing import StandardScaler
            
        except ImportError:
            logger.error("TensorFlow not installed. Install with: pip install tensorflow")
            raise
        
        # Convert to numpy arrays
        X_array = X.values if hasattr(X, 'values') else X
        y_array = y.values if hasattr(y, 'values') else y
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_array)
        
        # Create sequences
        X_seq, y_seq = self._prepare_sequences(X_scaled, y_array)
        
        # Build model
        self.model = Sequential([
            LSTM(self.lstm_units, return_sequences=True, input_shape=(self.sequence_length, X_seq.shape[2])),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units // 2, return_sequences=False),
            Dropout(self.dropout_rate),
            Dense(25, activation='relu'),
            Dropout(self.dropout_rate),
            Dense(1)
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train model with reduced verbosity
        try:
            history = self.model.fit(
                X_seq, y_seq,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                verbose=0,  # Silent training
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=10,
                        restore_best_weights=True,
                        verbose=0
                    )
                ]
            )
            
            self.is_fitted = True
            logger.info(f"LSTM fitted with {len(X_seq)} sequences, {len(history.history['loss'])} epochs")
            
        except Exception as e:
            logger.error(f"LSTM training failed: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with LSTM."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_array = X.values if hasattr(X, 'values') else X
        
        # Scale features
        X_scaled = self.scaler.transform(X_array)
        
        # Create sequences for prediction
        if len(X_scaled) < self.sequence_length:
            # If we don't have enough data for sequences, repeat the last available sequence
            if len(X_scaled) == 0:
                raise ValueError("No data available for prediction")
            
            # Pad with the last available sample
            padding_needed = self.sequence_length - len(X_scaled)
            last_sample = X_scaled[-1:] if len(X_scaled) > 0 else np.zeros((1, X_scaled.shape[1]))
            X_padded = np.vstack([np.repeat(last_sample, padding_needed, axis=0), X_scaled])
            X_seq = X_padded[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            predictions = self.model.predict(X_seq, verbose=0)
            return np.repeat(predictions[0], len(X))
        
        # Normal sequence creation
        X_seq, _ = self._prepare_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        # Make predictions
        predictions = self.model.predict(X_seq, verbose=0)
        
        # Pad predictions to match input length
        padded_predictions = np.full(len(X), predictions[-1] if len(predictions) > 0 else 0.0)
        padded_predictions[-len(predictions):] = predictions.flatten()
        
        return padded_predictions
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """LSTM doesn't have traditional feature importance."""
        if not self.is_fitted:
            return None
        
        # Return dummy importance based on LSTM architecture
        return {
            'lstm_layer_1': 0.4,
            'lstm_layer_2': 0.3,
            'dense_layer': 0.2,
            'sequence_length': 0.1
        }
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get LSTM hyperparameters."""
        return {
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'epochs': self.epochs,
            'batch_size': self.batch_size
        }