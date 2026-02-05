import logging
import numpy as np
import tensorflow as tf

class IPOForecaster:
    def __init__(self, shared_state, config, **kwargs):
        """
        Initializes the IPOForecaster with shared state, configuration, and optional keyword arguments.

        :param shared_state: An object or dictionary for sharing state across different components.
        :param config: A configuration object or dictionary containing settings for the forecaster.
        :param kwargs: Optional keyword arguments to store additional dependencies or parameters.
        """
        self.shared_state = shared_state
        self.config = config
        self.logger = logging.getLogger("IPOForecaster")
        self.logger.setLevel(logging.INFO)
        # The model path can now be retrieved from the config object, using getattr for attribute access
        model_path = getattr(config, 'IPO_FORECASTER_MODEL_PATH', 'models/ipo_forecaster.h5')
        self.model = self.load_model(model_path)

        # Optional: store additional dependencies from kwargs
        self.additional_dependencies = kwargs


    def load_model(self, path):
        """
        Loads the TensorFlow Keras model from the given path.

        :param path: The file path to the Keras model.
        :return: The loaded Keras model, or None if loading fails.
        """
        try:
            model = tf.keras.models.load_model(path)
            self.logger.info(f"✅ Loaded IPO forecasting model from {path}")
            return model
        except Exception as e:
            self.logger.error(f"❌ Failed to load IPO forecasting model: {e}")
            return None

    def predict_gain(self, price_sequence):
        """
        Predicts the expected gain based on a sequence of recent close prices.
        The model expects a sequence of 10 prices. If more are provided,
        only the last 10 will be used.

        :param price_sequence: List or array of recent close prices (length 10 recommended).
                               If the length is greater than 10, the last 10 elements are used.
        :return: float predicted percentage gain (e.g., 0.08 = +8%). Returns 0.0
                 if the model is not loaded or prediction fails.
        """
        if self.model is None:
            self.logger.warning("⚠️ Prediction skipped — model is not loaded.")
            return 0.0

        try:
            # Ensure the input data is a numpy array and reshape it
            # The model expects input shape: (batch_size, sequence_length, features)
            # Here, batch_size is 1, sequence_length is 10, features is 1 (for price).
            input_data = np.array(price_sequence[-10:]).reshape(1, -1, 1)

            # Make the prediction using the loaded model
            prediction = self.model.predict(input_data, verbose=0)

            # Extract the gain from the prediction output.
            # Assuming the model outputs a single float for gain.
            gain = float(prediction[0][0])
            return gain
        except Exception as e:
            self.logger.error(f"⚠️ Prediction failed: {e}")
            return 0.0

# Example Usage (uncomment to test if you have a model file)
# if __name__ == "__main__":
#     # Configure basic logging
#     logging.basicConfig(level=logging.INFO)

#     # Create a dummy model for demonstration purposes if you don't have a real one
#     # In a real scenario, you would train and save a model.
#     try:
#         # Define a simple sequential model (e.g., a simple LSTM or Dense network)
#         dummy_model = tf.keras.Sequential([
#             tf.keras.layers.Input(shape=(10, 1)), # Input shape: 10 time steps, 1 feature (price)
#             tf.keras.layers.LSTM(32, activation='relu'),
#             tf.keras.layers.Dense(1)
#         ])
#         dummy_model.compile(optimizer='adam', loss='mse')
#         # Ensure the 'models' directory exists
#         import os
#         os.makedirs('models', exist_ok=True)
#         dummy_model.save('models/ipo_forecaster.h5')
#         print("Dummy model created and saved for testing.")
#     except Exception as e:
#         print(f"Could not create dummy model: {e}. Please ensure TensorFlow is installed.")
#         print("If you have a real model, ensure it's at 'models/ipo_forecaster.h5'")

#     # Initialize the forecaster with dummy shared_state and config
#     class DummyConfig:
#         IPO_FORECASTER_MODEL_PATH = 'models/ipo_forecaster.h5'
#         some_other_setting = 'value'

#     dummy_shared_state = {}
#     dummy_config = DummyConfig()
#     forecaster = IPOForecaster(dummy_shared_state, dummy_config, extra_param='test')

#     # Example price sequence (last 10 close prices)
#     # In a real application, this would come from market data.
#     sample_price_sequence = [100, 101, 100.5, 102, 103, 102.5, 104, 105, 104.5, 106]

#     # Predict the gain
#     predicted_gain = forecaster.predict_gain(sample_price_sequence)

#     if predicted_gain != 0.0:
#         print(f"Predicted IPO gain: {predicted_gain:.2%}")
#     else:
#         print("Could not predict gain. Check logs for errors.")

#     # Test with a longer sequence
#     longer_sequence = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106]
#     predicted_gain_longer = forecaster.predict_gain(longer_sequence)
#     if predicted_gain_longer != 0.0:
#         print(f"Predicted IPO gain with longer sequence (last 10 used): {predicted_gain_longer:.2%}")
