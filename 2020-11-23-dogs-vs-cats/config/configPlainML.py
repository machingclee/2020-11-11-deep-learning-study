import plaidml.keras
import os
plaidml.tensorflow.keras.install_backend()
os.environ["keras_BACKEND"] = "plaidml.tensorflow.keras.backend"
