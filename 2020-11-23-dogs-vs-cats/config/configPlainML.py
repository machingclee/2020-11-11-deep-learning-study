import plaidml.keras
import os
plaidml.keras.install_backend()
os.environ["keras_BACKEND"] = "plaidml.keras.backend"
