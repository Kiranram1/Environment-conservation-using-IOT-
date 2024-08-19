import tensorflow as tf
import tensorflow_hub as hub

# Load the model from TensorFlow Hub or a saved model directory
model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')

# List available signatures
print("Available signatures:")
for key in model.signatures:
    print(f"Signature name: {key}")
