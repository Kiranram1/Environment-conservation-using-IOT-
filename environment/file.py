import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import io
import librosa

# Load the YAMNet model.
model = hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')

# Load and preprocess the audio file using librosa.
def load_and_preprocess_audio(file_path, sample_rate=16000):
    # Load the audio file
    waveform, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    # Ensure the waveform is 1D (no batch dimension)
    return waveform

# Split the audio into chunks of specified duration (in seconds).
def split_audio_into_chunks(waveform, sample_rate, chunk_duration=10):
    chunk_size = chunk_duration * sample_rate
    return [waveform[i:i + chunk_size] for i in range(0, len(waveform), chunk_size)]

# Find the class names from the CSV file content.
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]  # Skip CSV header
    return class_names

# Path to your audio file
audio_file_path = '/home/mk14/Downloads/audio_sample/gunsound/AK-47/hmmmm.wav'

# Load and preprocess your audio file
waveform = load_and_preprocess_audio(audio_file_path)

# Split the waveform into 10-second chunks
sample_rate = 16000
chunks = split_audio_into_chunks(waveform, sample_rate)

# Retrieve the class map CSV file path and read it
class_map_path = model.class_map_path().numpy()
class_map_csv_text = tf.io.read_file(class_map_path).numpy().decode('utf-8')
class_names = class_names_from_csv(class_map_csv_text)

# Process each chunk and print classification
for idx, chunk in enumerate(chunks):
    try:
        # Ensure the waveform chunk is 1D
        chunk = np.array(chunk, dtype=np.float32)

        # Convert the chunk to a tensor
        chunk_tensor = tf.convert_to_tensor(chunk, dtype=tf.float32)

        # Run the model with the chunk tensor
        scores, embeddings, log_mel_spectrogram = model(chunk_tensor)

        # Check output shapes (for debugging purposes)
        scores.shape.assert_is_compatible_with([None, 521])
        embeddings.shape.assert_is_compatible_with([None, 1024])
        log_mel_spectrogram.shape.assert_is_compatible_with([None, 64])

        # Find the class with the highest average score
        predicted_class_index = scores.numpy().mean(axis=0).argmax()
        predicted_class_name = class_names[predicted_class_index]

        print(f'Chunk {idx + 1} ({len(chunk)/sample_rate:.2f}s): Predicted class: {predicted_class_name}')

    except Exception as e:
        print(f'Error during model inference for chunk {idx + 1}: {e}')
