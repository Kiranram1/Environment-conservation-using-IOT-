import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import io
import sounddevice as sd
import time

# Load the YAMNet model
model = hub.load('https://tfhub.dev/google/yamnet/1')

# Parameters
SAMPLE_RATE = 16000
CHUNK_DURATION = 1  # Duration of each chunk in seconds
CHUNK_SIZE = CHUNK_DURATION * SAMPLE_RATE

# Target classes to monitor
target_classes = ["gunfire", "gunshot", "vehicle", "explosion"]

# Function to record audio from the microphone
def record_audio(duration, sample_rate=SAMPLE_RATE):
    print("Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    print("Recording complete.")
    return recording.flatten()

# Split the audio into chunks of specified duration (in seconds)
def split_audio_into_chunks(waveform, sample_rate, chunk_duration=10):
    chunk_size = chunk_duration * sample_rate
    return [waveform[i:i + chunk_size] for i in range(0, len(waveform), chunk_size)]

# Find the class names from the CSV file content
def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_map_csv = io.StringIO(class_map_csv_text)
    class_names = [display_name for (class_index, mid, display_name) in csv.reader(class_map_csv)]
    class_names = class_names[1:]  # Skip CSV header
    return class_names

# Retrieve the class map CSV file path and read it
class_map_path = model.class_map_path().numpy()
class_map_csv_text = tf.io.read_file(class_map_path).numpy().decode('utf-8')
class_names = class_names_from_csv(class_map_csv_text)

# Real-time audio classification
try:
    while True:
        # Record audio from the microphone
        waveform = record_audio(CHUNK_DURATION, SAMPLE_RATE)
        
        # Split the waveform into chunks (in case it's longer than the chunk duration)
        chunks = split_audio_into_chunks(waveform, SAMPLE_RATE)

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

                print(f'Chunk {idx + 1} ({len(chunk)/SAMPLE_RATE:.2f}s): Predicted class: {predicted_class_name}')

                # Check if the predicted class is one of the target classes
                if predicted_class_name.lower() in target_classes:
                    print(f"ALERT: Detected {predicted_class_name.upper()}!")

            except Exception as e:
                print(f'Error during model inference for chunk {idx + 1}: {e}')
        
        # Sleep briefly to avoid overloading the CPU
        time.sleep(1)

except KeyboardInterrupt:
    print("Real-time audio classification stopped.")
