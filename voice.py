import speech_recognition as sr
import torch
import torch.nn.functional as F
import numpy as np
import keyboard  # For detecting keypresses
import time
import threading
from deep_translator import GoogleTranslator  # More reliable translator

def softmax_convolution(audio_signal: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Smooth the audio signal using softmax convolution."""
    tensor_signal = torch.tensor(audio_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, kernel_size)) / kernel_size
    convolved = F.conv1d(tensor_signal, kernel, padding=kernel_size // 2)
    return F.softmax(convolved, dim=-1).squeeze().numpy()

def preprocess_audio(audio_data: np.ndarray) -> np.ndarray:
    """Normalize and clean audio data."""
    normalized_audio = audio_data / np.max(np.abs(audio_data))
    cleaned_audio = softmax_convolution(normalized_audio)
    return cleaned_audio

def detect_language(text: str) -> str:
    """Detect the language of the given text."""
    urdu_characters = 'ا ب پ ت ث ج چ ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ہ ی'
    if any(char in text for char in urdu_characters):
        return 'ur'
    return 'en'

def translate_text(text: str, target_lang: str = 'en') -> str:
    """Translate text using Deep Translator."""
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        print(f"Translation Error: {e}")
        return text  # Fallback: return original text if translation fails

def stop_recording(event):
    """Stop recording when 'e' key is pressed."""
    while not event.is_set():
        if keyboard.is_pressed('e'):
            event.set()
            print("Recording stopped.")
            break

def record_and_convert_to_text() -> None:
    """Record audio, recognize speech, and translate Urdu to English."""
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    stop_event = threading.Event()

    print("Press 's' to start recording and 'e' to end recording.")
    while True:
        if keyboard.is_pressed('s'):
            print("Recording... Press 'e' to stop.")
            break

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening...")

        # Start a thread to monitor the 'e' key press
        stop_thread = threading.Thread(target=stop_recording, args=(stop_event,))
        stop_thread.start()

        while not stop_event.is_set():
            try:
                audio_chunk = recognizer.listen(source, timeout=5)  # Listen for a short chunk of audio
                audio_np = np.frombuffer(audio_chunk.get_raw_data(), dtype=np.int16)
                preprocessed_audio = preprocess_audio(audio_np)

                # Recognize speech
                text = recognizer.recognize_google(audio_chunk, language="ur")
                print(f"Recognized (Urdu): {text}")

                # Detect language and translate if necessary
                lang = detect_language(text)
                if lang == 'ur':
                    translated_text = translate_text(text, target_lang='en')
                    print(f"Translated to English: {translated_text}")
                else:
                    print(f"Output: {text}")

            except sr.UnknownValueError:
                print("Could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
            except Exception as e:
                print(f"Error: {e}")

        stop_thread.join()  # Wait for the stop thread to finish

if __name__ == "__main__":
    record_and_convert_to_text()
s