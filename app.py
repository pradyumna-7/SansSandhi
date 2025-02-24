from fastapi import FastAPI
import joblib
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

app = FastAPI()

# Load the model
model = joblib.load("sanskrit_model.pkl")

# Load tokenizer (recreate from training phase)
tokenizer = Tokenizer(char_level=True, oov_token="[UNK]")
tokenizer.fit_on_texts(["abcdefghijklmnopqrstuvwxyzअआइईउऊऋएऐओऔकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह"])  # Refit on chars

MAX_LENGTH = 50  # Must match training


@app.get("/")
def home():
    return {"message": "Sanskrit Model API is running!"}


@app.post("/predict/")
def predict(word: str):
    # Split word into characters
    characters = list(word)

    # Prepare prediction result
    results = []

    # Make prediction for each character
    for char in characters:
        sequence = tokenizer.texts_to_sequences([char])
        padded_sequence = np.array([seq + [0] * (MAX_LENGTH - len(seq)) for seq in sequence])

        prediction = model.predict(padded_sequence)
        label = "SP" if prediction[0][0] > prediction[0][1] else "NSP"
        results.append({"character": char, "prediction": label})

    return {"word": word, "predictions": results}
