# import os
# import sys
# import argparse
# from random import shuffle
# from collections import OrderedDict

import joblib
# import chardet
# import gensim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Embedding
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer

class PreProcessing():
    def __init__(self, vocab_size=1000, max_length=50):
        self.tokenizer = Tokenizer(char_level=True, oov_token="[UNK]")
        self.vocab_size = vocab_size
        self.max_length = max_length

    def load_data(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [line.strip() for line in f.readlines()]
        return data

    def fit_tokenizer(self, texts):
        """Fit the tokenizer on the text data."""
        self.tokenizer.fit_on_texts(texts)
        print(f"Tokenizer fitted with vocab size: {len(self.tokenizer.word_index)}")

    def tokenize_and_pad(self, texts):
        """Tokenize and pad sequences."""
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded_sequences = self.pad_sequences(sequences, maxlen=self.max_length, padding_value=0)
        print(f"Tokenized and padded {len(texts)} sequences. Example: {padded_sequences[:1]}")
        return padded_sequences

    def label2vec(self, label_seq):
        label_map = {'NSP': 0, 'SP': 1}
        return label_map[label_seq[-1]]

    def word2vec(self, word_seq):
        """Convert a word sequence into indices using the tokenizer."""
        sequence = self.tokenizer.texts_to_sequences(["".join(word_seq)])
        print(f"Converted '{word_seq}' to indices: {sequence[0]}")
        return sequence[0]

    def pad_sequences(self, sequences, maxlen, padding_value=0):
        """Pad sequences to a uniform length."""
        padded = []
        for seq in sequences:
            if len(seq) < maxlen:
                padded.append(seq + [padding_value] * (maxlen - len(seq)))
            else:
                padded.append(seq[:maxlen])
        print(f"Padded sequences to maxlen={maxlen}")
        return np.array(padded)

    def pre_process(self, file_path):
        print("Preprocessing the dataset...")
        words = []
        labels = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        cleaned_lines = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                print(f"Skipping empty line {i}")
                continue
            if "\t" not in line:
                split_index = min(line.find("NSP"), line.find("SP"))
                if split_index > 0:
                    word_seq = line[:split_index].strip()
                    label_seq = line[split_index:].strip()
                    cleaned_lines.append(f"{word_seq}\t{label_seq}")
                    print(f"Fixed line {i} by manual splitting.")
                else:
                    print(f"Skipping malformed line {i}: {line}")
                    continue
            else:
                cleaned_lines.append(line)

        for line in cleaned_lines:
            try:
                word_seq, label_seq = line.split("\t")
                words.append(word_seq.split())
                labels.append(label_seq.split())
            except ValueError:
                print(f"Error unpacking line: {line}")
                continue

        # Fit tokenizer on all words
        flat_words = ["".join(seq) for seq in words]
        self.fit_tokenizer(flat_words)

        X = [self.word2vec(word_seq) for word_seq in words]
        X = self.pad_sequences(X, maxlen=self.max_length)
        y = np.array([self.label2vec(label_seq) for label_seq in labels])
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: X ({len(X)}) and y ({len(y)})")

        split1 = int(0.8 * len(X))
        split2 = int(0.9 * len(X))
        X_train, y_train = X[:split1], y[:split1]
        X_valid, y_valid = X[split1:split2], y[split1:split2]
        X_test, y_test = X[split2:], y[split2:]

        print(f"Data split: Train={len(X_train)}, Valid={len(X_valid)}, Test={len(X_test)}")
        print(f"Max index in training data: {np.max(X_train)}, Vocab size: {len(self.tokenizer.word_index) + 1}")
        return X_train, y_train, X_valid, y_valid, X_test, y_test

class MultiLayerPerceptron():
    def __init__(self, vocab_size, max_length):
        self.vocab_size = vocab_size
        self.max_length = max_length

    def mlp_model(self):
        print(f"Building model with vocab_size={self.vocab_size}, max_length={self.max_length}")
        model = Sequential()
        model.add(Embedding(input_dim=self.vocab_size, output_dim=50, input_length=self.max_length))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(100, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(2, activation='softmax'))
        return model

    def run_mlp(self, X_train, y_train, X_valid, y_valid):
        model = self.mlp_model()
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print("Starting model training...")
        history = model.fit(X_train, y_train, batch_size=100, epochs=100, validation_data=(X_valid, y_valid))
        print("Model training completed.")
        joblib.dump(model, 'sanskrit_model.pkl')
        return model


class Evaluation():
    def evaluate(self, model, X_test, y_test):
        # Predict probabilities or logits
        y_pred = model.predict(X_test)

        # Convert predictions to binary labels (if needed)
        y_pred = np.argmax(y_pred, axis=-1)  # Assumes last dimension holds probabilities or logits

        # Debug: Print shapes of y_test and y_pred
        print(f"Shape of y_test: {y_test.shape}")
        print(f"Shape of y_pred: {y_pred.shape}")

        # Ensure shapes are consistent
        if y_pred.shape[0] == 1:  # Check if batch size is 1
            y_pred = y_pred.squeeze(0)  # Remove the extra dimension

        # Flatten the arrays for comparison
        y_test_flat = y_test.flatten()
        y_pred_flat = y_pred.flatten()

        # Debug: Print shapes after flattening
        print(f"Shape of flattened y_test: {y_test_flat.shape}")
        print(f"Shape of flattened y_pred: {y_pred_flat.shape}")

        # Calculate accuracy
        accuracy = accuracy_score(y_test_flat, y_pred_flat)
        print(f"Accuracy: {accuracy:.4f}")

    def predict_sequence(self, model, input_word, tokenizer, max_length, pp):
        print(f"Predicting sequence for input: '{input_word}'")

        # Initialize a list to store the predicted labels for each character
        labels = []

        # Process each character in the input word
        for char in input_word:
            # Convert each character to a sequence of tokens
            sequence = tokenizer.texts_to_sequences([char])

            # Pad the sequence to the required length
            padded_sequence = pp.pad_sequences(sequence, maxlen=max_length)

            # Predict the label (SP or NSP) for this character
            prediction = model.predict(padded_sequence)

            # Assign label based on the model's output
            label = "SP" if prediction[0][0] > prediction[0][1] else "NSP"

            # Append the predicted label for the character
            labels.append(label)

        print(f"Predicted labels for characters: {labels}")
        return labels



def main():
    pp = PreProcessing()
    X_train, y_train, X_valid, y_valid, X_test, y_test = pp.pre_process('./data/dataset.txt')
    vocab_size = len(pp.tokenizer.word_index) + 1
    max_length = pp.max_length
    mlp = MultiLayerPerceptron(vocab_size, max_length)
    model = mlp.run_mlp(X_train, y_train, X_valid, y_valid)
    evaluation = Evaluation()
    evaluation.evaluate(model, X_test, y_test)

    input_word = "यॊयस्माज्जायते"  # Example word for prediction
    prediction = evaluation.predict_sequence(model, input_word, pp.tokenizer, max_length, pp)
    print("Prediction:", prediction)

if __name__ == '__main__':
    main()