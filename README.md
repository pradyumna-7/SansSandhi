# SansSandhi
This repository provides a character-level model for Sanskrit text classification, predicting whether each character in a word is part of a Sandhi point (SP) or not (NSP). It includes preprocessing, training with an MLP model, and APIs (FastAPI) and a Telegram bot for easy predictions.

# Sanskrit Sandhi Point Prediction Bot

This project provides a character-level classification model for Sanskrit text. It predicts whether each character in a word is part of a Sandhi Point (SP) or not (NSP). The system includes an API for model interaction and a Telegram bot for user-friendly predictions.

## Features
- **Character-level classification**: Predicts if each character in a Sanskrit word is part of a Sandhi point.
- **Preprocessing pipeline**: Tokenizes and pads Sanskrit words for input to the model.
- **API integration**: Provides a FastAPI server for model interaction.
- **Telegram bot**: Allows users to interact with the model easily by sending Sanskrit words.

## Setup Instructions

### Prerequisites

- Python 3.x
- Telegram bot token (Create a bot on Telegram via [BotFather](https://core.telegram.org/bots#botfather))
- Install dependencies via `pip` (detailed below).

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/sanskrit-sandhi-prediction.git
   cd sanskrit-sandhi-prediction```


2. **Set up a virtual environment (optional but recommended):**
    ```bash
    Copypython -m venv venv
    source venv/bin/activate  # For Linux/Mac
    venv\Scripts\activate     # For Windows

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt

4. **Set up the environment:**

 - Create a .env file in the root directory of the project
 - Add your Telegram bot token in the .env file:
    ```bash 
        TELEGRAM_BOT_TOKEN=your_bot_token_here




### Running the Model Locally

1. **Train the model:**

Run the following command to train the model:
    ```bash
    python SansSandhi.py

This will process the dataset, train the Multi-Layer Perceptron (MLP) model, and save it as sanskrit_model.pkl


2. **Start the FastAPI server:**

Start the FastAPI API to serve predictions:
    ```bash
    Copyuvicorn app:app --reload

The API will be available at http://127.0.0.1:8000



### Running the Telegram Bot

1. **Start the bot:**

Run the bot script:
    ```bash
    python bot.py

The bot will be active and respond to messages sent to it


2. **Interacting with the bot:**

Send a Sanskrit word to the bot
The bot will split the word by characters and predict whether each character is part of a Sandhi point (SP) or not (NSP)



### Example Usage

**Telegram Bot:**

Send a word like यॊयस्माज्जायते to the bot
The bot will return the predictions for each character: either SP or NSP



### How It Works

1. **Data Preprocessing:**

The dataset is preprocessed to tokenize Sanskrit words and split them into sequences of characters
The tokenizer is trained on a predefined set of characters and used to convert words into sequences of indices


2. **Model Training:**

A Multi-Layer Perceptron (MLP) model is built to classify each character in a word as part of a Sandhi point (SP) or not (NSP)
The model is trained on tokenized and padded sequences of words


3. **Prediction:**

The trained model is used to predict the label (SP or NSP) for each character in a given Sanskrit word
The FastAPI server and Telegram bot provide interfaces for users to interact with the model



### Folder Structure
    ```bash
    sanskrit-sandhi-prediction/
    ├── app.py             # FastAPI API to serve the model
    ├── bot.py             # Telegram bot script
    ├── train.py           # Script to train the model
    ├── sanskrit_model.pkl # Trained model file
    ├── requirements.txt   # Python dependencies
    ├── .env              # Store your Telegram bot token here
    └── data/
        └── dataset.txt   # Training dataset


