import logging
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# Replace with your bot token
TOKEN = "your_bot_token"
API_URL = "http://127.0.0.1:8000/predict/"  # Adjust based on deployment

logging.basicConfig(level=logging.INFO)

# Function for the /start command
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Send me a Sanskrit word, and I'll predict its label for each character.")

# Function to handle messages
async def handle_message(update: Update, context: CallbackContext):
    word = update.message.text
    response = requests.post(API_URL, params={"word": word})

    if response.status_code == 200:
        prediction_data = response.json().get("predictions", [])
        if prediction_data:
            result_text = "Predictions for each character:\n"
            for item in prediction_data:
                result_text += f"Character: {item['character']}, Prediction: {item['prediction']}\n"
            await update.message.reply_text(result_text)
        else:
            await update.message.reply_text("No predictions available. Something went wrong.")
    else:
        await update.message.reply_text("Something went wrong. Try again later.")

# Main function to run the bot
def main():
    application = Application.builder().token(TOKEN).build()

    # Adding handlers for the commands and messages
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start polling for messages
    application.run_polling()

if __name__ == "__main__":
    main()
