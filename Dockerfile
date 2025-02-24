# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy files to container
COPY requirements.txt .
COPY app.py .
COPY bot.py .
COPY sanskrit_model.pkl .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose API port
EXPOSE 8000

# Start both FastAPI and the bot
CMD uvicorn app:app --host 0.0.0.0 --port 8000 & python bot.py
