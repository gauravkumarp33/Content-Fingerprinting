FROM python:3.11-slim

WORKDIR /app



# Install Python dependencies
COPY requirements.txt .
# Install CPU-only torch to save massive amounts of space
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./app /app/app

# Expose the port Hugging Face Spaces runs on
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
