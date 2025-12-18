FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependency file first (layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Train model when container builds (optional but good)
RUN python model_train.py

# Expose Flask port
EXPOSE 5000

# Run Flask app
CMD ["python", "app.py"]
