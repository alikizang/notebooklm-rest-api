FROM python:3.11-slim

# Install system dependencies, including FFmpeg and build libraries for Pillow
RUN apt-get update && apt-get install -y \
    ffmpeg \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependencies and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium browser & required system libraries
RUN playwright install chromium
RUN playwright install-deps chromium

# Copy application files
COPY . .

# Expose the API port
EXPOSE 8000

# Start Uvicorn FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
