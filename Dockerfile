# 1. Use an official, lightweight Python runtime as a parent image
FROM python:3.11-slim

# 2. Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Install FFmpeg (Crucial for Faster-Whisper audio extraction)
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 4. Set the working directory in the container
WORKDIR /app

# 5. Copy the requirements file into the container
COPY requirements.txt .

# 6. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the rest of the application code
COPY . .

# 8. Expose the port Streamlit uses
EXPOSE 8501

# 9. Healthcheck to ensure the container doesn't crash silently
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 10. Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
