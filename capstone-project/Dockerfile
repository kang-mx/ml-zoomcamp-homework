FROM python:3.10-slim

# Prevent Python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY scaler.pkl .
COPY lstm_rul_final.pth .

# Expose Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
