FROM python:3.12-slim

# Avoid buffering in logs
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependendies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the project
COPY . .

# Default port for Cloud Run
ENV PORT=8080

# Run API with uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]

