FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PORT=8080

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and app
COPY kolam_r/ ./kolam_r/
COPY app/ ./app/
COPY .streamlit/ ./.streamlit/
COPY README.md .

EXPOSE 8080

CMD ["streamlit", "run", "app/prototype_app.py", "--server.port=8080", "--server.address=0.0.0.0"]
