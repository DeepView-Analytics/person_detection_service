# Use the base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y git ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip to avoid compatibility issues
RUN pip install --upgrade pip

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install GitHub repository dependency
RUN pip install git+https://github.com/DeepView-Analytics/schemas

# Copy the entire project into the container
COPY . .

# Expose the application port
EXPOSE 8000

# Run the FastAPI application using uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $FASTAPI_PORT"]
