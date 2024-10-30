# Use a base image with Python installed
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Install GitHub repository dependency
RUN pip install git+https://github.com/DeepView-Analytics/schemas

# Copy the entire project into the container
COPY . .

# Expose any necessary ports (e.g., if your app listens on a port)
EXPOSE 8080

# Run the FastAPI application using uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $FASTAPI_PORT"]