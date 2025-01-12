# Use the base image
FROM python:3.11.3-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Upgrade pip to avoid compatibility issues
RUN pip install --upgrade pip

# Install Git and clone the repository
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6  \
    && git clone https://github.com/DeepView-Analytics/schemas.git /schemas \
    && pip install /schemas


# Copy the requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


# Copy the entire project into the container
COPY . .

# Expose the application port
EXPOSE 8000
EXPOSE 9092
EXPOSE 6379

# Run the FastAPI application using uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000"]
