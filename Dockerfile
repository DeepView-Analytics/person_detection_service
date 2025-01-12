# Use the base image
FROM python:3.11.3-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Upgrade pip to avoid compatibility issues
RUN pip install --upgrade pip

# Install necessary system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libv4l-dev \
    libdc1394-22-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the repository from GitHub
RUN git clone https://github.com/DeepView-Analytics/schemas.git /schemas \
    && pip install /schemas

# Copy the requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Expose the application ports
EXPOSE 8000
EXPOSE 9092
EXPOSE 6379

# Run the FastAPI application using uvicorn
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000"]
