# Base image
FROM python:3.7-slim-buster

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Run the Flask app
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "5000"]