# Use Python latest as the base image
FROM python:latest

# Set working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The current directory will be mounted at runtime, 
# so we don't need to COPY the code into the image

# Set the default command to run Python scripts
ENTRYPOINT ["python3"]
