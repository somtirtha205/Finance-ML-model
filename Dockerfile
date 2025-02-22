# Use the official Python image from the Docker Hub
FROM python:3.12.7

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY docker/requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI app code
COPY docker/mlapp_api.py .

# Copy the model file
COPY model/ar.pkl .

# Expose the port FastAPI will run on
EXPOSE 8000

# Command to run the FastAPI app
CMD ["uvicorn", "mlapp_api:app", "--host", "0.0.0.0", "--port", "8000"]
