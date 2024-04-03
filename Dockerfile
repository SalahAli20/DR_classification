# Base image for deep learning tasks
FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

# Create a directory for the project
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies within the container
RUN apt-get update && apt-get install -y \
    python3-pip \
    pandas \
    numpy \
    torchvision \
    torch \
    scikit-learn \
    seaborn \
    matplotlib

# Install additional dependencies (if any)
RUN pip install -r requirements.txt 

# Expose port for TensorBoard or other visualization tools (optional)
EXPOSE 6006

# Command to run the main script
CMD ["python", "main.py"]
