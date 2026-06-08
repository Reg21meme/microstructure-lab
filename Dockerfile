FROM ubuntu:24.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy project files in
COPY pyproject.toml .
COPY python/ ./python/
COPY cpp/ ./cpp/

# Install Python dependencies
RUN pip3 install -e . --break-system-packages

CMD ["bash"]