# Base image
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# Set up environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# Define PYTHONPATH early to avoid undefined variable issues
ENV PYTHONPATH=/workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    openssh-client \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements file and install Python dependencies
COPY requirements.txt .
# Add verbose output and break into smaller chunks to better identify failures
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -v setuptools wheel && \
    pip install --no-cache-dir -v -r requirements.txt

# Install additional development tools
RUN pip install --no-cache-dir \
    black \
    isort \
    pytest \
    ipython

# Set up a non-root user for better security
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && mkdir -p /home/$USERNAME/.vscode-server /home/$USERNAME/.vscode-server-insiders \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME /workspace


# Set up permissions for the workspace
RUN mkdir -p /workspace/data \
    && chown -R $USERNAME:$USERNAME /workspace

# Switch to the non-root user
USER $USERNAME

# Update PYTHONPATH correctly (append to existing value)
ENV PYTHONPATH=/workspace:${PYTHONPATH}

# Keep the container running
CMD ["/bin/bash"]