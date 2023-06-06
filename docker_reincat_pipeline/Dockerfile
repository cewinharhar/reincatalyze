FROM nvidia/cuda:12.1.1-base-ubuntu22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    wget \
    tar \
    bzip2 \
    build-essential \
    git

# Boost library 
WORKDIR /usr/local

# Download Boost library
RUN wget https://boostorg.jfrog.io/artifactory/main/release/1.82.0/source/boost_1_82_0.tar.bz2

# Extract Boost library
RUN tar --bzip2 -xf boost_1_82_0.tar.bz2

# Build and install Boost
WORKDIR /usr/local/boost_1_82_0
RUN sudo ./bootstrap.sh && sudo ./b2 install

# Set LD_LIBRARY_PATH
RUN echo 'export PATH="/usr/local/boost_1_82_0/stage/lib:$PATH"' >> ~/.bashrc

# Clone Vina-GPU-2.0 repository
WORKDIR /usr/local
RUN git clone https://github.com/DeltaGroupNJUPT/Vina-GPU-2.0

# Update makefile for Vina-GPU-2.0
WORKDIR /usr/local/Vina-GPU-2.0/Vina-GPU+
RUN sed -i 's|BOOST_LIB_PATH=.*|BOOST_LIB_PATH=/usr/local/boost_1_82_0|' makefile
RUN sed -i 's|OPENCL_LIB_PATH=.*|OPENCL_LIB_PATH=/usr/local/cuda|' makefile
RUN sed -i 's|OPENCL_VERSION=.*|OPENCL_VERSION=-DOPENCL_3_0|' makefile
RUN sed -i 's|GPU_PLATFORM=.*|GPU_PLATFORM=-DNVIDIA_PLATFORM|' makefile

# Clean and build Vina-GPU-2.0
RUN make source

# Set the working directory
WORKDIR /usr/local/Vina-GPU-2.0

# Run your commands here
CMD ["/bin/bash"]

WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

# 
RUN pip3 install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app /code/app

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--workers", "2"]

#use it with:
#docker build --platform linux/amd64 -t gaesp .
#docker run -d --gpus all --name XXX -p 80:80 gaesp