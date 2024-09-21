# Use the Triton server base image
FROM nvcr.io/nvidia/tritonserver:24.08-py3

# Install necessary dependencies for building CTranslate2 and Triton backend
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libboost-all-dev \
    python3-pip \
    rapidjson-dev \
    wget \
    gnupg \
    gpg-agent \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB | gpg --dearmor | tee /usr/share/keyrings/oneapi-archive-keyring.gpg > /dev/null
RUN echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" | tee /etc/apt/sources.list.d/oneAPI.list

# Install Intel MKL
RUN apt-get update && apt-get install -y intel-basekit

# Clone and install CTranslate2 library (with submodules)
RUN git clone --recursive https://github.com/OpenNMT/CTranslate2 /CTranslate2 && \
    cd /CTranslate2 && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Clone the CTranslate2 Triton backend repository
RUN git clone https://github.com/speechmatics/ctranslate2_triton_backend /ctranslate2_triton_backend

# Build and install the CTranslate2 Triton backend
RUN cd /ctranslate2_triton_backend && mkdir build && cd build && cmake .. \
    -DCMAKE_INSTALL_PREFIX:PATH=/opt/tritonserver \
    -DTRITON_ENABLE_GPU=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRITON_COMMON_REPO_TAG=r24.08 \
    -DTRITON_CORE_REPO_TAG=r24.08 \
    -DTRITON_BACKEND_REPO_TAG=r24.08 \
    -Dctranslate2_DIR=/usr/local/share/cmake/CTranslate2 \
    && make -j$(nproc) install

# Install Python packages for tokenizer and other dependencies
RUN pip3 install --upgrade pip && \
    pip3 install transformers numpy sentencepiece

# Copy your model repository into the container
COPY ./model_repository /models

# Copy any additional files like tokenizers if needed
#COPY ./nllb-600M-tokenizer /nllb-600M-tokenizer
COPY ./m2m100 /nllb-600M-tokenizer

# Set environment variable for Triton model repository
ENV TRITON_MODEL_REPOSITORY_PATH=/models
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/tritonserver/backends/ctranslate2:/opt/intel/oneapi/2024.2/lib/

# Start the Triton server with the model repository path
CMD ["tritonserver", "--model-repository=/models"]
