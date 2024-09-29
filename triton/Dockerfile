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

# Install Python packages for tokenizer and other dependencies
RUN pip3 install --upgrade pip && \
    pip3 install resemble-enhance==0.0.2.dev240104122303 deepspeed~=0.12.1 torchaudio~=2.1.1 git+https://github.com/oza75/coqui-TTS.git@prod && \
    pip3 install transformers[torch] numpy sentencepiece faster-whisper soundfile && \
    pip3 uninstall -y onnxruntime && \
    pip3 install optimum[onnxruntime-gpu]

RUN  python3 -m pip install wheel && python3 -m pip install --upgrade tensorrt

RUN mkdir -p "/trt_caches/nllb-600M-mt-french-bambara-trt-caches"

# Copy your model repository into the container
COPY ./model_repository /models

RUN apt-get update -y && apt install -y libcudnn8
#RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
#    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
#    apt-get update -y && apt-get install google-cloud-cli -y

# Set environment variable for Triton model repository
ENV TRITON_MODEL_REPOSITORY_PATH=/models
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/lib/python3.10/dist-packages/tensorrt_bindings:/usr/local/lib/python3.10/dist-packages/tensorrt_libs:/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}"

EXPOSE 8000
EXPOSE 8001
EXPOSE 8002

# Start the Triton server with the model repository path
CMD ["tritonserver", "--model-repository=/models", "--exit-timeout-secs=5"]
