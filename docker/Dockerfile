FROM nvidia/cuda:10.0-devel-ubuntu18.04

ENV PYTORCH_VERSION=1.1.0
ENV TORCHVISION_VERSION=0.3.0
ENV CUDNN_VERSION=7.6.0.64-1+cuda10.0
ENV NCCL_VERSION=2.4.7-1+cuda10.0
ENV TENSORFLOW_VERSION=1.14.0
ENV HOROVOD_VERSION=0.16.4
ENV TRT_VERSION=5.1.5.0

# Python 2.7 or 3.6 is supported by Ubuntu Bionic out of the box
ARG python=3.6
ENV PYTHON_VERSION=${python}
ENV DEBIAN_FRONTEND=noninteractive

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    build-essential \
    cmake \
    g++-4.8 \
    git \
    curl \
    docker.io \
    vim \
    wget \
    ca-certificates \
    libcudnn7=${CUDNN_VERSION} \
    libnccl2=${NCCL_VERSION} \
    libnccl-dev=${NCCL_VERSION} \
    libjpeg-dev \
    libpng-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-tk \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    libgtk2.0-dev \
    unzip \
    bzip2 \
    htop \
    gnuplot \
    ffmpeg 

# Install Open MPI
RUN mkdir /tmp/openmpi && \
    cd /tmp/openmpi && \
    wget https://www.open-mpi.org/software/ompi/v4.0/downloads/openmpi-4.0.0.tar.gz && \
    tar zxf openmpi-4.0.0.tar.gz && \
    cd openmpi-4.0.0 && \
    ./configure --enable-orterun-prefix-by-default && \
    make -j $(nproc) all && \
    make install && \
    ldconfig && \
    rm -rf /tmp/openmpi

# Install OpenSSH for MPI to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

# Instal Python and pip
RUN if [[ "${PYTHON_VERSION}" == "3.6" ]]; then \
    apt-get install -y python${PYTHON_VERSION}-distutils; \
    fi

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

# Install PyTorch
RUN pip install future typing numpy==1.19.5 awscli
RUN pip install https://download.pytorch.org/whl/cu100/torch-${PYTORCH_VERSION}-cp36-cp36m-linux_x86_64.whl
RUN pip install https://download.pytorch.org/whl/cu100/torchvision-${TORCHVISION_VERSION}-cp36-cp36m-linux_x86_64.whl

RUN pip install numpy==1.19.5 \
    tensorflow-gpu==${TENSORFLOW_VERSION} \
    keras \
    h5py
# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 \
    pip install --no-cache-dir horovod==${HOROVOD_VERSION} && \
    ldconfig

# Configure environment variables - default working directory is "/workspace"
WORKDIR /workspace
ENV PYTHONPATH="/workspace"

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Settings for S3
RUN pip install keras h5py pandas \
    torch==${PYTORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} && ldconfig
RUN pip install awscli wandb tensorboardx tqdm numpy-quaternion termcolor path.py pillow opencv-python-headless==4.5.1.48 matplotlib jupyter boto3 mpi4py onnx onnxruntime pycuda evo

RUN pip install cython==0.29.10

# kp3d copy
RUN mkdir -p /workspace/experiments
RUN mkdir -p /workspace/kp3d
WORKDIR /workspace/kp3d

# Build externals for Pose evaluation, and add volume for persistence
RUN mkdir -p /workspace/kp3d/kp3d/externals/cpp
COPY kp3d/externals/cpp /workspace/kp3d/kp3d/externals/cpp
ADD Makefile /workspace/kp3d/Makefile
RUN cd /workspace/kp3d/ && make build-externals
VOLUME ["/workspace/kp3d/kp3d/externals/cpp/"]

# Copy self-supervised learning source
COPY . /workspace/kp3d
ENV PYTHONPATH="/workspace/kp3d:$PYTHONPATH"
