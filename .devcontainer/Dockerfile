FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build arguments for the user
ARG USERNAME=filtr_user
ARG USER_UID=1000
ARG USER_GID=1000
ARG TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

ENV CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PATH="/usr/local/cuda/bin:/home/${USERNAME}/.local/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies, Python 3.8, and gcc-9 / g++-9
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    python3.8 python3.8-dev python3.8-distutils \
    python3-pip python3-venv \
    gcc-9 g++-9 \
    build-essential git curl wget vim ca-certificates \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3 maps to python3.8
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 \
    && update-alternatives --set python3 /usr/bin/python3.8

# Make gcc / g++ version 9 the default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 100 \
    && update-alternatives --set gcc /usr/bin/gcc-9 \
    && update-alternatives --set g++ /usr/bin/g++-9

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir torch==1.12.0+cu113 torchvision==0.13.0+cu113 \
    --find-links https://download.pytorch.org/whl/torch_stable.html

# Create a non-root user
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash $USERNAME \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}

RUN apt-get update && apt-get install -y sudo
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME

USER ${USERNAME}
WORKDIR /home/${USERNAME}/filtr

COPY --chown=${USERNAME}:${USERNAME} requirements.txt ./requirements.txt

RUN python3 -m pip install --no-cache-dir --user -r requirements.txt

CMD ["/bin/bash"]
