FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build arguments for the user
ARG USERNAME=filtr_user
ARG USER_UID=1000
ARG USER_GID=1000

# Install system dependencies, Python 3.8, and gcc-9 / g++-9
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    python3.8 python3.8-dev python3.8-distutils \
    python3-pip python3-venv \
    gcc-9 g++-9 \
    build-essential git curl wget vim ca-certificates \
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

# Create a non-root user
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash $USERNAME \
    && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}

RUN apt-get update && apt-get install -y sudo
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME

USER ${USERNAME}
WORKDIR /home/${USERNAME}

CMD ["/bin/bash"]