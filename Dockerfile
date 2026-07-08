# parameters
ARG REPO_NAME="autonomousPipeline"
ARG DESCRIPTION="Autonomous Pipeline for lane following data collection with a GUI interface for controlling the process and storing data."
ARG MAINTAINER="Pankaj Bora borapankaj901@gmail.com"
# pick an icon from: https://fontawesome.com/v4.7.0/icons/
ARG ICON="cube"

# ==================================================>
# ==> Do not change the code below this line
ARG ARCH
ARG DISTRO=ente
ARG DOCKER_REGISTRY=docker.io
ARG BASE_IMAGE=dt-machine-learning-base-pytorch
ARG BASE_TAG=${DISTRO}-${ARCH}
ARG LAUNCHER=default

# define base image
FROM ${DOCKER_REGISTRY}/duckietown/${BASE_IMAGE}:${BASE_TAG} as base

# Rescue NVIDIA and CUDA libraries from the official L4T container since Duckietown OS strips them
FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3 AS cuda-rescue

FROM base

# recall all arguments
ARG DISTRO
ARG REPO_NAME
ARG DESCRIPTION
ARG MAINTAINER
ARG ICON
ARG BASE_TAG
ARG BASE_IMAGE
ARG LAUNCHER
# - buildkit
ARG TARGETPLATFORM
ARG TARGETOS
ARG TARGETARCH
ARG TARGETVARIANT

# check build arguments
RUN dt-build-env-check "${REPO_NAME}" "${MAINTAINER}" "${DESCRIPTION}"

# define/create repository path
ARG REPO_PATH="${CATKIN_WS_DIR}/src/${REPO_NAME}"
ARG LAUNCH_PATH="${LAUNCH_DIR}/${REPO_NAME}"
RUN mkdir -p "${REPO_PATH}" "${LAUNCH_PATH}"
WORKDIR "${REPO_PATH}"

# keep some arguments as environment variables
ENV DT_MODULE_TYPE="${REPO_NAME}" \
    DT_MODULE_DESCRIPTION="${DESCRIPTION}" \
    DT_MODULE_ICON="${ICON}" \
    DT_MAINTAINER="${MAINTAINER}" \
    DT_REPO_PATH="${REPO_PATH}" \
    DT_LAUNCH_PATH="${LAUNCH_PATH}" \
    DT_LAUNCHER="${LAUNCHER}"

# install apt dependencies
COPY ./dependencies-apt.txt "${REPO_PATH}/"
RUN apt-get update && apt-get install -y curl gnupg2 && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN dt-apt-install ${REPO_PATH}/dependencies-apt.txt

# Rescue stripped CUDA/TensorRT/Driver libraries from the official NVIDIA container
RUN mkdir -p /opt/nvidia-rescue/tegra
COPY --from=cuda-rescue /usr/local/cuda-10.2 /opt/nvidia-rescue/cuda-10.2
COPY --from=cuda-rescue /usr/lib/aarch64-linux-gnu/libcudnn* /opt/nvidia-rescue/
COPY --from=cuda-rescue /usr/lib/aarch64-linux-gnu/libcublas* /opt/nvidia-rescue/
COPY --from=cuda-rescue /usr/lib/aarch64-linux-gnu/libnvinfer* /opt/nvidia-rescue/
COPY --from=cuda-rescue /usr/lib/aarch64-linux-gnu/libnvparsers* /opt/nvidia-rescue/
COPY --from=cuda-rescue /usr/lib/aarch64-linux-gnu/libnvonnxparser* /opt/nvidia-rescue/
COPY --from=cuda-rescue /usr/lib/aarch64-linux-gnu/libmyelin* /opt/nvidia-rescue/
COPY --from=cuda-rescue /usr/lib/aarch64-linux-gnu/libnvblas* /opt/nvidia-rescue/
COPY --from=cuda-rescue /usr/lib/aarch64-linux-gnu/libcuda* /opt/nvidia-rescue/
COPY --from=cuda-rescue /usr/lib/aarch64-linux-gnu/tegra/* /opt/nvidia-rescue/tegra/

# install python3 dependencies
ARG PIP_INDEX_URL="https://pypi.org/simple"
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    LD_LIBRARY_PATH=/opt/nvidia-rescue:/opt/nvidia-rescue/tegra:/opt/nvidia-rescue/cuda-10.2/lib64:${LD_LIBRARY_PATH}

# install python dependencies
COPY ./dependencies-py3.* "${REPO_PATH}/"
RUN pip3 install -r "${REPO_PATH}/dependencies-py3.txt"

# Install ultralytics safely WITHOUT dependencies to avoid overwriting GPU torch wheels
RUN pip3 install --no-deps ultralytics

# copy the source code
COPY ./packages "${REPO_PATH}/packages"

# build packages
RUN . /opt/ros/${ROS_DISTRO}/setup.sh && \
  catkin build \
    --workspace ${CATKIN_WS_DIR}/

# install launcher scripts
COPY ./launchers/. "${LAUNCH_PATH}/"
RUN dt-install-launchers "${LAUNCH_PATH}"

# create desktop entry
RUN mkdir -p /root/Desktop
COPY assets/data_collector.desktop /root/Desktop/data_collector.desktop
RUN chmod +x /root/Desktop/data_collector.desktop

# define default command
CMD ["bash", "-c", "dt-launcher-${DT_LAUNCHER}"]

# store module metadata
LABEL org.duckietown.label.module.type="autonomousPipeline" \
    org.duckietown.label.module.description="Autonomous Pipeline for lane following data collection with a GUI interface for controlling the process and storing data." \
    org.duckietown.label.module.icon="${ICON}" \
    org.duckietown.label.platform.os="${TARGETOS}" \
    org.duckietown.label.platform.architecture="${TARGETARCH}" \
    org.duckietown.label.platform.variant="${TARGETVARIANT}" \
    org.duckietown.label.code.location="${REPO_PATH}" \
    org.duckietown.label.code.version.distro="${DISTRO}" \
    org.duckietown.label.base.image="${BASE_IMAGE}" \
    org.duckietown.label.base.tag="${BASE_TAG}" \
    org.duckietown.label.maintainer="${MAINTAINER}"
# <== Do not change the code above this line
# <==================================================
