#!/bin/bash
# ============================================================
# setup_native_jetson.sh — Native (no-Docker) bring-up for the
# Duckiebot's Jetson Nano.
#
# RUN THIS ON THE ROBOT (ssh duckie@duckiexp.local), not on your Mac.
# Confirmed on this bot: Ubuntu 18.04.6 (Bionic), Python 3.6.9, aarch64,
# nvidia-l4t-core 32.7.6, RAM 3.9GB total.
#
# Why native at all: the Docker path has been fighting Python-3.8-vs-3.6
# wheel mismatches and stripped CUDA/TensorRT libs for several commits.
# Ubuntu 18.04 + Python 3.6 is the officially supported combo for BOTH
# ROS Melodic (plain apt, no robostack workaround needed) and NVIDIA's
# official Jetson PyTorch/TensorRT wheels -- native sidesteps the whole
# mismatch.
#
# Why TensorRT+PyCUDA, not raw PyTorch, at runtime: NVIDIA's own forum
# guidance for 4GB Nanos is that PyTorch's CUDA kernel loading alone can
# spike RAM+swap by ~1.8GB the first time you call .cuda(). A raw
# TensorRT .engine loaded via plain `tensorrt` + `pycuda` (no torch
# import at inference time) avoids that entirely. See export_trt.py.
# PyTorch is therefore NOT installed by this script -- it isn't needed
# on-device at all, since export_trt.py builds .engine files straight
# from the .onnx files you already have (via trtexec), no torch required.
#
# Usage (run each stage, reviewing output before moving to the next):
#   bash setup_native_jetson.sh preflight
#   bash setup_native_jetson.sh install-ros
#   bash setup_native_jetson.sh install-pycuda
#   bash setup_native_jetson.sh build-workspace
# ============================================================

set -e
STAGE="${1:-}"

REPO_NAME=autonomousPipeline
WS_DIR="$HOME/native_ws"

banner() { echo; echo "======================================================="; echo " $1"; echo "======================================================="; }

# ------------------------------------------------------------
preflight() {
    banner "PREFLIGHT (read-only, no changes made)"

    echo "--- OS ---"; cat /etc/os-release | grep PRETTY_NAME
    echo "--- Python ---"; python3 --version
    echo "--- Disk ---"; df -h / /data 2>/dev/null
    echo "--- Memory ---"; free -h

    echo
    echo "--- CUDA / cuDNN / TensorRT apt packages ---"
    dpkg -l 2>/dev/null | grep -E 'cuda-toolkit|libcudnn|tensorrt|nvidia-jetpack|libnvinfer' || echo "  (none found -- likely need: sudo apt install nvidia-jetpack)"

    echo
    echo "--- trtexec ---"
    if command -v trtexec >/dev/null 2>&1; then
        echo "  found: $(command -v trtexec)"
    elif [ -x /usr/src/tensorrt/bin/trtexec ]; then
        echo "  found: /usr/src/tensorrt/bin/trtexec (not on PATH)"
    else
        echo "  NOT FOUND -- TensorRT toolkit may be incomplete"
    fi

    echo
    echo "--- python3 tensorrt / pycuda bindings ---"
    python3 -c "import tensorrt; print('  tensorrt OK, version', tensorrt.__version__)" 2>&1 | tail -1
    python3 -c "import pycuda; print('  pycuda OK')" 2>&1 | tail -1

    echo
    echo "--- ROS ---"
    if command -v roscore >/dev/null 2>&1; then
        echo "  roscore found: $(command -v roscore)"
    else
        echo "  no native ROS install found (expected -- everything currently runs in Docker)"
    fi

    echo
    echo "--- Existing Duckietown docker containers (host network mode?) ---"
    docker ps --format '{{.Names}}' 2>/dev/null | while read -r n; do
        mode=$(docker inspect -f '{{.HostConfig.NetworkMode}}' "$n" 2>/dev/null)
        echo "  $n : NetworkMode=$mode"
    done
    echo "  (native ROS nodes need NetworkMode=host on whichever container runs roscore"
    echo "   to see the same ROS master on localhost -- check above before assuming this works)"

    echo
    echo "Review the output above. If CUDA/cuDNN/TensorRT apt packages are missing, run:"
    echo "  sudo apt update && sudo apt install nvidia-jetpack"
    echo "before continuing to install-pycuda."
}

# ------------------------------------------------------------
install_ros() {
    banner "INSTALL ROS MELODIC (native, apt-based)"
    sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
    sudo apt update
    # tf-conversions: duckietown_msgs' CMakeLists.txt find_package()s it directly
    sudo apt install -y ros-melodic-ros-base ros-melodic-cv-bridge ros-melodic-tf-conversions python3-rosdep python3-catkin-tools python3-pip
    if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
        sudo rosdep init
    fi
    rosdep update
    echo "source /opt/ros/melodic/setup.bash" >> "$HOME/.bashrc"
    echo "ROS Melodic installed. Open a new shell (or 'source /opt/ros/melodic/setup.bash') to pick it up."
}

# ------------------------------------------------------------
install_pycuda() {
    banner "INSTALL PYCUDA (python3 bindings for the TensorRT runtime path)"
    sudo apt install -y python3-dev libboost-python-dev build-essential

    if ! command -v nvcc >/dev/null 2>&1; then
        for cudadir in /usr/local/cuda /usr/local/cuda-10.2; do
            if [ -d "$cudadir" ]; then
                echo "export PATH=$cudadir/bin:\$PATH" >> "$HOME/.bashrc"
                echo "export LD_LIBRARY_PATH=$cudadir/lib64:\$LD_LIBRARY_PATH" >> "$HOME/.bashrc"
                export PATH="$cudadir/bin:$PATH"
                export LD_LIBRARY_PATH="$cudadir/lib64:$LD_LIBRARY_PATH"
                echo "Added $cudadir to PATH/LD_LIBRARY_PATH (persisted in ~/.bashrc)"
                break
            fi
        done
    fi
    if ! command -v nvcc >/dev/null 2>&1; then
        echo "ERROR: nvcc still not found. CUDA toolkit isn't installed -- run 'sudo apt install nvidia-jetpack' first (see preflight)."
        exit 1
    fi

    pip3 install --user pycuda
    python3 -c "import pycuda.driver as cuda; cuda.init(); print('pycuda OK,', cuda.Device.count(), 'CUDA device(s)')"

    echo
    echo "--- python3 tensorrt bindings (should already exist system-wide from JetPack) ---"
    python3 -c "import tensorrt; print('tensorrt OK, version', tensorrt.__version__)" || \
        echo "MISSING: install with 'sudo apt install python3-libnvinfer python3-libnvinfer-dev'"
}

# ------------------------------------------------------------
build_workspace() {
    banner "BUILD NATIVE CATKIN WORKSPACE"
    source /opt/ros/melodic/setup.bash

    mkdir -p "$WS_DIR/src"
    cd "$WS_DIR/src"

    if [ ! -d duckietown_msgs ]; then
        git clone --depth 1 -b daffy https://github.com/duckietown/dt-ros-commons.git /tmp/dt-ros-commons
        mv /tmp/dt-ros-commons/packages/duckietown_msgs .
        rm -rf /tmp/dt-ros-commons
    fi

    if [ ! -L cond_imitation_learning_pkg ]; then
        if [ ! -d "$HOME/dev/$REPO_NAME" ]; then
            echo "ERROR: expected the repo synced to \$HOME/dev/$REPO_NAME on this robot."
            echo "From your Mac, run: rsync -av --exclude models --exclude .git ./ duckie@duckiexp.local:~/dev/$REPO_NAME/"
            exit 1
        fi
        ln -s "$HOME/dev/$REPO_NAME/packages/cond_imitation_learning_pkg" cond_imitation_learning_pkg
    fi

    echo "--- Python deps for cond_imitation_learning_pkg (no torch/torchvision needed for TensorRT backend) ---"
    # Installed via apt, not pip: this box's pip3 is too old to resolve prebuilt aarch64
    # wheels for current PyPI releases (falls back to building opencv-python-headless from
    # source, which needs scikit-build and is impractical on a Nano). apt's python3-opencv
    # also avoids a second OpenCV build living alongside the one ros-melodic-cv-bridge already
    # provides. The inference code decodes JPEGs directly with cv2, not cv_bridge.
    sudo apt install -y python3-opencv python3-pil python3-requests python3-numpy

    cd "$WS_DIR"
    catkin build

    echo
    echo "Workspace built at $WS_DIR. Source it with:"
    echo "  source $WS_DIR/devel/setup.bash"
}

# ------------------------------------------------------------
case "$STAGE" in
    preflight) preflight ;;
    install-ros) install_ros ;;
    install-pycuda) install_pycuda ;;
    build-workspace) build_workspace ;;
    *)
        echo "Usage: bash setup_native_jetson.sh {preflight|install-ros|install-pycuda|build-workspace}"
        exit 1
        ;;
esac
