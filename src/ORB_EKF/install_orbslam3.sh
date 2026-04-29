#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENDOR_DIR="${ROOT_DIR}/vendor"
PANGOLIN_SRC_DIR="${VENDOR_DIR}/Pangolin"
PANGOLIN_INSTALL_DIR="${VENDOR_DIR}/pangolin_install"
ORB_DIR="${VENDOR_DIR}/ORB_SLAM3"
PATCH_FILE="${ROOT_DIR}/orbslam3_ubuntu24.patch"

echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git pkg-config \
  libeigen3-dev libopencv-dev \
  libglew-dev libgl1-mesa-dev libegl1-mesa-dev \
  libwayland-dev libxkbcommon-dev wayland-protocols \
  libpython3-dev python3-dev python3-numpy \
  libgtk-3-dev ffmpeg \
  libavcodec-dev libavutil-dev libavformat-dev \
  libswscale-dev libavdevice-dev \
  libjpeg-dev libpng-dev libtiff5-dev libopenexr-dev

mkdir -p "${VENDOR_DIR}"

if [[ ! -d "${PANGOLIN_SRC_DIR}" ]]; then
  echo "Cloning Pangolin..."
  git clone https://github.com/stevenlovegrove/Pangolin.git "${PANGOLIN_SRC_DIR}"
fi

echo "Building Pangolin..."
cmake -B "${PANGOLIN_SRC_DIR}/build" -S "${PANGOLIN_SRC_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${PANGOLIN_INSTALL_DIR}" \
  -DBUILD_PANGOLIN_PYTHON=OFF
cmake --build "${PANGOLIN_SRC_DIR}/build" -j"$(nproc)"
cmake --install "${PANGOLIN_SRC_DIR}/build"

if [[ ! -d "${ORB_DIR}" ]]; then
  echo "Cloning ORB_SLAM3..."
  git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git "${ORB_DIR}"
fi

echo "Applying ORB-SLAM3 compatibility patch..."
if git -C "${ORB_DIR}" apply --check "${PATCH_FILE}" 2>/dev/null; then
  git -C "${ORB_DIR}" apply "${PATCH_FILE}"
  echo "Patch applied."
elif git -C "${ORB_DIR}" apply --reverse --check "${PATCH_FILE}" 2>/dev/null; then
  echo "Patch already applied."
else
  echo "Patch could not be applied cleanly."
  echo "If ORB-SLAM3 is on a different commit than expected, inspect ${PATCH_FILE} and patch manually."
  exit 1
fi

echo "Building DBoW2 (ORB-SLAM3 third-party)..."
cmake -B "${ORB_DIR}/Thirdparty/DBoW2/build" -S "${ORB_DIR}/Thirdparty/DBoW2" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${ORB_DIR}/Thirdparty/DBoW2/build" -j"$(nproc)"

echo "Building g2o (ORB-SLAM3 third-party)..."
cmake -B "${ORB_DIR}/Thirdparty/g2o/build" -S "${ORB_DIR}/Thirdparty/g2o" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${ORB_DIR}/Thirdparty/g2o/build" -j"$(nproc)"

echo "Building ORB-SLAM3 core library..."
cmake -B "${ORB_DIR}/build" -S "${ORB_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${PANGOLIN_INSTALL_DIR}" \
  -DCMAKE_CXX_FLAGS="-Wno-error=maybe-uninitialized -Wno-error=array-bounds -Wno-error=stringop-overflow"
cmake --build "${ORB_DIR}/build" -j"$(nproc)" --target ORB_SLAM3

echo "Extracting ORB vocabulary..."
tar -xf "${ORB_DIR}/Vocabulary/ORBvoc.txt.tar.gz" -C "${ORB_DIR}/Vocabulary/"

echo "Building Python bridge..."
"${ROOT_DIR}/build_orbslam3_backend.sh"

echo
echo "Install complete (standalone under ORB_EKF)."
echo "Vocabulary file:"
echo "  ${ORB_DIR}/Vocabulary/ORBvoc.txt"
echo
echo "Backend library:"
echo "  ${ROOT_DIR}/orbslam3_backend$(python3-config --extension-suffix)"
