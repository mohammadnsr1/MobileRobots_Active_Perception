#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORB_DIR="${ROOT_DIR}/vendor/ORB_SLAM3"
PANGOLIN_DIR="${ROOT_DIR}/vendor/pangolin_install"
PYBIND11_DIR="${ROOT_DIR}/vendor/Pangolin/components/pango_python/pybind11/include"
EXT_SUFFIX="$(python3-config --extension-suffix)"

PYTHON_INCLUDES="$(python3-config --includes)"
OPENCV_FLAGS="$(pkg-config --cflags --libs opencv4)"

g++ \
  -O3 \
  -Wall \
  -shared \
  -std=c++17 \
  -fPIC \
  ${PYTHON_INCLUDES} \
  -I/usr/include/eigen3 \
  -I"${PYBIND11_DIR}" \
  -I"${PANGOLIN_DIR}/include" \
  -I"${ORB_DIR}" \
  -I"${ORB_DIR}/include" \
  -I"${ORB_DIR}/include/CameraModels" \
  -I"${ORB_DIR}/Thirdparty/Sophus" \
  "${ROOT_DIR}/orbslam3_backend.cpp" \
  -o "${ROOT_DIR}/orbslam3_backend${EXT_SUFFIX}" \
  -L"${ORB_DIR}/lib" \
  -L"${ORB_DIR}/Thirdparty/DBoW2/lib" \
  -L"${ORB_DIR}/Thirdparty/g2o/lib" \
  -L"${PANGOLIN_DIR}/lib" \
  -Wl,--no-as-needed \
  -lORB_SLAM3 \
  -lDBoW2 \
  -lg2o \
  -lpango_display \
  -lpango_windowing \
  -lpango_image \
  -lpango_vars \
  -lpango_opengl \
  -lpango_core \
  -Wl,--as-needed \
  -Wl,-rpath,"${ORB_DIR}/lib" \
  -Wl,-rpath,"${ORB_DIR}/Thirdparty/DBoW2/lib" \
  -Wl,-rpath,"${ORB_DIR}/Thirdparty/g2o/lib" \
  -Wl,-rpath,"${PANGOLIN_DIR}/lib" \
  ${OPENCV_FLAGS}

echo "Built ${ROOT_DIR}/orbslam3_backend${EXT_SUFFIX}"
