#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <epoxy/gl.h>
#include <opencv2/core.hpp>

#include "System.h"

namespace py = pybind11;

namespace {

cv::Mat numpy_uint8_to_cv_mat(const py::array &array) {
    py::buffer_info info = array.request();
    if (info.itemsize != 1) {
        throw std::runtime_error("Expected uint8 image data.");
    }

    if (info.ndim == 2) {
        return cv::Mat(
                   static_cast<int>(info.shape[0]),
                   static_cast<int>(info.shape[1]),
                   CV_8UC1,
                   info.ptr)
            .clone();
    }

    if (info.ndim == 3) {
        const int rows = static_cast<int>(info.shape[0]);
        const int cols = static_cast<int>(info.shape[1]);
        const int channels = static_cast<int>(info.shape[2]);

        if (channels == 1) {
            return cv::Mat(rows, cols, CV_8UC1, info.ptr).clone();
        }
        if (channels == 3) {
            return cv::Mat(rows, cols, CV_8UC3, info.ptr).clone();
        }
        if (channels == 4) {
            return cv::Mat(rows, cols, CV_8UC4, info.ptr).clone();
        }
    }

    throw std::runtime_error("Expected a HxW, HxWx1, HxWx3, or HxWx4 uint8 image.");
}

py::array_t<float> se3_to_numpy(const Sophus::SE3f &pose) {
    Eigen::Matrix4f matrix = pose.matrix();
    py::array_t<float> output({4, 4});
    std::memcpy(output.mutable_data(), matrix.data(), 16 * sizeof(float));
    return output;
}

}  // namespace

class PyOrbSlamStereoSystem {
public:
    PyOrbSlamStereoSystem(
        const std::string &vocabulary_file,
        const std::string &settings_file,
        bool use_viewer)
        : system_(std::make_unique<ORB_SLAM3::System>(
              vocabulary_file,
              settings_file,
              ORB_SLAM3::System::STEREO,
              use_viewer)) {}

    py::array_t<float> track_stereo(
        const py::array &left_image,
        const py::array &right_image,
        double timestamp_sec) {
        cv::Mat left = numpy_uint8_to_cv_mat(left_image);
        cv::Mat right = numpy_uint8_to_cv_mat(right_image);
        Sophus::SE3f pose = system_->TrackStereo(left, right, timestamp_sec);
        return se3_to_numpy(pose);
    }

    int get_tracking_state() const {
        return system_->GetTrackingState();
    }

    bool is_lost() const {
        return system_->isLost();
    }

    void shutdown() {
        if (system_) {
            system_->Shutdown();
        }
    }

private:
    std::unique_ptr<ORB_SLAM3::System> system_;
};

class PyOrbSlamMonocularSystem {
public:
    PyOrbSlamMonocularSystem(
        const std::string &vocabulary_file,
        const std::string &settings_file,
        bool use_viewer)
        : system_(std::make_unique<ORB_SLAM3::System>(
              vocabulary_file,
              settings_file,
              ORB_SLAM3::System::MONOCULAR,
              use_viewer)) {}

    py::array_t<float> track_monocular(
        const py::array &image,
        double timestamp_sec) {
        cv::Mat frame = numpy_uint8_to_cv_mat(image);
        Sophus::SE3f pose = system_->TrackMonocular(frame, timestamp_sec);
        return se3_to_numpy(pose);
    }

    int get_tracking_state() const {
        return system_->GetTrackingState();
    }

    bool is_lost() const {
        return system_->isLost();
    }

    void shutdown() {
        if (system_) {
            system_->Shutdown();
        }
    }

private:
    std::unique_ptr<ORB_SLAM3::System> system_;
};

PYBIND11_MODULE(orbslam3_backend, module) {
    module.doc() = "Minimal pybind11 bridge for ORB-SLAM3 stereo and monocular tracking.";

    py::class_<PyOrbSlamStereoSystem>(module, "StereoSystem")
        .def(py::init<const std::string &, const std::string &, bool>(),
             py::arg("vocabulary_file"),
             py::arg("settings_file"),
             py::arg("use_viewer") = false)
        .def("track_stereo", &PyOrbSlamStereoSystem::track_stereo,
             py::arg("left_image"),
             py::arg("right_image"),
             py::arg("timestamp_sec"))
        .def("get_tracking_state", &PyOrbSlamStereoSystem::get_tracking_state)
        .def("is_lost", &PyOrbSlamStereoSystem::is_lost)
        .def("shutdown", &PyOrbSlamStereoSystem::shutdown);

    py::class_<PyOrbSlamMonocularSystem>(module, "MonocularSystem")
        .def(py::init<const std::string &, const std::string &, bool>(),
             py::arg("vocabulary_file"),
             py::arg("settings_file"),
             py::arg("use_viewer") = false)
        .def("track_monocular", &PyOrbSlamMonocularSystem::track_monocular,
             py::arg("image"),
             py::arg("timestamp_sec"))
        .def("get_tracking_state", &PyOrbSlamMonocularSystem::get_tracking_state)
        .def("is_lost", &PyOrbSlamMonocularSystem::is_lost)
        .def("shutdown", &PyOrbSlamMonocularSystem::shutdown);
}
