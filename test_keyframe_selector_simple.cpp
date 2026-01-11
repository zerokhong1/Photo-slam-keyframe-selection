/**
 * @file test_keyframe_selector_simple.cpp
 * @brief Simple test for Gaussian Keyframe Selector
 * @author Claude AI
 * @date 11/01/2026
 * 
 * This is a standalone test that can be run without the full Photo-SLAM system.
 * It tests the keyframe selector logic with mock data.
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>

// Mock GaussianCloud class for testing
namespace PhotoSLAM {

class GaussianCloud {
public:
    struct Gaussian {
        Eigen::Vector3f position;
        Eigen::Vector3f scale;
        float opacity;
    };
    
    void AddGaussian(const Eigen::Vector3f& pos, 
                     const Eigen::Vector3f& scale, 
                     float opacity) {
        gaussians_.push_back({pos, scale, opacity});
    }
    
    size_t Size() const { return gaussians_.size(); }
    
    Eigen::Vector3f GetPosition(size_t i) const { 
        return gaussians_[i].position; 
    }
    
    Eigen::Vector3f GetScale(size_t i) const { 
        return gaussians_[i].scale; 
    }
    
    float GetOpacity(size_t i) const { 
        return gaussians_[i].opacity; 
    }
    
    void Clear() { gaussians_.clear(); }
    
private:
    std::vector<Gaussian> gaussians_;
};

// Include the actual header (or redefine for testing)
#include "gaussian_keyframe_selector.h"

} // namespace PhotoSLAM

using namespace PhotoSLAM;

//==============================================================================
// Test Functions
//==============================================================================

void TestEmptyGaussians() {
    std::cout << "\n=== Test: Empty Gaussians ===" << std::endl;
    
    KeyframeSelectorParams params;
    params.coverage_threshold = 0.7f;
    params.uncertainty_threshold = 0.3f;
    
    GaussianKeyframeSelector selector(params);
    selector.SetVerbose(true);
    
    // Create empty Gaussian cloud
    auto gaussians = std::make_shared<GaussianCloud>();
    
    // Create test image and camera
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC1);
    Sophus::SE3f Tcw;  // Identity pose
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0, 0) = 500; K(1, 1) = 500;  // fx, fy
    K(0, 2) = 320; K(1, 2) = 240;  // cx, cy
    
    // Compute metrics
    KeyframeMetrics metrics = selector.ComputeMetrics(image, Tcw, gaussians, K);
    
    // Verify
    assert(metrics.coverage == 0.0f && "Empty cloud should have 0 coverage");
    assert(metrics.uncertainty == 1.0f && "Empty cloud should have max uncertainty");
    assert(selector.NeedKeyframeGaussian(metrics) && "Should need keyframe for empty cloud");
    
    std::cout << "Result: " << metrics.ToString() << std::endl;
    std::cout << "PASSED!" << std::endl;
}

void TestFullCoverage() {
    std::cout << "\n=== Test: Full Coverage ===" << std::endl;
    
    KeyframeSelectorParams params;
    params.coverage_threshold = 0.7f;
    params.grid_size = 32;
    
    GaussianKeyframeSelector selector(params);
    selector.SetVerbose(true);
    
    // Create Gaussian cloud with full coverage
    auto gaussians = std::make_shared<GaussianCloud>();
    
    // Add Gaussians in a grid pattern covering the entire view
    for (float x = -1.0f; x <= 1.0f; x += 0.1f) {
        for (float y = -0.75f; y <= 0.75f; y += 0.1f) {
            // Position at z=2 meters
            Eigen::Vector3f pos(x, y, 2.0f);
            Eigen::Vector3f scale(0.02f, 0.02f, 0.02f);
            gaussians->AddGaussian(pos, scale, 1.0f);
        }
    }
    
    std::cout << "Created " << gaussians->Size() << " Gaussians" << std::endl;
    
    // Create test image and camera
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC1);
    Sophus::SE3f Tcw;  // Identity pose
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0, 0) = 500; K(1, 1) = 500;
    K(0, 2) = 320; K(1, 2) = 240;
    
    // Compute metrics
    KeyframeMetrics metrics = selector.ComputeMetrics(image, Tcw, gaussians, K);
    
    std::cout << "Result: " << metrics.ToString() << std::endl;
    
    // Verify high coverage
    assert(metrics.coverage > 0.7f && "Full grid should have high coverage");
    
    std::cout << "PASSED!" << std::endl;
}

void TestPartialCoverage() {
    std::cout << "\n=== Test: Partial Coverage ===" << std::endl;
    
    KeyframeSelectorParams params;
    params.coverage_threshold = 0.7f;
    params.grid_size = 32;
    
    GaussianKeyframeSelector selector(params);
    selector.SetVerbose(true);
    
    // Create Gaussian cloud with only left half coverage
    auto gaussians = std::make_shared<GaussianCloud>();
    
    // Add Gaussians only in left half of view
    for (float x = -1.0f; x <= 0.0f; x += 0.1f) {
        for (float y = -0.75f; y <= 0.75f; y += 0.1f) {
            Eigen::Vector3f pos(x, y, 2.0f);
            Eigen::Vector3f scale(0.02f, 0.02f, 0.02f);
            gaussians->AddGaussian(pos, scale, 1.0f);
        }
    }
    
    std::cout << "Created " << gaussians->Size() << " Gaussians (left half only)" << std::endl;
    
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC1);
    Sophus::SE3f Tcw;
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0, 0) = 500; K(1, 1) = 500;
    K(0, 2) = 320; K(1, 2) = 240;
    
    KeyframeMetrics metrics = selector.ComputeMetrics(image, Tcw, gaussians, K);
    
    std::cout << "Result: " << metrics.ToString() << std::endl;
    
    // Should have roughly 50% coverage
    assert(metrics.coverage < 0.7f && "Partial coverage should be below threshold");
    assert(selector.NeedKeyframeGaussian(metrics) && "Should need keyframe for partial coverage");
    
    std::cout << "PASSED!" << std::endl;
}

void TestCameraMovement() {
    std::cout << "\n=== Test: Camera Movement ===" << std::endl;
    
    KeyframeSelectorParams params;
    params.coverage_threshold = 0.7f;
    
    GaussianKeyframeSelector selector(params);
    selector.SetVerbose(true);
    
    // Create Gaussian cloud
    auto gaussians = std::make_shared<GaussianCloud>();
    
    // Add Gaussians in center
    for (float x = -0.5f; x <= 0.5f; x += 0.1f) {
        for (float y = -0.5f; y <= 0.5f; y += 0.1f) {
            Eigen::Vector3f pos(x, y, 2.0f);
            Eigen::Vector3f scale(0.02f, 0.02f, 0.02f);
            gaussians->AddGaussian(pos, scale, 1.0f);
        }
    }
    
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC1);
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0, 0) = 500; K(1, 1) = 500;
    K(0, 2) = 320; K(1, 2) = 240;
    
    // Test at original position
    Sophus::SE3f Tcw1;  // Identity
    KeyframeMetrics metrics1 = selector.ComputeMetrics(image, Tcw1, gaussians, K);
    std::cout << "Position 1 (center): " << metrics1.ToString() << std::endl;
    
    // Move camera to the right
    Eigen::Vector3f translation(2.0f, 0.0f, 0.0f);
    Sophus::SE3f Tcw2(Eigen::Matrix3f::Identity(), translation);
    KeyframeMetrics metrics2 = selector.ComputeMetrics(image, Tcw2, gaussians, K);
    std::cout << "Position 2 (right): " << metrics2.ToString() << std::endl;
    
    // Coverage should be lower when camera moved away from Gaussians
    assert(metrics2.coverage < metrics1.coverage && 
           "Coverage should decrease when camera moves away");
    
    std::cout << "PASSED!" << std::endl;
}

void TestParameterLoading() {
    std::cout << "\n=== Test: Parameter Loading ===" << std::endl;
    
    // Create a temporary YAML file
    std::string yaml_content = R"(%YAML:1.0
Gaussian.CoverageThreshold: 0.65
Gaussian.GridSize: 48
Gaussian.UncertaintyThreshold: 0.25
Gaussian.DistanceWeight: 1.5
Gaussian.ScaleWeight: 0.6
Gaussian.AngleWeight: 0.4
Gaussian.InfoGainThreshold: 0.45
Gaussian.OptimalBaselineAngle: 25.0
Gaussian.MinKeyframeInterval: 8
Gaussian.MaxKeyframeInterval: 40
)";
    
    std::string temp_file = "/tmp/test_config.yaml";
    std::ofstream ofs(temp_file);
    ofs << yaml_content;
    ofs.close();
    
    // Load parameters
    KeyframeSelectorParams params;
    params.LoadFromFile(temp_file);
    
    // Verify
    assert(std::abs(params.coverage_threshold - 0.65f) < 0.001f);
    assert(params.grid_size == 48);
    assert(std::abs(params.uncertainty_threshold - 0.25f) < 0.001f);
    assert(std::abs(params.distance_weight - 1.5f) < 0.001f);
    assert(std::abs(params.optimal_baseline_angle - 25.0f) < 0.001f);
    assert(params.min_frames_between_kf == 8);
    assert(params.max_frames_between_kf == 40);
    
    std::cout << "PASSED!" << std::endl;
    
    // Cleanup
    std::remove(temp_file.c_str());
}

void TestDecisionLogic() {
    std::cout << "\n=== Test: Decision Logic ===" << std::endl;
    
    KeyframeSelectorParams params;
    params.coverage_threshold = 0.7f;
    params.uncertainty_threshold = 0.3f;
    params.info_gain_threshold = 0.5f;
    
    GaussianKeyframeSelector selector(params);
    
    // Test case 1: High coverage, low uncertainty -> no keyframe
    KeyframeMetrics m1;
    m1.coverage = 0.85f;
    m1.uncertainty = 0.2f;
    m1.information_gain = 0.3f;
    m1.is_new_region = false;
    assert(!selector.NeedKeyframeGaussian(m1) && "Should NOT need keyframe");
    std::cout << "  High coverage, low uncertainty: NO keyframe - OK" << std::endl;
    
    // Test case 2: Low coverage -> keyframe
    KeyframeMetrics m2;
    m2.coverage = 0.5f;
    m2.uncertainty = 0.2f;
    m2.information_gain = 0.3f;
    m2.is_new_region = false;
    assert(selector.NeedKeyframeGaussian(m2) && "Should need keyframe (low coverage)");
    std::cout << "  Low coverage: YES keyframe - OK" << std::endl;
    
    // Test case 3: High uncertainty -> keyframe
    KeyframeMetrics m3;
    m3.coverage = 0.85f;
    m3.uncertainty = 0.5f;
    m3.information_gain = 0.3f;
    m3.is_new_region = false;
    assert(selector.NeedKeyframeGaussian(m3) && "Should need keyframe (high uncertainty)");
    std::cout << "  High uncertainty: YES keyframe - OK" << std::endl;
    
    // Test case 4: New region -> keyframe
    KeyframeMetrics m4;
    m4.coverage = 0.25f;
    m4.uncertainty = 0.2f;
    m4.information_gain = 0.3f;
    m4.is_new_region = true;
    assert(selector.NeedKeyframeGaussian(m4) && "Should need keyframe (new region)");
    std::cout << "  New region: YES keyframe - OK" << std::endl;
    
    // Test case 5: High info gain -> keyframe
    KeyframeMetrics m5;
    m5.coverage = 0.85f;
    m5.uncertainty = 0.2f;
    m5.information_gain = 0.7f;
    m5.is_new_region = false;
    assert(selector.NeedKeyframeGaussian(m5) && "Should need keyframe (high info gain)");
    std::cout << "  High info gain: YES keyframe - OK" << std::endl;
    
    std::cout << "PASSED!" << std::endl;
}

//==============================================================================
// Main
//==============================================================================

int main(int argc, char** argv) {
    std::cout << "========================================" << std::endl;
    std::cout << "Gaussian Keyframe Selector Tests" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        TestEmptyGaussians();
        TestFullCoverage();
        TestPartialCoverage();
        TestCameraMovement();
        TestParameterLoading();
        TestDecisionLogic();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "All tests PASSED!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Test FAILED with exception: " << e.what() << std::endl;
        return 1;
    }
}
