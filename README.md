# Hướng Dẫn Modify Keyframe Selection trong Photo-SLAM

> **Tác giả**: Claude AI  
> **Ngày tạo**: 11/01/2026  
> **Phiên bản**: 1.0

---

## Mục Lục

1. [Tổng Quan](#1-tổng-quan)
2. [Kiến Trúc Keyframe Selection](#2-kiến-trúc-keyframe-selection)
3. [Phân Tích Code Hiện Tại](#3-phân-tích-code-hiện-tại)
4. [Thiết Kế Gaussian-Aware Selection](#4-thiết-kế-gaussian-aware-selection)
5. [Implementation Chi Tiết](#5-implementation-chi-tiết)
6. [Configuration & Parameters](#6-configuration--parameters)
7. [Testing & Debugging](#7-testing--debugging)
8. [Best Practices](#8-best-practices)

---

## 1. Tổng Quan

### 1.1 Tại sao cần Modify Keyframe Selection?

Photo-SLAM kế thừa keyframe selection từ ORB-SLAM3, được thiết kế cho **geometric SLAM** thuần túy. Tuy nhiên, Photo-SLAM sử dụng **3D Gaussian Splatting** để tạo photorealistic reconstruction, đòi hỏi các tiêu chí khác:

| ORB-SLAM3 (Geometric) | Photo-SLAM (Photometric) |
|----------------------|--------------------------|
| Feature overlap ratio | Gaussian coverage |
| Tracking quality | Rendering uncertainty |
| Map point visibility | View diversity cho Gaussians |

### 1.2 Mục tiêu

- **Đảm bảo coverage**: Mọi vùng trong scene đều có đủ Gaussians
- **Tối ưu view diversity**: Chọn keyframes từ các góc nhìn đa dạng
- **Cân bằng efficiency**: Không tạo quá nhiều keyframes gây overhead

### 1.3 Cấu trúc thư mục liên quan

```
Photo-SLAM/
├── ORB-SLAM3/
│   ├── include/
│   │   └── Tracking.h              # Tracking class declaration
│   └── src/
│       └── Tracking.cc             # NeedNewKeyFrame() implementation
├── include/
│   ├── gaussian_mapper.h           # GaussianMapper declaration
│   └── gaussian_keyframe.h         # Gaussian keyframe management
├── src/
│   ├── gaussian_mapper.cpp         # Gaussian operations
│   └── gaussian_keyframe.cpp       # Keyframe-Gaussian relationship
└── configs/
    └── *.yaml                      # Configuration files
```

---

## 2. Kiến Trúc Keyframe Selection

### 2.1 Flow tổng quan

```
┌─────────────────────────────────────────────────────────────┐
│                      New Frame Arrives                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ORB Feature Extraction                    │
│              (ORBextractor::operator())                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      Pose Estimation                         │
│         (TrackWithMotionModel / TrackReferenceKeyFrame)     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   NeedNewKeyFrame()?                         │
│                                                              │
│  ┌─────────────────────┐    ┌─────────────────────┐         │
│  │ Original Conditions │ OR │ Gaussian Conditions │         │
│  │  - Max frames       │    │  - Low coverage     │         │
│  │  - Low overlap      │    │  - High uncertainty │         │
│  │  - Tracking weak    │    │  - Info gain high   │         │
│  └─────────────────────┘    └─────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                   YES                  NO
                    │                   │
                    ▼                   ▼
    ┌───────────────────────┐   ┌───────────────────────┐
    │  CreateNewKeyFrame()  │   │   Continue Tracking   │
    │  - Add to map         │   │                       │
    │  - Init Gaussians     │   │                       │
    │  - Trigger optimizer  │   │                       │
    └───────────────────────┘   └───────────────────────┘
```

### 2.2 Interaction giữa các components

```
┌─────────────┐         ┌─────────────────┐         ┌────────────────┐
│   Tracking  │ ──────▶ │ GaussianMapper  │ ──────▶ │ GaussianCloud  │
│             │         │                 │         │                │
│ - Pose      │         │ - Coverage      │         │ - Positions    │
│ - Features  │         │ - Uncertainty   │         │ - Scales       │
│ - Keyframe  │         │ - Info Gain     │         │ - Opacities    │
│   decision  │         │                 │         │ - SH coeffs    │
└─────────────┘         └─────────────────┘         └────────────────┘
```

---

## 3. Phân Tích Code Hiện Tại

### 3.1 NeedNewKeyFrame() trong ORB-SLAM3

File: `ORB-SLAM3/src/Tracking.cc`

```cpp
bool Tracking::NeedNewKeyFrame()
{
    // Condition 0: Only tracking mode - no keyframes
    if(mbOnlyTracking)
        return false;

    // Condition 1: Local Mapping is freezed by Loop Closure
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpAtlas->KeyFramesInMap();

    // Condition 2: Not enough frames since last keyframe
    if(mCurrentFrame.mnId < mnLastKeyFrameId + mMinFrames)
        return false;

    // Condition 3: Check Local Mapping queue
    if(mpLocalMapper->KeyframesInQueue() > 2)
        return false;

    // Condition 4: Enough tracked points but low overlap with reference
    int nRefMatches = mpReferenceKF->TrackedMapPoints(2);
    
    float thRefRatio = 0.9f;
    if(nKFs < 2)
        thRefRatio = 0.4f;
    if(mSensor == System::MONOCULAR)
        thRefRatio = 0.9f;

    // Decision conditions
    const bool c1 = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
    const bool c2 = (mnMatchesInliers < nRefMatches * thRefRatio) && 
                    mnMatchesInliers > 15;

    if(c1 || c2)
    {
        if(mpLocalMapper->AcceptKeyFrames())
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            // Additional checks for stereo/RGBD
            if(mSensor != System::MONOCULAR)
            {
                if(mpLocalMapper->KeyframesInQueue() < 3)
                    return true;
            }
            return false;
        }
    }
    
    return false;
}
```

### 3.2 Các vấn đề với approach hiện tại

1. **Không xét Gaussian coverage**: Có thể tạo keyframe ở vùng đã có đủ Gaussians
2. **Bỏ qua photometric quality**: Không đánh giá rendering quality từ view hiện tại
3. **Thiếu view diversity**: Không optimize cho multi-view reconstruction

---

## 4. Thiết Kế Gaussian-Aware Selection

### 4.1 Metrics mới cần implement

#### 4.1.1 Gaussian Coverage

Đo tỷ lệ image được cover bởi projected Gaussians:

```
Coverage = (Số cells có Gaussians) / (Tổng số cells trong grid)
```

#### 4.1.2 Rendering Uncertainty

Đánh giá độ tin cậy của rendering từ view hiện tại:

```
Uncertainty = f(distance, scale, view_angle)
```

#### 4.1.3 Information Gain

Lượng thông tin mới mà keyframe này đóng góp:

```
InfoGain = Σ (angle_score × gaussian_uncertainty)
```

### 4.2 Decision Logic

```
need_keyframe = (
    # Original conditions
    (frames_since_last_kf > max_frames) OR
    (feature_overlap < threshold) OR
    
    # NEW: Gaussian conditions
    (gaussian_coverage < coverage_threshold) OR
    (rendering_uncertainty > uncertainty_threshold) OR
    (information_gain > info_gain_threshold)
)
```

### 4.3 Priority và weighting

```cpp
// Priority levels
enum KeyframeReason {
    REASON_MAX_FRAMES = 0,        // Lowest priority
    REASON_LOW_OVERLAP = 1,
    REASON_LOW_COVERAGE = 2,      // NEW
    REASON_HIGH_UNCERTAINTY = 3,  // NEW
    REASON_HIGH_INFO_GAIN = 4     // Highest priority - NEW
};
```

---

## 5. Implementation Chi Tiết

### 5.1 Header file modifications

#### File: `include/gaussian_keyframe_selector.h` (NEW)

```cpp
#ifndef GAUSSIAN_KEYFRAME_SELECTOR_H
#define GAUSSIAN_KEYFRAME_SELECTOR_H

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <memory>

namespace PhotoSLAM {

// Forward declarations
class GaussianCloud;
class KeyFrame;

/**
 * @brief Gaussian-aware keyframe selection criteria
 */
struct KeyframeMetrics {
    float coverage;           // [0, 1] - ratio of image covered by Gaussians
    float uncertainty;        // [0, inf) - rendering uncertainty
    float information_gain;   // [0, inf) - potential info gain from this view
    bool is_new_region;       // true if mostly unseen area
    
    // Debug info
    int visible_gaussians;
    int total_gaussians;
    float min_depth;
    float max_depth;
};

/**
 * @brief Parameters for keyframe selection
 */
struct KeyframeSelectorParams {
    // Coverage parameters
    float coverage_threshold = 0.7f;      // Min coverage to skip keyframe
    int grid_size = 32;                    // Grid cell size in pixels
    
    // Uncertainty parameters
    float uncertainty_threshold = 0.3f;   // Max acceptable uncertainty
    float distance_weight = 1.0f;
    float scale_weight = 0.5f;
    float angle_weight = 0.3f;
    
    // Information gain parameters
    float info_gain_threshold = 0.5f;
    float optimal_baseline_angle = 20.0f; // degrees
    
    // Temporal constraints
    int min_frames_between_kf = 5;
    int max_frames_between_kf = 30;
    
    // Load from YAML
    void LoadFromFile(const std::string& filename);
};

/**
 * @brief Gaussian-aware keyframe selector
 */
class GaussianKeyframeSelector {
public:
    GaussianKeyframeSelector(const KeyframeSelectorParams& params);
    ~GaussianKeyframeSelector() = default;
    
    /**
     * @brief Compute all metrics for current view
     * @param image Current grayscale image
     * @param Tcw Camera pose (world to camera)
     * @param gaussians Pointer to Gaussian cloud
     * @param K Camera intrinsic matrix
     * @return KeyframeMetrics struct with all computed values
     */
    KeyframeMetrics ComputeMetrics(
        const cv::Mat& image,
        const Sophus::SE3f& Tcw,
        const std::shared_ptr<GaussianCloud>& gaussians,
        const Eigen::Matrix3f& K
    );
    
    /**
     * @brief Check if new keyframe is needed based on Gaussian criteria
     * @param metrics Pre-computed metrics
     * @return true if keyframe should be created
     */
    bool NeedKeyframeGaussian(const KeyframeMetrics& metrics);
    
    /**
     * @brief Compute coverage ratio
     */
    float ComputeCoverage(
        const cv::Mat& image,
        const Sophus::SE3f& Tcw,
        const std::shared_ptr<GaussianCloud>& gaussians,
        const Eigen::Matrix3f& K
    );
    
    /**
     * @brief Compute rendering uncertainty
     */
    float ComputeUncertainty(
        const Sophus::SE3f& Tcw,
        const std::shared_ptr<GaussianCloud>& gaussians
    );
    
    /**
     * @brief Compute information gain from this view
     */
    float ComputeInformationGain(
        const Sophus::SE3f& Tcw,
        const std::shared_ptr<GaussianCloud>& gaussians,
        const std::vector<std::shared_ptr<KeyFrame>>& existing_keyframes
    );
    
    // Setters
    void SetParams(const KeyframeSelectorParams& params) { mParams = params; }
    void SetVerbose(bool verbose) { mbVerbose = verbose; }
    
private:
    KeyframeSelectorParams mParams;
    bool mbVerbose = false;
    
    // Helper functions
    bool IsGaussianVisible(
        const Eigen::Vector3f& position,
        const Sophus::SE3f& Tcw,
        const Eigen::Matrix3f& K,
        int width, int height
    );
    
    float ComputeAngleDifference(
        const Eigen::Vector3f& gaussian_pos,
        const Eigen::Vector3f& view1_pos,
        const Eigen::Vector3f& view2_pos
    );
    
    float ComputeGaussianUncertainty(
        const Eigen::Vector3f& gaussian_pos,
        const Eigen::Vector3f& gaussian_scale,
        const Eigen::Vector3f& camera_pos
    );
};

} // namespace PhotoSLAM

#endif // GAUSSIAN_KEYFRAME_SELECTOR_H
```

### 5.2 Implementation file

#### File: `src/gaussian_keyframe_selector.cpp` (NEW)

```cpp
#include "gaussian_keyframe_selector.h"
#include "gaussian_cloud.h"
#include "KeyFrame.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>

namespace PhotoSLAM {

//==============================================================================
// KeyframeSelectorParams
//==============================================================================

void KeyframeSelectorParams::LoadFromFile(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    
    if (!fs.isOpened()) {
        std::cerr << "[KeyframeSelector] Warning: Cannot open " << filename 
                  << ", using defaults." << std::endl;
        return;
    }
    
    // Coverage params
    if (!fs["Gaussian.CoverageThreshold"].empty())
        fs["Gaussian.CoverageThreshold"] >> coverage_threshold;
    if (!fs["Gaussian.GridSize"].empty())
        fs["Gaussian.GridSize"] >> grid_size;
    
    // Uncertainty params
    if (!fs["Gaussian.UncertaintyThreshold"].empty())
        fs["Gaussian.UncertaintyThreshold"] >> uncertainty_threshold;
    if (!fs["Gaussian.DistanceWeight"].empty())
        fs["Gaussian.DistanceWeight"] >> distance_weight;
    if (!fs["Gaussian.ScaleWeight"].empty())
        fs["Gaussian.ScaleWeight"] >> scale_weight;
    if (!fs["Gaussian.AngleWeight"].empty())
        fs["Gaussian.AngleWeight"] >> angle_weight;
    
    // Info gain params
    if (!fs["Gaussian.InfoGainThreshold"].empty())
        fs["Gaussian.InfoGainThreshold"] >> info_gain_threshold;
    if (!fs["Gaussian.OptimalBaselineAngle"].empty())
        fs["Gaussian.OptimalBaselineAngle"] >> optimal_baseline_angle;
    
    // Temporal params
    if (!fs["Gaussian.MinKeyframeInterval"].empty())
        fs["Gaussian.MinKeyframeInterval"] >> min_frames_between_kf;
    if (!fs["Gaussian.MaxKeyframeInterval"].empty())
        fs["Gaussian.MaxKeyframeInterval"] >> max_frames_between_kf;
    
    fs.release();
    
    std::cout << "[KeyframeSelector] Loaded params: "
              << "coverage=" << coverage_threshold 
              << ", uncertainty=" << uncertainty_threshold
              << ", info_gain=" << info_gain_threshold << std::endl;
}

//==============================================================================
// GaussianKeyframeSelector
//==============================================================================

GaussianKeyframeSelector::GaussianKeyframeSelector(
    const KeyframeSelectorParams& params)
    : mParams(params), mbVerbose(false)
{
}

KeyframeMetrics GaussianKeyframeSelector::ComputeMetrics(
    const cv::Mat& image,
    const Sophus::SE3f& Tcw,
    const std::shared_ptr<GaussianCloud>& gaussians,
    const Eigen::Matrix3f& K)
{
    KeyframeMetrics metrics;
    
    if (!gaussians || gaussians->Size() == 0) {
        // No Gaussians yet - definitely need keyframe
        metrics.coverage = 0.0f;
        metrics.uncertainty = 1.0f;
        metrics.information_gain = 1.0f;
        metrics.is_new_region = true;
        metrics.visible_gaussians = 0;
        metrics.total_gaussians = 0;
        return metrics;
    }
    
    metrics.coverage = ComputeCoverage(image, Tcw, gaussians, K);
    metrics.uncertainty = ComputeUncertainty(Tcw, gaussians);
    metrics.is_new_region = (metrics.coverage < 0.3f);
    metrics.total_gaussians = gaussians->Size();
    
    if (mbVerbose) {
        std::cout << "[KeyframeSelector] Metrics: "
                  << "coverage=" << metrics.coverage
                  << ", uncertainty=" << metrics.uncertainty
                  << ", visible=" << metrics.visible_gaussians
                  << "/" << metrics.total_gaussians << std::endl;
    }
    
    return metrics;
}

bool GaussianKeyframeSelector::NeedKeyframeGaussian(
    const KeyframeMetrics& metrics)
{
    // Condition 1: Low coverage
    bool low_coverage = (metrics.coverage < mParams.coverage_threshold);
    
    // Condition 2: High uncertainty
    bool high_uncertainty = (metrics.uncertainty > mParams.uncertainty_threshold);
    
    // Condition 3: New region
    bool new_region = metrics.is_new_region;
    
    // Condition 4: High information gain
    bool high_info_gain = (metrics.information_gain > mParams.info_gain_threshold);
    
    bool need_kf = low_coverage || high_uncertainty || new_region || high_info_gain;
    
    if (mbVerbose && need_kf) {
        std::cout << "[KeyframeSelector] Need KF: "
                  << "low_cov=" << low_coverage
                  << ", high_unc=" << high_uncertainty
                  << ", new_reg=" << new_region
                  << ", high_info=" << high_info_gain << std::endl;
    }
    
    return need_kf;
}

float GaussianKeyframeSelector::ComputeCoverage(
    const cv::Mat& image,
    const Sophus::SE3f& Tcw,
    const std::shared_ptr<GaussianCloud>& gaussians,
    const Eigen::Matrix3f& K)
{
    if (!gaussians || gaussians->Size() == 0) {
        return 0.0f;
    }
    
    int width = image.cols;
    int height = image.rows;
    
    // Create coverage grid
    int grid_cols = (width + mParams.grid_size - 1) / mParams.grid_size;
    int grid_rows = (height + mParams.grid_size - 1) / mParams.grid_size;
    std::vector<bool> grid_covered(grid_cols * grid_rows, false);
    
    // Camera intrinsics
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);
    
    int visible_count = 0;
    
    // Project each Gaussian to image
    for (size_t i = 0; i < gaussians->Size(); ++i) {
        Eigen::Vector3f pw = gaussians->GetPosition(i);
        
        // Transform to camera frame
        Eigen::Vector3f pc = Tcw * pw;
        
        // Check if in front of camera
        if (pc.z() <= 0.1f) continue;
        
        // Project to image plane
        float u = fx * pc.x() / pc.z() + cx;
        float v = fy * pc.y() / pc.z() + cy;
        
        // Check bounds
        if (u < 0 || u >= width || v < 0 || v >= height) continue;
        
        visible_count++;
        
        // Mark grid cell as covered
        int gx = static_cast<int>(u) / mParams.grid_size;
        int gy = static_cast<int>(v) / mParams.grid_size;
        
        gx = std::clamp(gx, 0, grid_cols - 1);
        gy = std::clamp(gy, 0, grid_rows - 1);
        
        grid_covered[gy * grid_cols + gx] = true;
        
        // Also mark neighboring cells based on Gaussian scale
        Eigen::Vector3f scale = gaussians->GetScale(i);
        float avg_scale = scale.mean();
        float projected_radius = (fx * avg_scale) / pc.z();
        int cell_radius = static_cast<int>(projected_radius / mParams.grid_size) + 1;
        
        for (int dy = -cell_radius; dy <= cell_radius; ++dy) {
            for (int dx = -cell_radius; dx <= cell_radius; ++dx) {
                int nx = gx + dx;
                int ny = gy + dy;
                if (nx >= 0 && nx < grid_cols && ny >= 0 && ny < grid_rows) {
                    grid_covered[ny * grid_cols + nx] = true;
                }
            }
        }
    }
    
    // Compute coverage ratio
    int covered_cells = std::count(grid_covered.begin(), grid_covered.end(), true);
    float coverage = static_cast<float>(covered_cells) / grid_covered.size();
    
    return coverage;
}

float GaussianKeyframeSelector::ComputeUncertainty(
    const Sophus::SE3f& Tcw,
    const std::shared_ptr<GaussianCloud>& gaussians)
{
    if (!gaussians || gaussians->Size() == 0) {
        return 1.0f;  // Max uncertainty if no Gaussians
    }
    
    Eigen::Vector3f camera_center = Tcw.inverse().translation();
    
    float total_uncertainty = 0.0f;
    int count = 0;
    
    for (size_t i = 0; i < gaussians->Size(); ++i) {
        Eigen::Vector3f pos = gaussians->GetPosition(i);
        Eigen::Vector3f scale = gaussians->GetScale(i);
        
        // Check if Gaussian is visible (in front of camera)
        Eigen::Vector3f pc = Tcw * pos;
        if (pc.z() <= 0.1f) continue;
        
        // Compute uncertainty for this Gaussian
        float gaussian_unc = ComputeGaussianUncertainty(pos, scale, camera_center);
        total_uncertainty += gaussian_unc;
        count++;
    }
    
    if (count == 0) return 1.0f;
    
    return total_uncertainty / count;
}

float GaussianKeyframeSelector::ComputeInformationGain(
    const Sophus::SE3f& Tcw,
    const std::shared_ptr<GaussianCloud>& gaussians,
    const std::vector<std::shared_ptr<KeyFrame>>& existing_keyframes)
{
    if (!gaussians || gaussians->Size() == 0 || existing_keyframes.empty()) {
        return 1.0f;  // High info gain if nothing exists
    }
    
    Eigen::Vector3f cam_pos = Tcw.inverse().translation();
    float total_info_gain = 0.0f;
    int count = 0;
    
    for (size_t i = 0; i < gaussians->Size(); ++i) {
        Eigen::Vector3f gaussian_pos = gaussians->GetPosition(i);
        
        // Check visibility
        Eigen::Vector3f pc = Tcw * gaussian_pos;
        if (pc.z() <= 0.1f) continue;
        
        // Find minimum angle difference from existing keyframes
        float min_angle_diff = M_PI;  // Max possible
        
        for (const auto& kf : existing_keyframes) {
            Eigen::Vector3f kf_pos = kf->GetCameraCenter();
            float angle_diff = ComputeAngleDifference(gaussian_pos, cam_pos, kf_pos);
            min_angle_diff = std::min(min_angle_diff, angle_diff);
        }
        
        // Information gain is higher when view angle is different but not too different
        // Optimal baseline is around 15-30 degrees
        float optimal_angle = mParams.optimal_baseline_angle * M_PI / 180.0f;
        float angle_score = std::exp(-std::pow(min_angle_diff - optimal_angle, 2) / 0.1f);
        
        // Weight by Gaussian uncertainty (uncertain Gaussians benefit more from new views)
        Eigen::Vector3f scale = gaussians->GetScale(i);
        float gaussian_unc = scale.mean();  // Larger scale = more uncertain
        
        total_info_gain += angle_score * gaussian_unc;
        count++;
    }
    
    if (count == 0) return 0.0f;
    
    return total_info_gain / count;
}

//==============================================================================
// Helper Functions
//==============================================================================

bool GaussianKeyframeSelector::IsGaussianVisible(
    const Eigen::Vector3f& position,
    const Sophus::SE3f& Tcw,
    const Eigen::Matrix3f& K,
    int width, int height)
{
    Eigen::Vector3f pc = Tcw * position;
    if (pc.z() <= 0.1f) return false;
    
    float fx = K(0, 0);
    float fy = K(1, 1);
    float cx = K(0, 2);
    float cy = K(1, 2);
    
    float u = fx * pc.x() / pc.z() + cx;
    float v = fy * pc.y() / pc.z() + cy;
    
    return (u >= 0 && u < width && v >= 0 && v < height);
}

float GaussianKeyframeSelector::ComputeAngleDifference(
    const Eigen::Vector3f& gaussian_pos,
    const Eigen::Vector3f& view1_pos,
    const Eigen::Vector3f& view2_pos)
{
    Eigen::Vector3f dir1 = (gaussian_pos - view1_pos).normalized();
    Eigen::Vector3f dir2 = (gaussian_pos - view2_pos).normalized();
    
    float dot = std::clamp(dir1.dot(dir2), -1.0f, 1.0f);
    return std::acos(dot);
}

float GaussianKeyframeSelector::ComputeGaussianUncertainty(
    const Eigen::Vector3f& gaussian_pos,
    const Eigen::Vector3f& gaussian_scale,
    const Eigen::Vector3f& camera_pos)
{
    // Distance component
    float distance = (gaussian_pos - camera_pos).norm();
    float distance_unc = mParams.distance_weight * distance;
    
    // Scale component (larger Gaussians are more uncertain)
    float avg_scale = gaussian_scale.mean();
    float scale_unc = mParams.scale_weight * avg_scale;
    
    // Combined uncertainty
    return distance_unc * scale_unc;
}

} // namespace PhotoSLAM
```

### 5.3 Modify Tracking.cc

#### File: `ORB-SLAM3/src/Tracking.cc` (MODIFIED)

```cpp
// Add includes at top
#include "gaussian_keyframe_selector.h"

// Add member variable in Tracking class (in header)
// std::shared_ptr<PhotoSLAM::GaussianKeyframeSelector> mpGaussianSelector;

// Initialize in constructor
void Tracking::InitGaussianSelector(const std::string& strSettingsFile)
{
    PhotoSLAM::KeyframeSelectorParams params;
    params.LoadFromFile(strSettingsFile);
    mpGaussianSelector = std::make_shared<PhotoSLAM::GaussianKeyframeSelector>(params);
    mpGaussianSelector->SetVerbose(mbVerbose);
}

// Modified NeedNewKeyFrame
bool Tracking::NeedNewKeyFrame()
{
    // ==================== ORIGINAL CONDITIONS ====================
    
    if(mbOnlyTracking)
        return false;

    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false;

    const int nKFs = mpAtlas->KeyFramesInMap();

    // Minimum frames constraint
    if(mCurrentFrame.mnId < mnLastKeyFrameId + mMinFrames)
        return false;

    // Local Mapping queue check
    if(mSensor == System::MONOCULAR && mpLocalMapper->KeyframesInQueue() > 2)
        return false;

    // Reference keyframe overlap
    int nRefMatches = mpReferenceKF->TrackedMapPoints(2);
    
    float thRefRatio = 0.9f;
    if(nKFs < 2)
        thRefRatio = 0.4f;
    if(mSensor == System::MONOCULAR)
        thRefRatio = 0.9f;

    // Original conditions
    const bool c1_max_frames = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
    const bool c2_low_overlap = (mnMatchesInliers < nRefMatches * thRefRatio) && 
                                 mnMatchesInliers > 15;

    // ==================== NEW: GAUSSIAN CONDITIONS ====================
    
    bool c3_low_coverage = false;
    bool c4_high_uncertainty = false;
    bool c5_new_region = false;
    
    if (mpGaussianSelector && mpGaussianMapper) {
        // Get current pose
        Sophus::SE3f Tcw;
        if (mSensor == System::MONOCULAR || mSensor == System::IMU_MONOCULAR) {
            Tcw = mCurrentFrame.GetPose();
        } else {
            cv::Mat Tcw_cv = mCurrentFrame.mTcw;
            Tcw = Converter::toSE3f(Tcw_cv);
        }
        
        // Compute Gaussian-aware metrics
        PhotoSLAM::KeyframeMetrics metrics = mpGaussianSelector->ComputeMetrics(
            mImGray,
            Tcw,
            mpGaussianMapper->GetGaussianCloud(),
            mK_eigen  // Eigen version of camera matrix
        );
        
        c3_low_coverage = (metrics.coverage < 0.7f);
        c4_high_uncertainty = (metrics.uncertainty > 0.3f);
        c5_new_region = metrics.is_new_region;
        
        // Debug logging
        if (mbVerbose) {
            std::cout << "[Tracking::NeedNewKeyFrame] "
                      << "Frame " << mCurrentFrame.mnId
                      << " | coverage=" << metrics.coverage
                      << " | uncertainty=" << metrics.uncertainty
                      << " | c1=" << c1_max_frames
                      << " | c2=" << c2_low_overlap
                      << " | c3=" << c3_low_coverage
                      << " | c4=" << c4_high_uncertainty
                      << " | c5=" << c5_new_region
                      << std::endl;
        }
    }

    // ==================== COMBINED DECISION ====================
    
    bool need_keyframe = (c1_max_frames || c2_low_overlap) || 
                         (c3_low_coverage || c4_high_uncertainty || c5_new_region);

    if (need_keyframe)
    {
        if (mpLocalMapper->AcceptKeyFrames())
        {
            return true;
        }
        else
        {
            mpLocalMapper->InterruptBA();
            if (mSensor != System::MONOCULAR)
            {
                if (mpLocalMapper->KeyframesInQueue() < 3)
                    return true;
                else
                    return false;
            }
            else
            {
                return false;
            }
        }
    }
    
    return false;
}
```

---

## 6. Configuration & Parameters

### 6.1 YAML Configuration File

#### File: `configs/TUM_RGBD_gaussian.yaml`

```yaml
%YAML:1.0

# ============================================================
# Camera Parameters (unchanged)
# ============================================================
Camera.type: "PinHole"
Camera.fx: 517.3
Camera.fy: 516.5
Camera.cx: 318.6
Camera.cy: 255.3

Camera.k1: 0.2624
Camera.k2: -0.9531
Camera.p1: -0.0054
Camera.p2: 0.0026
Camera.k3: 1.1633

Camera.width: 640
Camera.height: 480
Camera.fps: 30.0

Camera.RGB: 1

# ============================================================
# ORB Extractor Parameters (unchanged)
# ============================================================
ORBextractor.nFeatures: 1000
ORBextractor.scaleFactor: 1.2
ORBextractor.nLevels: 8
ORBextractor.iniThFAST: 20
ORBextractor.minThFAST: 7

# ============================================================
# NEW: Gaussian Keyframe Selection Parameters
# ============================================================

# Coverage threshold
# Keyframe created if coverage < this value
# Range: [0.0, 1.0], Default: 0.7
Gaussian.CoverageThreshold: 0.7

# Grid size for coverage computation (pixels)
# Smaller = more precise but slower
# Default: 32
Gaussian.GridSize: 32

# Uncertainty threshold
# Keyframe created if uncertainty > this value
# Range: [0.0, inf), Default: 0.3
Gaussian.UncertaintyThreshold: 0.3

# Weights for uncertainty computation
Gaussian.DistanceWeight: 1.0
Gaussian.ScaleWeight: 0.5
Gaussian.AngleWeight: 0.3

# Information gain threshold
# Keyframe created if info gain > this value
# Range: [0.0, inf), Default: 0.5
Gaussian.InfoGainThreshold: 0.5

# Optimal baseline angle (degrees)
# Best angle difference for multi-view reconstruction
# Default: 20.0
Gaussian.OptimalBaselineAngle: 20.0

# Temporal constraints
# Minimum frames between keyframes
Gaussian.MinKeyframeInterval: 5
# Maximum frames without keyframe
Gaussian.MaxKeyframeInterval: 30

# ============================================================
# Viewer Parameters (unchanged)
# ============================================================
Viewer.KeyFrameSize: 0.05
Viewer.KeyFrameLineWidth: 1.0
Viewer.GraphLineWidth: 0.9
Viewer.PointSize: 2.0
Viewer.CameraSize: 0.08
Viewer.CameraLineWidth: 3.0
Viewer.ViewpointX: 0.0
Viewer.ViewpointY: -0.7
Viewer.ViewpointZ: -1.8
Viewer.ViewpointF: 500.0
```

### 6.2 Parameter Tuning Guide

| Parameter | Low Value Effect | High Value Effect | Recommended Range |
|-----------|-----------------|-------------------|-------------------|
| `CoverageThreshold` | More keyframes, better coverage | Fewer keyframes, possible gaps | 0.6 - 0.8 |
| `UncertaintyThreshold` | More keyframes, lower uncertainty | Fewer keyframes, higher uncertainty | 0.2 - 0.4 |
| `GridSize` | More precise coverage | Faster computation | 16 - 64 |
| `OptimalBaselineAngle` | Closer viewpoints | Wider baselines | 15 - 30 |
| `MinKeyframeInterval` | Faster but redundant | Slower but efficient | 3 - 10 |

---

## 7. Testing & Debugging

### 7.1 Visualization Tool

#### File: `src/keyframe_visualizer.cpp` (NEW)

```cpp
#include <opencv2/opencv.hpp>
#include "gaussian_keyframe_selector.h"

namespace PhotoSLAM {

class KeyframeVisualizer {
public:
    static cv::Mat VisualizeKeyframeDecision(
        const cv::Mat& image,
        const KeyframeMetrics& metrics,
        bool is_keyframe,
        int grid_size = 32)
    {
        cv::Mat vis;
        cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);
        
        int width = image.cols;
        int height = image.rows;
        int grid_cols = (width + grid_size - 1) / grid_size;
        int grid_rows = (height + grid_size - 1) / grid_size;
        
        // Draw grid
        for (int i = 1; i < grid_cols; ++i) {
            int x = i * grid_size;
            cv::line(vis, cv::Point(x, 0), cv::Point(x, height), 
                     cv::Scalar(50, 50, 50), 1);
        }
        for (int i = 1; i < grid_rows; ++i) {
            int y = i * grid_size;
            cv::line(vis, cv::Point(0, y), cv::Point(width, y), 
                     cv::Scalar(50, 50, 50), 1);
        }
        
        // Draw info bar
        cv::rectangle(vis, cv::Point(0, 0), cv::Point(width, 60), 
                      cv::Scalar(0, 0, 0), cv::FILLED);
        
        // Text info
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);
        
        // Line 1: Coverage and Uncertainty
        ss << "Coverage: " << metrics.coverage 
           << " | Uncertainty: " << metrics.uncertainty;
        cv::putText(vis, ss.str(), cv::Point(10, 20), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // Line 2: Gaussian counts
        ss.str("");
        ss << "Visible: " << metrics.visible_gaussians 
           << "/" << metrics.total_gaussians << " Gaussians";
        cv::putText(vis, ss.str(), cv::Point(10, 40), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
        // Keyframe indicator
        std::string kf_text = is_keyframe ? "KEYFRAME" : "skip";
        cv::Scalar kf_color = is_keyframe ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        cv::putText(vis, kf_text, cv::Point(width - 120, 40), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, kf_color, 2);
        
        // Border color based on decision
        cv::rectangle(vis, cv::Point(0, 0), cv::Point(width - 1, height - 1), 
                      kf_color, 3);
        
        return vis;
    }
    
    static void SaveVisualization(
        const cv::Mat& vis,
        const std::string& output_dir,
        int frame_id,
        bool is_keyframe)
    {
        std::stringstream ss;
        ss << output_dir << "/frame_" << std::setw(6) << std::setfill('0') 
           << frame_id << (is_keyframe ? "_KF" : "") << ".png";
        cv::imwrite(ss.str(), vis);
    }
};

} // namespace PhotoSLAM
```

### 7.2 Unit Tests

#### File: `tests/test_keyframe_selector.cpp`

```cpp
#include <gtest/gtest.h>
#include "gaussian_keyframe_selector.h"
#include "gaussian_cloud.h"

using namespace PhotoSLAM;

class KeyframeSelectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        KeyframeSelectorParams params;
        params.coverage_threshold = 0.7f;
        params.uncertainty_threshold = 0.3f;
        selector = std::make_shared<GaussianKeyframeSelector>(params);
        
        // Create test Gaussian cloud
        gaussians = std::make_shared<GaussianCloud>();
    }
    
    std::shared_ptr<GaussianKeyframeSelector> selector;
    std::shared_ptr<GaussianCloud> gaussians;
};

TEST_F(KeyframeSelectorTest, EmptyGaussiansNeedKeyframe) {
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC1);
    Sophus::SE3f Tcw;  // Identity
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0, 0) = 500; K(1, 1) = 500;  // fx, fy
    K(0, 2) = 320; K(1, 2) = 240;  // cx, cy
    
    KeyframeMetrics metrics = selector->ComputeMetrics(image, Tcw, gaussians, K);
    
    EXPECT_FLOAT_EQ(metrics.coverage, 0.0f);
    EXPECT_FLOAT_EQ(metrics.uncertainty, 1.0f);
    EXPECT_TRUE(selector->NeedKeyframeGaussian(metrics));
}

TEST_F(KeyframeSelectorTest, FullCoverageNoKeyframe) {
    // Add Gaussians covering entire image
    for (float x = -1.0f; x <= 1.0f; x += 0.1f) {
        for (float y = -1.0f; y <= 1.0f; y += 0.1f) {
            gaussians->AddGaussian(
                Eigen::Vector3f(x, y, 2.0f),  // position
                Eigen::Vector3f(0.01f, 0.01f, 0.01f),  // scale
                1.0f  // opacity
            );
        }
    }
    
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC1);
    Sophus::SE3f Tcw;
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0, 0) = 500; K(1, 1) = 500;
    K(0, 2) = 320; K(1, 2) = 240;
    
    KeyframeMetrics metrics = selector->ComputeMetrics(image, Tcw, gaussians, K);
    
    EXPECT_GT(metrics.coverage, 0.7f);
    EXPECT_FALSE(selector->NeedKeyframeGaussian(metrics));
}

TEST_F(KeyframeSelectorTest, PartialCoverageNeedKeyframe) {
    // Add Gaussians only in left half
    for (float x = -1.0f; x <= 0.0f; x += 0.1f) {
        for (float y = -1.0f; y <= 1.0f; y += 0.1f) {
            gaussians->AddGaussian(
                Eigen::Vector3f(x, y, 2.0f),
                Eigen::Vector3f(0.01f, 0.01f, 0.01f),
                1.0f
            );
        }
    }
    
    cv::Mat image = cv::Mat::zeros(480, 640, CV_8UC1);
    Sophus::SE3f Tcw;
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();
    K(0, 0) = 500; K(1, 1) = 500;
    K(0, 2) = 320; K(1, 2) = 240;
    
    KeyframeMetrics metrics = selector->ComputeMetrics(image, Tcw, gaussians, K);
    
    EXPECT_LT(metrics.coverage, 0.7f);
    EXPECT_TRUE(selector->NeedKeyframeGaussian(metrics));
}
```

### 7.3 Logging & Profiling

```cpp
// Add to Tracking.cc for detailed logging
void Tracking::LogKeyframeDecision(
    int frame_id,
    const PhotoSLAM::KeyframeMetrics& metrics,
    bool c1, bool c2, bool c3, bool c4, bool c5,
    bool final_decision)
{
    static std::ofstream log_file("keyframe_decisions.csv");
    static bool header_written = false;
    
    if (!header_written) {
        log_file << "frame_id,coverage,uncertainty,visible_gaussians,total_gaussians,"
                 << "c1_max_frames,c2_low_overlap,c3_low_coverage,c4_high_uncertainty,"
                 << "c5_new_region,is_keyframe" << std::endl;
        header_written = true;
    }
    
    log_file << frame_id << ","
             << metrics.coverage << ","
             << metrics.uncertainty << ","
             << metrics.visible_gaussians << ","
             << metrics.total_gaussians << ","
             << c1 << "," << c2 << "," << c3 << "," << c4 << "," << c5 << ","
             << final_decision << std::endl;
}
```

---

## 8. Best Practices

### 8.1 Dos

✅ **Test incrementally**: Modify một component, test kỹ trước khi tiếp tục

✅ **Log extensively**: Ghi log đầy đủ để debug

✅ **Use config files**: Không hardcode parameters

✅ **Profile performance**: Đo thời gian computation của mỗi metric

✅ **Visualize results**: Tạo visualization để verify behavior

### 8.2 Don'ts

❌ **Over-engineer**: Bắt đầu đơn giản, thêm complexity khi cần

❌ **Ignore existing conditions**: Gaussian conditions nên bổ sung, không thay thế hoàn toàn ORB-SLAM3 conditions

❌ **Forget edge cases**: Handle trường hợp không có Gaussians, camera behind scene, etc.

❌ **Skip testing**: Mỗi modification cần unit tests

### 8.3 Performance Tips

```cpp
// 1. Cache frequently accessed data
class CachedGaussianData {
    std::vector<Eigen::Vector3f> positions;
    std::vector<Eigen::Vector3f> scales;
    bool is_valid = false;
    
    void Update(const std::shared_ptr<GaussianCloud>& gaussians) {
        // Only update when needed
        if (gaussians->GetLastModifiedTime() > last_update_time) {
            // Refresh cache
        }
    }
};

// 2. Use spatial data structures for faster queries
#include <nanoflann.hpp>
// Build KD-tree for Gaussian positions

// 3. Skip computation for distant Gaussians
const float MAX_DISTANCE = 10.0f;  // meters
if (distance > MAX_DISTANCE) continue;

// 4. Reduce grid resolution for fast approximate coverage
int fast_grid_size = 64;  // Coarser grid for initial check
```

---

## Appendix A: Troubleshooting

| Problem | Possible Cause | Solution |
|---------|---------------|----------|
| Too many keyframes | Coverage threshold too high | Reduce `CoverageThreshold` to 0.6 |
| Too few keyframes | Coverage threshold too low | Increase `CoverageThreshold` to 0.8 |
| Slow processing | Grid too fine | Increase `GridSize` to 64 |
| Holes in reconstruction | Missing keyframes in certain views | Lower `UncertaintyThreshold` |
| Redundant keyframes | Temporal constraint too weak | Increase `MinKeyframeInterval` |

## Appendix B: References

1. ORB-SLAM3: https://github.com/UZ-SLAMLab/ORB_SLAM3
2. Photo-SLAM: https://github.com/HuajianUP/Photo-SLAM
3. 3D Gaussian Splatting: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

---

*Document generated for Photo-SLAM keyframe selection modification guide.*
