/**
 * @file gaussian_keyframe_selector.cpp
 * @brief Implementation of Gaussian-aware keyframe selection
 * @author Claude AI
 * @date 11/01/2026
 * @version 1.0
 */

#include "gaussian_keyframe_selector.h"
#include "gaussian_cloud.h"  // Your Gaussian cloud class
#include "KeyFrame.h"        // ORB-SLAM3 KeyFrame

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>

namespace PhotoSLAM {

//==============================================================================
// KeyframeMetrics Implementation
//==============================================================================

std::string KeyframeMetrics::ToString() const {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(3);
    ss << "KeyframeMetrics {"
       << " coverage=" << coverage
       << ", uncertainty=" << uncertainty
       << ", info_gain=" << information_gain
       << ", is_new_region=" << (is_new_region ? "true" : "false")
       << ", visible=" << visible_gaussians << "/" << total_gaussians
       << ", depth=[" << min_depth << ", " << max_depth << "]"
       << ", avg_scale=" << avg_scale
       << " }";
    return ss.str();
}

//==============================================================================
// KeyframeSelectorParams Implementation
//==============================================================================

void KeyframeSelectorParams::LoadFromFile(const std::string& filename) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    
    if (!fs.isOpened()) {
        std::cerr << "[KeyframeSelectorParams] Warning: Cannot open " << filename 
                  << ", using default values." << std::endl;
        return;
    }
    
    // Coverage params
    if (!fs["Gaussian.CoverageThreshold"].empty())
        fs["Gaussian.CoverageThreshold"] >> coverage_threshold;
    if (!fs["Gaussian.GridSize"].empty())
        fs["Gaussian.GridSize"] >> grid_size;
    if (!fs["Gaussian.MinGaussiansPerCell"].empty())
        fs["Gaussian.MinGaussiansPerCell"] >> min_gaussians_per_cell;
    
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
    if (!fs["Gaussian.AngleScoreSigma"].empty())
        fs["Gaussian.AngleScoreSigma"] >> angle_score_sigma;
    
    // Temporal params
    if (!fs["Gaussian.MinKeyframeInterval"].empty())
        fs["Gaussian.MinKeyframeInterval"] >> min_frames_between_kf;
    if (!fs["Gaussian.MaxKeyframeInterval"].empty())
        fs["Gaussian.MaxKeyframeInterval"] >> max_frames_between_kf;
    
    // Visibility params
    if (!fs["Gaussian.MinDepth"].empty())
        fs["Gaussian.MinDepth"] >> min_depth;
    if (!fs["Gaussian.MaxDepth"].empty())
        fs["Gaussian.MaxDepth"] >> max_depth;
    
    fs.release();
    
    std::cout << "[KeyframeSelectorParams] Loaded from " << filename << std::endl;
    Print();
}

void KeyframeSelectorParams::SaveToFile(const std::string& filename) const {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    
    if (!fs.isOpened()) {
        std::cerr << "[KeyframeSelectorParams] Error: Cannot write to " << filename << std::endl;
        return;
    }
    
    fs << "Gaussian.CoverageThreshold" << coverage_threshold;
    fs << "Gaussian.GridSize" << grid_size;
    fs << "Gaussian.MinGaussiansPerCell" << min_gaussians_per_cell;
    
    fs << "Gaussian.UncertaintyThreshold" << uncertainty_threshold;
    fs << "Gaussian.DistanceWeight" << distance_weight;
    fs << "Gaussian.ScaleWeight" << scale_weight;
    fs << "Gaussian.AngleWeight" << angle_weight;
    
    fs << "Gaussian.InfoGainThreshold" << info_gain_threshold;
    fs << "Gaussian.OptimalBaselineAngle" << optimal_baseline_angle;
    fs << "Gaussian.AngleScoreSigma" << angle_score_sigma;
    
    fs << "Gaussian.MinKeyframeInterval" << min_frames_between_kf;
    fs << "Gaussian.MaxKeyframeInterval" << max_frames_between_kf;
    
    fs << "Gaussian.MinDepth" << min_depth;
    fs << "Gaussian.MaxDepth" << max_depth;
    
    fs.release();
    
    std::cout << "[KeyframeSelectorParams] Saved to " << filename << std::endl;
}

void KeyframeSelectorParams::Print() const {
    std::cout << "========== KeyframeSelectorParams ==========" << std::endl;
    std::cout << "Coverage:" << std::endl;
    std::cout << "  threshold     = " << coverage_threshold << std::endl;
    std::cout << "  grid_size     = " << grid_size << std::endl;
    std::cout << "  min_per_cell  = " << min_gaussians_per_cell << std::endl;
    std::cout << "Uncertainty:" << std::endl;
    std::cout << "  threshold     = " << uncertainty_threshold << std::endl;
    std::cout << "  dist_weight   = " << distance_weight << std::endl;
    std::cout << "  scale_weight  = " << scale_weight << std::endl;
    std::cout << "  angle_weight  = " << angle_weight << std::endl;
    std::cout << "Info Gain:" << std::endl;
    std::cout << "  threshold     = " << info_gain_threshold << std::endl;
    std::cout << "  baseline_ang  = " << optimal_baseline_angle << " deg" << std::endl;
    std::cout << "Temporal:" << std::endl;
    std::cout << "  min_interval  = " << min_frames_between_kf << std::endl;
    std::cout << "  max_interval  = " << max_frames_between_kf << std::endl;
    std::cout << "Visibility:" << std::endl;
    std::cout << "  depth_range   = [" << min_depth << ", " << max_depth << "] m" << std::endl;
    std::cout << "=============================================" << std::endl;
}

//==============================================================================
// GaussianKeyframeSelector Implementation
//==============================================================================

GaussianKeyframeSelector::GaussianKeyframeSelector(const KeyframeSelectorParams& params)
    : mParams(params)
    , mbVerbose(false)
{
}

KeyframeMetrics GaussianKeyframeSelector::ComputeMetrics(
    const cv::Mat& image,
    const Sophus::SE3f& Tcw,
    const std::shared_ptr<GaussianCloud>& gaussians,
    const Eigen::Matrix3f& K)
{
    KeyframeMetrics metrics;
    
    // Handle empty Gaussian cloud
    if (!gaussians || gaussians->Size() == 0) {
        metrics.coverage = 0.0f;
        metrics.uncertainty = 1.0f;
        metrics.information_gain = 1.0f;
        metrics.is_new_region = true;
        metrics.visible_gaussians = 0;
        metrics.total_gaussians = 0;
        
        if (mbVerbose) {
            std::cout << "[GaussianKeyframeSelector] No Gaussians - need keyframe" << std::endl;
        }
        
        return metrics;
    }
    
    metrics.total_gaussians = static_cast<int>(gaussians->Size());
    
    // Compute coverage
    metrics.coverage = ComputeCoverage(image, Tcw, gaussians, K);
    
    // Compute uncertainty
    metrics.uncertainty = ComputeUncertainty(Tcw, gaussians);
    
    // Determine if new region
    metrics.is_new_region = (metrics.coverage < 0.3f);
    
    // Information gain requires existing keyframes - set to 0 for now
    // This should be computed separately with access to keyframe list
    metrics.information_gain = 0.0f;
    
    if (mbVerbose) {
        std::cout << "[GaussianKeyframeSelector] " << metrics.ToString() << std::endl;
    }
    
    return metrics;
}

bool GaussianKeyframeSelector::NeedKeyframeGaussian(const KeyframeMetrics& metrics) const {
    // Condition 1: Low coverage - significant portion of image not covered
    bool low_coverage = (metrics.coverage < mParams.coverage_threshold);
    
    // Condition 2: High uncertainty - rendering quality would be poor
    bool high_uncertainty = (metrics.uncertainty > mParams.uncertainty_threshold);
    
    // Condition 3: New region - mostly unseen area
    bool new_region = metrics.is_new_region;
    
    // Condition 4: High information gain - view would contribute significantly
    bool high_info_gain = (metrics.information_gain > mParams.info_gain_threshold);
    
    bool need_keyframe = low_coverage || high_uncertainty || new_region || high_info_gain;
    
    if (mbVerbose && need_keyframe) {
        std::cout << "[GaussianKeyframeSelector] Need keyframe: "
                  << "low_coverage=" << low_coverage
                  << ", high_uncertainty=" << high_uncertainty
                  << ", new_region=" << new_region
                  << ", high_info_gain=" << high_info_gain
                  << std::endl;
    }
    
    return need_keyframe;
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
    
    const int width = image.cols;
    const int height = image.rows;
    
    // Create coverage grid
    const int grid_cols = (width + mParams.grid_size - 1) / mParams.grid_size;
    const int grid_rows = (height + mParams.grid_size - 1) / mParams.grid_size;
    std::vector<int> grid_counts(grid_cols * grid_rows, 0);
    
    // Camera intrinsics
    const float fx = K(0, 0);
    const float fy = K(1, 1);
    const float cx = K(0, 2);
    const float cy = K(1, 2);
    
    int visible_count = 0;
    float total_scale = 0.0f;
    
    // Project each Gaussian
    for (size_t i = 0; i < gaussians->Size(); ++i) {
        Eigen::Vector3f pw = gaussians->GetPosition(i);
        
        // Transform to camera frame
        Eigen::Vector3f pc = Tcw * pw;
        
        // Depth check
        if (pc.z() < mParams.min_depth || pc.z() > mParams.max_depth) {
            continue;
        }
        
        // Project to image plane
        const float inv_z = 1.0f / pc.z();
        const float u = fx * pc.x() * inv_z + cx;
        const float v = fy * pc.y() * inv_z + cy;
        
        // Bounds check
        if (u < 0 || u >= width || v < 0 || v >= height) {
            continue;
        }
        
        visible_count++;
        
        // Get Gaussian scale for coverage radius
        Eigen::Vector3f scale = gaussians->GetScale(i);
        float avg_scale = scale.mean();
        total_scale += avg_scale;
        
        // Compute projected radius (approximate)
        float projected_radius = (fx * avg_scale) * inv_z;
        int cell_radius = std::max(1, static_cast<int>(projected_radius / mParams.grid_size));
        
        // Mark cells within radius as covered
        int center_gx = static_cast<int>(u) / mParams.grid_size;
        int center_gy = static_cast<int>(v) / mParams.grid_size;
        
        for (int dy = -cell_radius; dy <= cell_radius; ++dy) {
            for (int dx = -cell_radius; dx <= cell_radius; ++dx) {
                int gx = center_gx + dx;
                int gy = center_gy + dy;
                
                if (gx >= 0 && gx < grid_cols && gy >= 0 && gy < grid_rows) {
                    // Check if within circular radius
                    float dist_sq = dx * dx + dy * dy;
                    if (dist_sq <= cell_radius * cell_radius) {
                        grid_counts[gy * grid_cols + gx]++;
                    }
                }
            }
        }
    }
    
    // Count covered cells (cells with enough Gaussians)
    int covered_cells = 0;
    for (int count : grid_counts) {
        if (count >= mParams.min_gaussians_per_cell) {
            covered_cells++;
        }
    }
    
    float coverage = static_cast<float>(covered_cells) / grid_counts.size();
    
    if (mbVerbose) {
        std::cout << "[ComputeCoverage] visible=" << visible_count
                  << "/" << gaussians->Size()
                  << ", covered_cells=" << covered_cells
                  << "/" << grid_counts.size()
                  << ", coverage=" << coverage
                  << std::endl;
    }
    
    return coverage;
}

float GaussianKeyframeSelector::ComputeUncertainty(
    const Sophus::SE3f& Tcw,
    const std::shared_ptr<GaussianCloud>& gaussians)
{
    if (!gaussians || gaussians->Size() == 0) {
        return 1.0f;  // Maximum uncertainty
    }
    
    // Get camera center in world coordinates
    Eigen::Vector3f camera_center = Tcw.inverse().translation();
    
    float total_uncertainty = 0.0f;
    int count = 0;
    
    for (size_t i = 0; i < gaussians->Size(); ++i) {
        Eigen::Vector3f pos = gaussians->GetPosition(i);
        Eigen::Vector3f scale = gaussians->GetScale(i);
        
        // Check if Gaussian is in front of camera
        Eigen::Vector3f pc = Tcw * pos;
        if (pc.z() < mParams.min_depth || pc.z() > mParams.max_depth) {
            continue;
        }
        
        // Compute uncertainty for this Gaussian
        float gaussian_unc = ComputeGaussianUncertainty(pos, scale, camera_center);
        total_uncertainty += gaussian_unc;
        count++;
    }
    
    if (count == 0) {
        return 1.0f;
    }
    
    // Normalize uncertainty
    float avg_uncertainty = total_uncertainty / count;
    
    // Clamp to reasonable range [0, 1]
    return std::clamp(avg_uncertainty, 0.0f, 1.0f);
}

float GaussianKeyframeSelector::ComputeInformationGain(
    const Sophus::SE3f& Tcw,
    const std::shared_ptr<GaussianCloud>& gaussians,
    const std::vector<std::shared_ptr<KeyFrame>>& existing_keyframes)
{
    if (!gaussians || gaussians->Size() == 0) {
        return 1.0f;  // High info gain if no Gaussians exist
    }
    
    if (existing_keyframes.empty()) {
        return 1.0f;  // High info gain if no keyframes exist
    }
    
    Eigen::Vector3f cam_pos = Tcw.inverse().translation();
    
    float total_info_gain = 0.0f;
    int count = 0;
    
    // Convert optimal angle to radians
    const float optimal_angle_rad = mParams.optimal_baseline_angle * M_PI / 180.0f;
    
    for (size_t i = 0; i < gaussians->Size(); ++i) {
        Eigen::Vector3f gaussian_pos = gaussians->GetPosition(i);
        
        // Check visibility
        Eigen::Vector3f pc = Tcw * gaussian_pos;
        if (pc.z() < mParams.min_depth || pc.z() > mParams.max_depth) {
            continue;
        }
        
        // Find minimum angle difference from existing keyframes
        float min_angle_diff = M_PI;  // Maximum possible
        
        for (const auto& kf : existing_keyframes) {
            if (!kf) continue;
            
            Eigen::Vector3f kf_pos = kf->GetCameraCenter();
            float angle_diff = ComputeAngleDifference(gaussian_pos, cam_pos, kf_pos);
            min_angle_diff = std::min(min_angle_diff, angle_diff);
        }
        
        // Information gain scoring
        // Higher when angle is close to optimal baseline
        float angle_deviation = min_angle_diff - optimal_angle_rad;
        float angle_score = std::exp(
            -angle_deviation * angle_deviation / 
            (2.0f * mParams.angle_score_sigma * mParams.angle_score_sigma)
        );
        
        // Weight by Gaussian uncertainty (larger scale = more uncertain = more benefit)
        Eigen::Vector3f scale = gaussians->GetScale(i);
        float scale_factor = scale.mean();
        
        total_info_gain += angle_score * scale_factor;
        count++;
    }
    
    if (count == 0) {
        return 0.0f;
    }
    
    return total_info_gain / count;
}

//==============================================================================
// Helper Functions
//==============================================================================

bool GaussianKeyframeSelector::IsGaussianVisible(
    const Eigen::Vector3f& position,
    const Sophus::SE3f& Tcw,
    const Eigen::Matrix3f& K,
    int width,
    int height) const
{
    Eigen::Vector2f projected;
    return ProjectGaussian(position, Tcw, K, width, height, projected);
}

bool GaussianKeyframeSelector::ProjectGaussian(
    const Eigen::Vector3f& position,
    const Sophus::SE3f& Tcw,
    const Eigen::Matrix3f& K,
    int width,
    int height,
    Eigen::Vector2f& projected_point) const
{
    // Transform to camera frame
    Eigen::Vector3f pc = Tcw * position;
    
    // Depth check
    if (pc.z() < mParams.min_depth || pc.z() > mParams.max_depth) {
        return false;
    }
    
    // Project using intrinsics
    const float fx = K(0, 0);
    const float fy = K(1, 1);
    const float cx = K(0, 2);
    const float cy = K(1, 2);
    
    const float inv_z = 1.0f / pc.z();
    projected_point.x() = fx * pc.x() * inv_z + cx;
    projected_point.y() = fy * pc.y() * inv_z + cy;
    
    // Bounds check
    return (projected_point.x() >= 0 && projected_point.x() < width &&
            projected_point.y() >= 0 && projected_point.y() < height);
}

float GaussianKeyframeSelector::ComputeAngleDifference(
    const Eigen::Vector3f& gaussian_pos,
    const Eigen::Vector3f& view1_pos,
    const Eigen::Vector3f& view2_pos) const
{
    // Compute viewing directions
    Eigen::Vector3f dir1 = (gaussian_pos - view1_pos).normalized();
    Eigen::Vector3f dir2 = (gaussian_pos - view2_pos).normalized();
    
    // Compute angle between directions
    float dot = std::clamp(dir1.dot(dir2), -1.0f, 1.0f);
    return std::acos(dot);
}

float GaussianKeyframeSelector::ComputeGaussianUncertainty(
    const Eigen::Vector3f& gaussian_pos,
    const Eigen::Vector3f& gaussian_scale,
    const Eigen::Vector3f& camera_pos) const
{
    // Distance component
    float distance = (gaussian_pos - camera_pos).norm();
    float normalized_distance = distance / mParams.max_depth;  // Normalize by max depth
    float distance_unc = mParams.distance_weight * normalized_distance;
    
    // Scale component (larger Gaussians have higher uncertainty)
    float avg_scale = gaussian_scale.mean();
    float scale_unc = mParams.scale_weight * avg_scale;
    
    // Anisotropy component (elongated Gaussians have higher uncertainty)
    float scale_var = 0.0f;
    float scale_mean = avg_scale;
    for (int j = 0; j < 3; ++j) {
        float diff = gaussian_scale[j] - scale_mean;
        scale_var += diff * diff;
    }
    scale_var /= 3.0f;
    float anisotropy_unc = mParams.angle_weight * std::sqrt(scale_var);
    
    // Combined uncertainty
    return distance_unc + scale_unc + anisotropy_unc;
}

} // namespace PhotoSLAM
