#ifndef GAUSSIAN_KEYFRAME_SELECTOR_H
#define GAUSSIAN_KEYFRAME_SELECTOR_H

/**
 * @file gaussian_keyframe_selector.h
 * @brief Gaussian-aware keyframe selection for Photo-SLAM
 * @author Claude AI
 * @date 11/01/2026
 * @version 1.0
 */

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <memory>
#include <string>

namespace PhotoSLAM {

// Forward declarations
class GaussianCloud;
class KeyFrame;

/**
 * @brief Gaussian-aware keyframe selection metrics
 * 
 * This struct contains all computed metrics used to decide
 * whether a new keyframe should be created.
 */
struct KeyframeMetrics {
    // Primary metrics
    float coverage;           ///< [0, 1] - ratio of image covered by Gaussians
    float uncertainty;        ///< [0, inf) - rendering uncertainty from this view
    float information_gain;   ///< [0, inf) - potential info gain from this view
    bool is_new_region;       ///< true if mostly unseen area
    
    // Debug/diagnostic info
    int visible_gaussians;    ///< Number of Gaussians visible in current view
    int total_gaussians;      ///< Total number of Gaussians in map
    float min_depth;          ///< Minimum depth of visible Gaussians
    float max_depth;          ///< Maximum depth of visible Gaussians
    float avg_scale;          ///< Average scale of visible Gaussians
    
    // Constructor with defaults
    KeyframeMetrics()
        : coverage(0.0f)
        , uncertainty(1.0f)
        , information_gain(0.0f)
        , is_new_region(true)
        , visible_gaussians(0)
        , total_gaussians(0)
        , min_depth(0.0f)
        , max_depth(0.0f)
        , avg_scale(0.0f)
    {}
    
    // Helper to print metrics
    std::string ToString() const;
};

/**
 * @brief Parameters for keyframe selection
 * 
 * All parameters can be loaded from a YAML configuration file.
 */
struct KeyframeSelectorParams {
    //==========================================================================
    // Coverage parameters
    //==========================================================================
    
    /// Minimum coverage ratio required to skip keyframe creation
    /// If coverage < this value, keyframe will be created
    /// Range: [0.0, 1.0], Default: 0.7
    float coverage_threshold = 0.7f;
    
    /// Grid cell size in pixels for coverage computation
    /// Smaller = more precise but slower
    /// Default: 32
    int grid_size = 32;
    
    /// Minimum number of Gaussians per grid cell to consider it "covered"
    /// Default: 1
    int min_gaussians_per_cell = 1;
    
    //==========================================================================
    // Uncertainty parameters
    //==========================================================================
    
    /// Maximum acceptable uncertainty
    /// If uncertainty > this value, keyframe will be created
    /// Range: [0.0, inf), Default: 0.3
    float uncertainty_threshold = 0.3f;
    
    /// Weight for distance component in uncertainty computation
    float distance_weight = 1.0f;
    
    /// Weight for scale component in uncertainty computation
    float scale_weight = 0.5f;
    
    /// Weight for view angle component in uncertainty computation
    float angle_weight = 0.3f;
    
    //==========================================================================
    // Information gain parameters
    //==========================================================================
    
    /// Minimum information gain to trigger keyframe creation
    /// Range: [0.0, inf), Default: 0.5
    float info_gain_threshold = 0.5f;
    
    /// Optimal baseline angle (degrees) for multi-view reconstruction
    /// Default: 20.0
    float optimal_baseline_angle = 20.0f;
    
    /// Standard deviation for angle score computation
    float angle_score_sigma = 0.1f;
    
    //==========================================================================
    // Temporal constraints
    //==========================================================================
    
    /// Minimum number of frames between keyframes
    int min_frames_between_kf = 5;
    
    /// Maximum number of frames without creating a keyframe
    int max_frames_between_kf = 30;
    
    //==========================================================================
    // Visibility parameters
    //==========================================================================
    
    /// Minimum depth for Gaussian to be considered visible (meters)
    float min_depth = 0.1f;
    
    /// Maximum depth for Gaussian to be considered visible (meters)
    float max_depth = 100.0f;
    
    //==========================================================================
    // Methods
    //==========================================================================
    
    /// Load parameters from YAML file
    void LoadFromFile(const std::string& filename);
    
    /// Save parameters to YAML file
    void SaveToFile(const std::string& filename) const;
    
    /// Print parameters to stdout
    void Print() const;
};

/**
 * @brief Gaussian-aware keyframe selector
 * 
 * This class implements Gaussian-aware keyframe selection criteria
 * to complement ORB-SLAM3's geometric-based selection.
 * 
 * Main features:
 * - Coverage computation: measures how well current view is covered by Gaussians
 * - Uncertainty estimation: estimates rendering quality from current view
 * - Information gain: measures potential contribution of this view
 * 
 * Usage:
 * @code
 * KeyframeSelectorParams params;
 * params.LoadFromFile("config.yaml");
 * 
 * GaussianKeyframeSelector selector(params);
 * 
 * KeyframeMetrics metrics = selector.ComputeMetrics(image, Tcw, gaussians, K);
 * bool need_kf = selector.NeedKeyframeGaussian(metrics);
 * @endcode
 */
class GaussianKeyframeSelector {
public:
    //==========================================================================
    // Constructors & Destructor
    //==========================================================================
    
    /**
     * @brief Construct with parameters
     * @param params Configuration parameters
     */
    explicit GaussianKeyframeSelector(const KeyframeSelectorParams& params);
    
    /**
     * @brief Default destructor
     */
    ~GaussianKeyframeSelector() = default;
    
    // Disable copy
    GaussianKeyframeSelector(const GaussianKeyframeSelector&) = delete;
    GaussianKeyframeSelector& operator=(const GaussianKeyframeSelector&) = delete;
    
    // Enable move
    GaussianKeyframeSelector(GaussianKeyframeSelector&&) = default;
    GaussianKeyframeSelector& operator=(GaussianKeyframeSelector&&) = default;
    
    //==========================================================================
    // Main Interface
    //==========================================================================
    
    /**
     * @brief Compute all metrics for current view
     * 
     * @param image Current grayscale image
     * @param Tcw Camera pose (world to camera transformation)
     * @param gaussians Pointer to Gaussian cloud
     * @param K Camera intrinsic matrix (3x3)
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
     * 
     * @param metrics Pre-computed metrics from ComputeMetrics()
     * @return true if keyframe should be created
     */
    bool NeedKeyframeGaussian(const KeyframeMetrics& metrics) const;
    
    //==========================================================================
    // Individual Metric Computations
    //==========================================================================
    
    /**
     * @brief Compute coverage ratio
     * 
     * Coverage measures the fraction of the image that is covered by
     * projected Gaussians. Uses a grid-based approach for efficiency.
     * 
     * @param image Current image (for dimensions)
     * @param Tcw Camera pose
     * @param gaussians Gaussian cloud
     * @param K Camera intrinsics
     * @return Coverage ratio in [0, 1]
     */
    float ComputeCoverage(
        const cv::Mat& image,
        const Sophus::SE3f& Tcw,
        const std::shared_ptr<GaussianCloud>& gaussians,
        const Eigen::Matrix3f& K
    );
    
    /**
     * @brief Compute rendering uncertainty
     * 
     * Uncertainty estimates how reliable the rendering would be from
     * the current viewpoint, based on distance, Gaussian scale, and
     * view angle relative to training views.
     * 
     * @param Tcw Camera pose
     * @param gaussians Gaussian cloud
     * @return Uncertainty value in [0, inf)
     */
    float ComputeUncertainty(
        const Sophus::SE3f& Tcw,
        const std::shared_ptr<GaussianCloud>& gaussians
    );
    
    /**
     * @brief Compute information gain from this view
     * 
     * Information gain measures how much this viewpoint would contribute
     * to improving the Gaussian reconstruction, based on view diversity
     * relative to existing keyframes.
     * 
     * @param Tcw Camera pose
     * @param gaussians Gaussian cloud
     * @param existing_keyframes List of existing keyframes
     * @return Information gain value in [0, inf)
     */
    float ComputeInformationGain(
        const Sophus::SE3f& Tcw,
        const std::shared_ptr<GaussianCloud>& gaussians,
        const std::vector<std::shared_ptr<KeyFrame>>& existing_keyframes
    );
    
    //==========================================================================
    // Setters & Getters
    //==========================================================================
    
    /// Set parameters
    void SetParams(const KeyframeSelectorParams& params) { mParams = params; }
    
    /// Get current parameters
    const KeyframeSelectorParams& GetParams() const { return mParams; }
    
    /// Enable/disable verbose logging
    void SetVerbose(bool verbose) { mbVerbose = verbose; }
    
    /// Check if verbose mode is enabled
    bool IsVerbose() const { return mbVerbose; }
    
private:
    //==========================================================================
    // Private Members
    //==========================================================================
    
    KeyframeSelectorParams mParams;  ///< Configuration parameters
    bool mbVerbose = false;          ///< Verbose logging flag
    
    //==========================================================================
    // Helper Functions
    //==========================================================================
    
    /**
     * @brief Check if a Gaussian is visible in current view
     */
    bool IsGaussianVisible(
        const Eigen::Vector3f& position,
        const Sophus::SE3f& Tcw,
        const Eigen::Matrix3f& K,
        int width,
        int height
    ) const;
    
    /**
     * @brief Compute angle difference between two viewing directions
     */
    float ComputeAngleDifference(
        const Eigen::Vector3f& gaussian_pos,
        const Eigen::Vector3f& view1_pos,
        const Eigen::Vector3f& view2_pos
    ) const;
    
    /**
     * @brief Compute uncertainty for a single Gaussian
     */
    float ComputeGaussianUncertainty(
        const Eigen::Vector3f& gaussian_pos,
        const Eigen::Vector3f& gaussian_scale,
        const Eigen::Vector3f& camera_pos
    ) const;
    
    /**
     * @brief Project Gaussian to image plane
     * @return true if projection is valid (in front of camera and within image)
     */
    bool ProjectGaussian(
        const Eigen::Vector3f& position,
        const Sophus::SE3f& Tcw,
        const Eigen::Matrix3f& K,
        int width,
        int height,
        Eigen::Vector2f& projected_point
    ) const;
};

} // namespace PhotoSLAM

#endif // GAUSSIAN_KEYFRAME_SELECTOR_H
