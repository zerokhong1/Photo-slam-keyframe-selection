/**
 * @file keyframe_visualizer.h
 * @brief Visualization tools for keyframe selection debugging
 * @author Claude AI
 * @date 11/01/2026
 */

#ifndef KEYFRAME_VISUALIZER_H
#define KEYFRAME_VISUALIZER_H

#include <opencv2/opencv.hpp>
#include <string>
#include <sstream>
#include <iomanip>

namespace PhotoSLAM {

// Forward declaration
struct KeyframeMetrics;

/**
 * @brief Visualization utilities for keyframe selection debugging
 */
class KeyframeVisualizer {
public:
    /**
     * @brief Draw coverage grid overlay on image
     * 
     * @param image Input image (will be modified)
     * @param grid_coverage Vector of coverage values per grid cell
     * @param grid_cols Number of grid columns
     * @param grid_rows Number of grid rows
     * @param grid_size Size of each grid cell in pixels
     */
    static void DrawCoverageGrid(
        cv::Mat& image,
        const std::vector<float>& grid_coverage,
        int grid_cols,
        int grid_rows,
        int grid_size)
    {
        for (int gy = 0; gy < grid_rows; ++gy) {
            for (int gx = 0; gx < grid_cols; ++gx) {
                int idx = gy * grid_cols + gx;
                float coverage = grid_coverage[idx];
                
                // Color based on coverage (red = low, green = high)
                int r = static_cast<int>((1.0f - coverage) * 255);
                int g = static_cast<int>(coverage * 255);
                cv::Scalar color(0, g, r);
                
                // Draw rectangle
                cv::Point tl(gx * grid_size, gy * grid_size);
                cv::Point br((gx + 1) * grid_size - 1, (gy + 1) * grid_size - 1);
                
                // Semi-transparent overlay
                cv::Mat roi = image(cv::Rect(tl, br));
                cv::Mat overlay(roi.size(), roi.type(), color);
                cv::addWeighted(roi, 0.7, overlay, 0.3, 0, roi);
            }
        }
        
        // Draw grid lines
        for (int i = 1; i < grid_cols; ++i) {
            int x = i * grid_size;
            cv::line(image, cv::Point(x, 0), cv::Point(x, image.rows),
                     cv::Scalar(128, 128, 128), 1);
        }
        for (int i = 1; i < grid_rows; ++i) {
            int y = i * grid_size;
            cv::line(image, cv::Point(0, y), cv::Point(image.cols, y),
                     cv::Scalar(128, 128, 128), 1);
        }
    }

    /**
     * @brief Create visualization of keyframe decision
     * 
     * @param image Input grayscale or color image
     * @param metrics Keyframe metrics
     * @param is_keyframe Whether this frame was selected as keyframe
     * @param grid_size Grid cell size for coverage display
     * @return Visualization image
     */
    static cv::Mat VisualizeKeyframeDecision(
        const cv::Mat& image,
        const KeyframeMetrics& metrics,
        bool is_keyframe,
        int grid_size = 32)
    {
        cv::Mat vis;
        
        // Convert to color if grayscale
        if (image.channels() == 1) {
            cv::cvtColor(image, vis, cv::COLOR_GRAY2BGR);
        } else {
            vis = image.clone();
        }
        
        int width = vis.cols;
        int height = vis.rows;
        
        // Create info panel at top
        int panel_height = 80;
        cv::Mat panel = cv::Mat::zeros(panel_height, width, CV_8UC3);
        
        // Background gradient
        for (int y = 0; y < panel_height; ++y) {
            float alpha = static_cast<float>(y) / panel_height;
            uchar val = static_cast<uchar>(30 + 20 * alpha);
            panel.row(y).setTo(cv::Scalar(val, val, val));
        }
        
        // Text information
        std::stringstream ss;
        ss << std::fixed << std::setprecision(2);
        
        // Line 1: Coverage and Uncertainty
        ss << "Coverage: " << (metrics.coverage * 100) << "%"
           << "  |  Uncertainty: " << metrics.uncertainty;
        cv::putText(panel, ss.str(), cv::Point(10, 25),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
        
        // Line 2: Gaussian counts
        ss.str("");
        ss << "Visible Gaussians: " << metrics.visible_gaussians
           << " / " << metrics.total_gaussians;
        cv::putText(panel, ss.str(), cv::Point(10, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
        
        // Line 3: Info gain
        ss.str("");
        ss << "Info Gain: " << metrics.information_gain
           << "  |  New Region: " << (metrics.is_new_region ? "YES" : "NO");
        cv::putText(panel, ss.str(), cv::Point(10, 75),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(200, 200, 200), 1);
        
        // Keyframe indicator (right side)
        std::string kf_text = is_keyframe ? "KEYFRAME" : "skip";
        cv::Scalar kf_color = is_keyframe ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
        int baseline;
        cv::Size text_size = cv::getTextSize(kf_text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 2, &baseline);
        cv::putText(panel, kf_text, cv::Point(width - text_size.width - 20, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, kf_color, 2);
        
        // Combine panel and image
        cv::Mat combined;
        cv::vconcat(panel, vis, combined);
        
        // Draw border based on decision
        cv::rectangle(combined, cv::Point(0, 0), 
                      cv::Point(combined.cols - 1, combined.rows - 1),
                      kf_color, 3);
        
        return combined;
    }

    /**
     * @brief Draw projected Gaussians on image
     * 
     * @param image Image to draw on (will be modified)
     * @param projected_points Projected 2D positions of Gaussians
     * @param projected_radii Projected radii of Gaussians
     * @param uncertainties Per-Gaussian uncertainty values
     */
    static void DrawProjectedGaussians(
        cv::Mat& image,
        const std::vector<cv::Point2f>& projected_points,
        const std::vector<float>& projected_radii,
        const std::vector<float>& uncertainties)
    {
        for (size_t i = 0; i < projected_points.size(); ++i) {
            const cv::Point2f& pt = projected_points[i];
            float radius = projected_radii[i];
            float uncertainty = uncertainties[i];
            
            // Color based on uncertainty (green = low, red = high)
            int r = static_cast<int>(std::min(uncertainty * 255, 255.0f));
            int g = static_cast<int>(std::max((1.0f - uncertainty) * 255, 0.0f));
            cv::Scalar color(0, g, r);
            
            // Draw circle
            cv::circle(image, pt, static_cast<int>(radius), color, 1);
            
            // Draw center point
            cv::circle(image, pt, 2, color, -1);
        }
    }

    /**
     * @brief Save visualization to file
     * 
     * @param vis Visualization image
     * @param output_dir Output directory
     * @param frame_id Frame ID
     * @param is_keyframe Whether this was a keyframe
     */
    static void SaveVisualization(
        const cv::Mat& vis,
        const std::string& output_dir,
        int frame_id,
        bool is_keyframe)
    {
        std::stringstream ss;
        ss << output_dir << "/frame_" 
           << std::setw(6) << std::setfill('0') << frame_id
           << (is_keyframe ? "_KF" : "") << ".png";
        
        cv::imwrite(ss.str(), vis);
    }

    /**
     * @brief Create side-by-side comparison of multiple frames
     * 
     * @param images Vector of images to compare
     * @param labels Labels for each image
     * @param cols Number of columns in grid
     * @return Combined comparison image
     */
    static cv::Mat CreateComparison(
        const std::vector<cv::Mat>& images,
        const std::vector<std::string>& labels,
        int cols = 2)
    {
        if (images.empty()) return cv::Mat();
        
        int rows = (images.size() + cols - 1) / cols;
        int cell_width = images[0].cols;
        int cell_height = images[0].rows;
        
        cv::Mat result = cv::Mat::zeros(rows * cell_height, cols * cell_width, CV_8UC3);
        
        for (size_t i = 0; i < images.size(); ++i) {
            int r = i / cols;
            int c = i % cols;
            
            cv::Rect roi(c * cell_width, r * cell_height, cell_width, cell_height);
            
            cv::Mat cell;
            if (images[i].channels() == 1) {
                cv::cvtColor(images[i], cell, cv::COLOR_GRAY2BGR);
            } else {
                cell = images[i].clone();
            }
            
            // Add label
            if (i < labels.size()) {
                cv::putText(cell, labels[i], cv::Point(10, 30),
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            }
            
            cell.copyTo(result(roi));
        }
        
        return result;
    }
};

} // namespace PhotoSLAM

#endif // KEYFRAME_VISUALIZER_H
