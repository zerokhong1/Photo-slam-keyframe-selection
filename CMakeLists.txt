cmake_minimum_required(VERSION 3.16)
project(PhotoSLAM_KeyframeSelector VERSION 1.0.0)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ==============================================================================
# Find Dependencies
# ==============================================================================

# OpenCV
find_package(OpenCV 4.0 REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
message(STATUS "Eigen3 version: ${EIGEN3_VERSION_STRING}")

# Sophus
find_package(Sophus REQUIRED)
message(STATUS "Found Sophus")

# Pangolin (for visualization, optional)
find_package(Pangolin QUIET)
if(Pangolin_FOUND)
    message(STATUS "Pangolin found - visualization enabled")
    add_definitions(-DUSE_PANGOLIN)
endif()

# GTest (for testing, optional)
find_package(GTest QUIET)
if(GTest_FOUND)
    message(STATUS "GTest found - testing enabled")
    enable_testing()
endif()

# ==============================================================================
# Include Directories
# ==============================================================================

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/ORB-SLAM3/include
    ${PROJECT_SOURCE_DIR}/ORB-SLAM3/include/CameraModels
    ${PROJECT_SOURCE_DIR}/ORB-SLAM3/Thirdparty/Sophus
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

# ==============================================================================
# Gaussian Keyframe Selector Library
# ==============================================================================

add_library(gaussian_keyframe_selector SHARED
    src/gaussian_keyframe_selector.cpp
)

target_link_libraries(gaussian_keyframe_selector
    ${OpenCV_LIBS}
    Eigen3::Eigen
    Sophus::Sophus
)

target_include_directories(gaussian_keyframe_selector PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Set library properties
set_target_properties(gaussian_keyframe_selector PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    PUBLIC_HEADER "include/gaussian_keyframe_selector.h;include/keyframe_visualizer.h"
)

# ==============================================================================
# Example/Test Executables
# ==============================================================================

# Simple test executable
add_executable(test_keyframe_selector
    tests/test_keyframe_selector_simple.cpp
)

target_link_libraries(test_keyframe_selector
    gaussian_keyframe_selector
    ${OpenCV_LIBS}
)

# ==============================================================================
# Unit Tests (if GTest found)
# ==============================================================================

if(GTest_FOUND)
    add_executable(unit_tests
        tests/test_keyframe_selector.cpp
    )
    
    target_link_libraries(unit_tests
        gaussian_keyframe_selector
        GTest::gtest
        GTest::gtest_main
        ${OpenCV_LIBS}
    )
    
    add_test(NAME KeyframeSelectorTests COMMAND unit_tests)
endif()

# ==============================================================================
# Installation
# ==============================================================================

install(TARGETS gaussian_keyframe_selector
    EXPORT GaussianKeyframeSelectorTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include/photo_slam
)

install(EXPORT GaussianKeyframeSelectorTargets
    FILE GaussianKeyframeSelectorTargets.cmake
    NAMESPACE PhotoSLAM::
    DESTINATION lib/cmake/GaussianKeyframeSelector
)

# Install config files
install(FILES
    configs/TUM_RGBD_gaussian.yaml
    DESTINATION share/photo_slam/configs
)

# ==============================================================================
# Summary
# ==============================================================================

message(STATUS "")
message(STATUS "=============== Configuration Summary ===============")
message(STATUS "  Project: ${PROJECT_NAME} v${PROJECT_VERSION}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  OpenCV: ${OpenCV_VERSION}")
message(STATUS "  Eigen3: ${EIGEN3_VERSION_STRING}")
message(STATUS "  Pangolin: ${Pangolin_FOUND}")
message(STATUS "  GTest: ${GTest_FOUND}")
message(STATUS "=====================================================")
message(STATUS "")
