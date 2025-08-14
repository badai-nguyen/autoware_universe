
#ifndef AUTOWARE__CUDA_POINTCLOUD_PREPROCESSOR__CUDA_GROUND_SEGMENTATION__CUDA_SCAN_GROUND_SEGMENTATION_FILTER_HPP_
#define AUTOWARE__CUDA_POINTCLOUD_PREPROCESSOR__CUDA_GROUND_SEGMENTATION__CUDA_SCAN_GROUND_SEGMENTATION_FILTER_HPP_

#include <autoware/cuda_pointcloud_preprocessor/point_types.hpp>
#include <autoware/cuda_utils/cuda_check_error.hpp>
#include <autoware_vehicle_info_utils/vehicle_info.hpp>
#include <cuda_blackboard/cuda_pointcloud2.hpp>
#include <cuda_blackboard/cuda_unique_ptr.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <memory>

namespace autoware::cuda_pointcloud_preprocessor
{

using autoware::vehicle_info_utils::VehicleInfo;
struct PointTypeStruct
{
  float x;
  float y;
  float z;
  std::uint8_t intensity;
  std::uint8_t return_type;
  std::uint16_t channel;
};

struct PointsCenteroid
{
  float radius_avg;
  float height_avg;
  float height_max;
  float height_min;
  uint16_t grid_id;
  std::vector<size_t> point_indices;
  std::vector<float> height_list;
  std::vector<float> radius_list;
  
}

// structure to hold parameter values
struct FilterParameters
{
  uint16_t gnd_grid_continual_thresh;
  float non_ground_height_threshold;
  float low_priority_region_x;
  float center_pcl_shift{0.0f};  // virtual center of pcl to center mass

  // common parameters
  float radial_divider_angle_rad;  // distance in rads between dividers
  size_t radial_dividers_num;
  VehicleInfo vehicle_info;

  // common thresholds
  float global_slope_max_angle_rad;  // radians
  float local_slope_max_angle_rad;   // radians
  float global_slope_max_ratio;
  float local_slope_max_ratio;
  float split_points_distance_tolerance;  // distance in meters between concentric divisions

  // non-grid mode parameters
  bool use_virtual_ground_point;
  float split_height_distance;  // minimum height threshold regardless the slope,
                                // useful for close points

  // grid mode parameters
  bool use_recheck_ground_cluster;  // to enable recheck ground cluster
  float recheck_start_distance;     // distance to start rechecking ground cluster
  bool use_lowest_point;  // to select lowest point for reference in recheck ground cluster,
                          // otherwise select middle point
  float detection_range_z_max;

  // grid parameters
  float grid_size_m;
  float grid_mode_switch_radius;  // non linear grid size switching distance
  uint16_t gnd_grid_buffer_size;
  float virtual_lidar_z;
};

class CudaScanGroundSegmentationFilter
{
public:
  explicit CudaScanGroundSegmentationFilter(
    const FilterParameters & filter_parameters, const int64_t max_mem_pool_size_in_byte);
  ~CudaScanGroundSegmentationFilter() = default;

  // Method to process the point cloud data and filter ground points
  std::unique_ptr<cuda_blackboard::CudaPointCloud2> classifyPointcloud(
    const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input_points);

  size_t number_input_points_;
  size_t input_pointcloud_step_;
  size_t input_xyzi_offset_[4];

  // Parameters
  FilterParameters filter_parameters_;

private:
  // Internal methods for ground segmentation logic

  template <typename T>
  T * allocateBufferFromPool(size_t num_elements);

  template <typename T>
  void returnBufferToPool(T * buffer);

  void getObstaclePointcloud(
    const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input_points,
    PointTypeStruct * output_points, size_t * num_output_points);
  /*
  * This function splits the input point cloud into radial divisions.
  * Each division corresponds to a specific angle range defined by the radial_divider_angle_rad.
  * The points in each division are sorted by their distance from the center of the point cloud.
  * @param input_points The input point cloud data.
  * @param indices_list_dev point to device memory where array of radial division indices will be stored.
  * @note This function assumes that the input point cloud is already allocated in device memory.
  */
  void getRadialDivisions(
    const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input_points,
    int * indices_list_dev);

  cudaStream_t ground_segment_stream_{};
  cudaMemPool_t mem_pool_{};
};
}  // namespace autoware::cuda_pointcloud_preprocessor

#endif  // AUTOWARE__CUDA_POINTCLOUD_PREPROCESSOR__CUDA_GROUND_SEGMENTATION__CUDA_SCAN_GROUND_SEGMENTATION_FILTER_HPP_
