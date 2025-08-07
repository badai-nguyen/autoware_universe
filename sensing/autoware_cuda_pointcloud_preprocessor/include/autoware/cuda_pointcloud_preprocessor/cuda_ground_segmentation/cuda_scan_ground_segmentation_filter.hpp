
#ifndef AUTOWARE__CUDA_POINTCLOUD_PREPROCESSOR__CUDA_GROUND_SEGMENTATION__CUDA_SCAN_GROUND_SEGMENTATION_FILTER_HPP_
#define AUTOWARE__CUDA_POINTCLOUD_PREPROCESSOR__CUDA_GROUND_SEGMENTATION__CUDA_SCAN_GROUND_SEGMENTATION_FILTER_HPP_

#include <autoware/cuda_pointcloud_preprocessor/point_types.hpp>
#include <autoware/cuda_utils/cuda_check_error.hpp>
#include <cuda_blackboard/cuda_pointcloud2.hpp>
#include <cuda_blackboard/cuda_unique_ptr.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <memory>

namespace autoware::cuda_pointcloud_preprocessor
{
struct PointTypeStruct
{
  float x;
  float y;
  float z;
  std::uint8_t intensity;
  std::uint8_t return_type;
  std::uint16_t channel;
};

class CudaScanGroundSegmentationFilter
{
public:
  explicit CudaScanGroundSegmentationFilter(
    const double height_threshold, const int64_t max_mem_pool_size_in_byte);
  ~CudaScanGroundSegmentationFilter() = default;

  // Method to process the point cloud data and filter ground points
  std::unique_ptr<cuda_blackboard::CudaPointCloud2> classifyPointcloud(
    const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input_points);

  float height_threshold_;
  size_t number_input_points_;
  size_t input_pointcloud_step_;
  size_t input_xyzi_offset_[4];

private:
  // Internal methods for ground segmentation logic

  template <typename T>
  T * allocateBufferFromPool(size_t num_elements);

  template <typename T>
  void returnBufferToPool(T * buffer);

  void getObstaclePointcloud(
    const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input_points,
    PointTypeStruct * output_points, size_t * num_output_points);

  cudaStream_t ground_segment_stream_{};
  cudaMemPool_t mem_pool_{};
};
}  // namespace autoware::cuda_pointcloud_preprocessor

#endif  // AUTOWARE__CUDA_POINTCLOUD_PREPROCESSOR__CUDA_GROUND_SEGMENTATION__CUDA_SCAN_GROUND_SEGMENTATION_FILTER_HPP_
