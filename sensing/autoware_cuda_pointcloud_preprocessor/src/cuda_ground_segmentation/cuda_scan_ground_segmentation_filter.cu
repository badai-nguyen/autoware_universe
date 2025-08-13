#include "autoware/cuda_pointcloud_preprocessor/cuda_ground_segmentation/cuda_scan_ground_segmentation_filter.hpp"

#include <cub/cub.cuh>

#include <sensor_msgs/msg/point_field.hpp>

#include <cmath>
#include <memory>
#include <optional>

namespace autoware::cuda_pointcloud_preprocessor
{
namespace
{

template <typename T>
__device__ const T getElementValue(
  const uint8_t * data, const size_t point_index, const size_t point_step, const size_t offset)
{
  return *reinterpret_cast<const T *>(data + offset + point_index * point_step);
}

__global__ void markValidKernel(
  const PointTypeStruct * __restrict__ input_points, const size_t num_points, float z_threshold,
  int * __restrict__ flags)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx > num_points) {
    return;
  }
  // Mark the point as valid if its z value is above the height threshold
  flags[idx] = (input_points[idx].z > z_threshold) ? 1 : 0;
}

__global__ void scatterKernel(
  const PointTypeStruct * __restrict__ input_points, const int * __restrict__ flags,
  const int * __restrict__ indices, size_t num_points, PointTypeStruct * __restrict__ output_points)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) {
    return;
  }
  // If the point is valid, copy it to the output points using the indices
  if (flags[idx]) {
    const int output_index = indices[idx];
    output_points[output_index] = input_points[idx];
  }
}

}  // namespace

CudaScanGroundSegmentationFilter::CudaScanGroundSegmentationFilter(
  const FilterParameters & filter_parameters, const int64_t max_mem_pool_size_in_byte)
: filter_parameters_(filter_parameters)
{
  CHECK_CUDA_ERROR(cudaStreamCreate(&ground_segment_stream_));

  {
    int current_device_id = 0;
    CHECK_CUDA_ERROR(cudaGetDevice(&current_device_id));
    cudaMemPoolProps pool_props = {};
    pool_props.allocType = cudaMemAllocationTypePinned;
    pool_props.location.id = current_device_id;
    pool_props.location.type = cudaMemLocationTypeDevice;

    CHECK_CUDA_ERROR(cudaMemPoolCreate(&mem_pool_, &pool_props));

    uint64_t pool_release_threshold = max_mem_pool_size_in_byte;
    CHECK_CUDA_ERROR(cudaMemPoolSetAttribute(
      mem_pool_, cudaMemPoolAttrReleaseThreshold, static_cast<void *>(&pool_release_threshold)));
  }
}

std::unique_ptr<cuda_blackboard::CudaPointCloud2>
CudaScanGroundSegmentationFilter::classifyPointcloud(
  const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input_points)
{
  number_input_points_ = input_points->width * input_points->height;
  input_pointcloud_step_ = input_points->point_step;
  const size_t max_bytes = number_input_points_ * sizeof(PointTypeStruct);

  auto filtered_output = std::make_unique<cuda_blackboard::CudaPointCloud2>();
  filtered_output->data = cuda_blackboard::make_unique<std::uint8_t[]>(max_bytes);

  auto * output_points_dev = reinterpret_cast<PointTypeStruct *>(filtered_output->data.get());
  size_t num_output_points = 0;

  getObstaclePointcloud(input_points, output_points_dev, &num_output_points);

  filtered_output->header = input_points->header;
  filtered_output->height = 1;  // Set height to 1 for unorganized point cloud
  filtered_output->width = static_cast<uint32_t>(num_output_points);
  filtered_output->is_bigendian = input_points->is_bigendian;
  filtered_output->point_step = input_points->point_step;
  filtered_output->row_step = static_cast<uint32_t>(num_output_points * sizeof(PointTypeStruct));
  filtered_output->is_dense = input_points->is_dense;
  filtered_output->fields = input_points->fields;

  return filtered_output;
}

template <typename T>
T * CudaScanGroundSegmentationFilter::allocateBufferFromPool(size_t num_elements)
{
  T * buffer{};
  CHECK_CUDA_ERROR(
    cudaMallocFromPoolAsync(&buffer, num_elements * sizeof(T), mem_pool_, ground_segment_stream_));
  CHECK_CUDA_ERROR(cudaMemsetAsync(buffer, 0, num_elements * sizeof(T), ground_segment_stream_));

  return buffer;
}

template <typename T>
void CudaScanGroundSegmentationFilter::returnBufferToPool(T * buffer)
{
  // Return (but not actual) working buffer to the pool
  CHECK_CUDA_ERROR(cudaFreeAsync(buffer, ground_segment_stream_));
}

void CudaScanGroundSegmentationFilter::getObstaclePointcloud(
  const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input_points,
  PointTypeStruct * output_points_dev, size_t * num_output_points_host)
{
  const size_t n = number_input_points_;
  if (n == 0) {
    *num_output_points_host = 0;
    return;
  }

  const auto * input_points_dev =
    reinterpret_cast<const PointTypeStruct *>(input_points->data.get());

  auto * flag_dev =
    allocateBufferFromPool<int>(number_input_points_);  // Buffer to hold flags for each point
  auto * indices_dev = allocateBufferFromPool<int>(
    number_input_points_);  // Buffer to hold indices of non-ground points

  void * temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  // Implement the logic to filter out ground points based on height_threshold_
  // and fill output_points with the coordinates of non-ground points.
  // This is a placeholder for the actual CUDA kernel call.
  dim3 block_dim(512);
  dim3 grid_dim((number_input_points_ + block_dim.x - 1) / block_dim.x);

  markValidKernel<<<grid_dim, block_dim, 0, ground_segment_stream_>>>(
    input_points_dev, n, filter_parameters_.non_ground_height_threshold, flag_dev);

  cub::DeviceScan::ExclusiveSum(
    temp_storage, temp_storage_bytes, flag_dev, indices_dev, static_cast<int>(n),
    ground_segment_stream_);
  CHECK_CUDA_ERROR(
    cudaMallocFromPoolAsync(&temp_storage, temp_storage_bytes, mem_pool_, ground_segment_stream_));
  cub::DeviceScan::ExclusiveSum(
    temp_storage, temp_storage_bytes, flag_dev, indices_dev, static_cast<int>(n),
    ground_segment_stream_);

  scatterKernel<<<grid_dim, block_dim, 0, ground_segment_stream_>>>(
    input_points_dev, flag_dev, indices_dev, n, output_points_dev);

  // Count the number of valid points
  int last_index = 0;
  int last_flag = 0;
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    &last_index, indices_dev + n - 1, sizeof(int), cudaMemcpyDeviceToHost, ground_segment_stream_));
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    &last_flag, flag_dev + n - 1, sizeof(int), cudaMemcpyDeviceToHost, ground_segment_stream_));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(ground_segment_stream_));

  const size_t num_output_points = static_cast<size_t>(last_flag + last_index);
  *num_output_points_host = num_output_points;

  if (temp_storage) {
    CHECK_CUDA_ERROR(cudaFreeAsync(temp_storage, ground_segment_stream_));
  }
  returnBufferToPool(flag_dev);
  returnBufferToPool(indices_dev);
}

}  // namespace autoware::cuda_pointcloud_preprocessor
