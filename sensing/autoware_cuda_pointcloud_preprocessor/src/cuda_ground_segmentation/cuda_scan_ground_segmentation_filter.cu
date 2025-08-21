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
__device__ __forceinline__ float fastAtan2_0_2Pi(float y, float x)
{
  const float PI = 3.14159265358979323846f;
  const float PI2 = 6.28318530717958647692f;
  const float PI_2 = 1.57079632679489661923f;

  // Avoid divide-by-zero
  float abs_y = fabsf(y) + 1e-10f;

  float r, angle;

  if (x >= 0.0f) {
    // First and fourth quadrants
    r = (x - abs_y) / (x + abs_y);
    angle = PI_2 - r * (0.2447f + 0.0663f * fabsf(r));  // polynomial approx
    if (y < 0.0f) {
      angle = PI2 - angle;  // 4th quadrant
    }
  } else {
    // Second and third quadrants
    r = (x + abs_y) / (abs_y - x);
    angle = 3.0f * PI_2 - r * (0.2447f + 0.0663f * fabsf(r));  // poly approx
    if (y < 0.0f) {
      angle = angle - PI2;  // wrap into [0, 2π]
    }
  }

  // Ensure within [0, 2π]
  if (angle < 0.0f) angle += PI2;
  return angle;
}
__device__ inline int getCellID(
  const PointTypeStruct & point, const float center_x, const float center_y,
  const float inv_sector_angle_rad, const float cell_size_m, const int max_num_cells_per_sector,
  const int max_num_cells)
{
  const float dx = point.x - center_x;
  const float dy = point.y - center_y;
  const float radius = sqrtf(dx * dx + dy * dy);
  const float angle = fastAtan2_0_2Pi(dy, dx);
  // Determine the sector index
  int division_sector_index = static_cast<int>(angle * inv_sector_angle_rad);

  // Determine the radial cell index

  int cell_index_in_sector = static_cast<int>(radius / cell_size_m);

  // combine to get unique cell ID

  int cell_id = division_sector_index * max_num_cells_per_sector + cell_index_in_sector;

  // clamp invalid values

  if (cell_id < 0 || cell_id >= max_num_cells) {
    return -1;
  }
  return cell_id;
}

__global__ void initPoints(ClassifiedPointTypeStruct * arr, int N)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;

  arr[idx].x = 0.0f;
  arr[idx].y = 0.0f;
  arr[idx].z = 0.0f;
  arr[idx].intensity = 0;
  arr[idx].return_type = 0;
  arr[idx].channel = 0;
  arr[idx].type = PointType::INIT;
  arr[idx].radius = -1.0f;
  arr[idx].origin_index = 0;
}

__global__ void setFlagsKernel(int * flags, int n, const int value)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) flags[i] = value;  // write real int 0 or 1
}

__global__ void getCellNumPointsKernel(
  const CellCentroid * __restrict__ cells_centroid_list_dev, const size_t num_cells,
  int * __restrict__ num_points_per_cell)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_cells) {
    return;  // Out of bounds
  }
  num_points_per_cell[idx] = cells_centroid_list_dev[idx].num_points;
}

__global__ void assignPointToClassifyPointKernel(
  const PointTypeStruct * __restrict__ input_points, const size_t num_points,
  const int * num_points_per_cell_dev, const int * cell_start_point_idx,
  const CellCentroid * cells_centroid_dev, int * cell_counts_dev,
  const FilterParameters * filter_parameters_dev, ClassifiedPointTypeStruct * classified_points_dev)
{
  // This kernel split pointcloud into sectors and cells
  // Each point is allocated to a cell
  // The number points in each cells is set as num_points_per_cell_dev
  // The point is allocated to a cell based on its angle and distance from the center
  // This is a placeholder for the actual implementation
  // memory index for classified_points_dev is calculated as

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_points) {
    return;  // Out of bounds
  }
  const auto inv_sector_angle_rad = 1.0f / filter_parameters_dev->sector_angle_rad;
  // Calculate the angle and distance from the center
  const float x = input_points[idx].x - filter_parameters_dev->center_x;
  const float y = input_points[idx].y - filter_parameters_dev->center_y;
  const float radius = sqrtf(x * x + y * y);
  const float angle = fastAtan2_0_2Pi(y, x);  // replace with approximate atan
  // Determine the radial division index
  auto division_sector_index = static_cast<int>(angle * inv_sector_angle_rad);
  auto cell_index_in_sector = static_cast<int>(radius / filter_parameters_dev->cell_divider_size_m);
  auto cell_id =
    division_sector_index * filter_parameters_dev->max_num_cells_per_sector + cell_index_in_sector;
  if (cell_id < 0 || cell_id >= filter_parameters_dev->max_num_cells) {
    return;  // Out of bounds
  }

  // const auto cell_id = getCellID(
  //   input_points[idx], center_x, center_y, inv_sector_angle_rad, cell_size_m,
  //   max_number_cels_per_sector, max_num_cells);
  // if (cell_id < 0) {
  //   return;
  // }

  // calc index of point in classified_point_dev
  // the last index of point in cell is located in cell_counts_dev
  // attomically get slot index in the current cell
  int slot_idx = atomicAdd(&cell_counts_dev[cell_id], 1);
  // check local bounds for slot_idx
  if (slot_idx >= num_points_per_cell_dev[cell_id]) {
    return;  // Out of bounds
  }
  auto classify_point_idx = static_cast<size_t>(cell_start_point_idx[cell_id] + slot_idx);
  // Check overall bounds for classified_points_dev
  if (classify_point_idx >= num_points) {
    return;  // Out of bounds
  }
  // add pointcloud to output grid list
  auto & assign_classified_point_dev = classified_points_dev[classify_point_idx];
  assign_classified_point_dev.x = input_points[idx].x;
  assign_classified_point_dev.y = input_points[idx].y;
  assign_classified_point_dev.z = input_points[idx].z;
  assign_classified_point_dev.intensity = input_points[idx].intensity;
  assign_classified_point_dev.return_type = input_points[idx].return_type;
  assign_classified_point_dev.channel = input_points[idx].channel;
  assign_classified_point_dev.type = PointType::INIT;

  assign_classified_point_dev.radius = radius;
  assign_classified_point_dev.origin_index = idx;  // index in the original point cloud
  // Update the cell centroid
}

__device__ void updateGndPointInCell(
  CellCentroid & cell, const ClassifiedPointTypeStruct & gnd_point)
{
  cell.gnd_avg_z = cell.gnd_avg_z * cell.num_ground_points + gnd_point.z;
  cell.gnd_avg_x = cell.gnd_avg_x * cell.num_ground_points + gnd_point.x;
  cell.gnd_avg_y = cell.gnd_avg_y * cell.num_ground_points + gnd_point.y;
  cell.radius_avg = cell.radius_avg * cell.num_ground_points + gnd_point.radius;

  cell.num_ground_points++;

  cell.gnd_avg_x = cell.gnd_avg_x / cell.num_ground_points;
  cell.gnd_avg_y = cell.gnd_avg_y / cell.num_ground_points;
  cell.gnd_avg_z = cell.gnd_avg_z / cell.num_ground_points;
  cell.radius_avg = cell.radius_avg / cell.num_ground_points;
  // Update the min and max height
  if (gnd_point.z > cell.gnd_max_z) {
    cell.gnd_max_z = gnd_point.z;
  }
  if (gnd_point.z < cell.gnd_min_z) {
    cell.gnd_min_z = gnd_point.z;
  }
}
__device__ void removeGndPointInCell(CellCentroid & cell, const ClassifiedPointTypeStruct & point)
{
  cell.gnd_avg_x =
    (cell.gnd_avg_x * cell.num_ground_points - point.x) / (cell.num_ground_points - 1);
  cell.gnd_avg_y =
    (cell.gnd_avg_y * cell.num_ground_points - point.y) / (cell.num_ground_points - 1);
  cell.gnd_avg_z =
    (cell.gnd_avg_z * cell.num_ground_points - point.z) / (cell.num_ground_points - 1);
  cell.radius_avg =
    (cell.radius_avg * cell.num_ground_points - point.radius) / (cell.num_ground_points - 1);
  cell.num_ground_points--;
}

__device__ void updatePrevCellCentroid(const CellCentroid & current, CellCentroid & previous)
{
  previous.gnd_avg_x = current.gnd_avg_x;
  previous.gnd_avg_y = current.gnd_avg_y;
  previous.gnd_avg_z = current.gnd_avg_z;
}

__device__ void checkSegmentMode(
  const CellCentroid * centroid_cells, const int cell_idx_in_sector, const int sector_start_index,
  const int continues_checking_cell_num, SegmentationMode & mode)
{
  mode = SegmentationMode::UNINITIALIZED;
  if (cell_idx_in_sector == 0) {
    // If this is the first cell in the sector, we need to check the previous cells
    return;
  }
  // UNITIALIZED if all previous cell in the same sector has no ground points
  int prev_cell_id_in_sector = cell_idx_in_sector - 1;
  for (prev_cell_id_in_sector = cell_idx_in_sector - 1; prev_cell_id_in_sector > 0;
       --prev_cell_id_in_sector) {
    // find the latest cell with ground points
    auto prev_cell_in_sector = centroid_cells[sector_start_index + prev_cell_id_in_sector];
    if (prev_cell_in_sector.num_ground_points > 0) {
      break;
    }
  }
  if (prev_cell_id_in_sector == 0) {
    // If no previous cell has ground points, set mode to UNINITIALIZED
    mode = SegmentationMode::UNINITIALIZED;
    return;
  }
  // If previous cell has no points, set mode to BREAK
  if (prev_cell_id_in_sector < cell_idx_in_sector - 1) {
    mode = SegmentationMode::BREAK;
  }
  // if all continuous checking previous cells has points, set mode to CONTINUOUS
  for (int i = cell_idx_in_sector - 1; i > cell_idx_in_sector - continues_checking_cell_num; --i) {
    if (i < 0) {
      mode = SegmentationMode::DISCONTINUOUS;
      return;
    }

    auto check_cell = centroid_cells[sector_start_index + i];
    if (check_cell.num_ground_points <= 0) {
      mode = SegmentationMode::DISCONTINUOUS;
      return;  // If any previous cell has no ground points, set mode to DISCONTINUOUS
    }
    // If all previous cells have ground points, set mode to CONTINUOUS
    mode = SegmentationMode::CONTINUOUS;
    return;
  }
}

__device__ float calcLocalGndGradient(
  const CellCentroid * centroid_cells, const int continues_checking_cell_num,
  const int sector_start_index, const int cell_idx_in_sector, const float gradient_threshold)
{
  // Calculate the local ground gradient based on the previous cells
  if (continues_checking_cell_num < 2) {
    return 0.0f;  // Not enough data to calculate gradient
  }
  auto cell_id = sector_start_index + cell_idx_in_sector;

  float orig_z = centroid_cells[cell_id - continues_checking_cell_num].gnd_avg_z;
  float orig_radius = centroid_cells[cell_id - continues_checking_cell_num].radius_avg;
  float gradient = 0.0f;
  int valid_gradients = 0;

  // Calculate gradients from reference to each valid previous cell
  for (int i = 1; i < continues_checking_cell_num; ++i) {
    const auto & prev_cell = centroid_cells[cell_id - i];
    if (prev_cell.num_ground_points > 0) {
      float dz = prev_cell.gnd_avg_z - orig_z;
      float dr = prev_cell.radius_avg - orig_radius;

      // Avoid division by zero
      if (fabsf(dr) > 1e-6f) {
        gradient += dz / dr;
        valid_gradients++;
      }
    }
  }

  if (valid_gradients == 0) {
    return 0.0f;  // No valid gradients found
  }
  gradient /= valid_gradients;
  gradient = gradient > gradient_threshold ? gradient_threshold : gradient;    // Clamp to threshold
  gradient = gradient < -gradient_threshold ? -gradient_threshold : gradient;  // Clamp to threshold
  return gradient;  // Return average gradient
}

__device__ void recheckCell(
  CellCentroid & cell, ClassifiedPointTypeStruct * classify_points,
  const size_t idx_start_point_of_cell, const size_t num_points_of_cell,
  const FilterParameters * filter_parameters_dev, const int cell_idx)
{
  // This function is called to recheck the current cell
  // It should be implemented based on the specific requirements of the segmentation algorithm
  if (cell.num_ground_points < 2) {
    // If the cell has less than 2 ground points, we can skip rechecking
    return;
  }
  for (int i = idx_start_point_of_cell; i < idx_start_point_of_cell + num_points_of_cell; i++) {
    // Recheck the point
    auto & point = classify_points[i];
    if (point.type != PointType::GROUND) {
      continue;  // Skip non-ground points
    }
    // Apply the rechecking logic
    if (point.z > cell.gnd_min_z + filter_parameters_dev->non_ground_height_threshold) {
      point.type = PointType::NON_GROUND;
      removeGndPointInCell(cell, point);
    }
  }
}

__device__ void SegmentInitializedCell(
  CellCentroid * centroid_cells, ClassifiedPointTypeStruct * classify_points,
  const size_t idx_start_point_of_cell, const size_t num_points_of_cell,
  const FilterParameters * filter_parameters_dev, const int sector_start_cell_index,
  const int cell_idx_in_sector)
{
  auto cell_id = sector_start_cell_index + cell_idx_in_sector;
  auto & current_cell = centroid_cells[cell_id];  // Use reference, not copy
  for (size_t i = 0; i < num_points_of_cell; ++i) {
    auto & point = classify_points[idx_start_point_of_cell + i];
    // 1. height is out-of-range
    if (
      point.z > filter_parameters_dev->detection_range_z_max ||
      point.z < -filter_parameters_dev->non_ground_height_threshold) {
      point.type = PointType::OUT_OF_RANGE;
      continue;  // Skip non-ground points
    }
    if (
      point.z / point.radius > filter_parameters_dev->global_slope_max_ratio &&
      point.z > filter_parameters_dev->non_ground_height_threshold) {
      point.type = PointType::NON_GROUND;
      continue;  // Skip non-ground points
    }
    if (
      abs(point.z / point.radius) < filter_parameters_dev->global_slope_max_ratio &&
      abs(point.z) < filter_parameters_dev->non_ground_height_threshold) {
      // If the point is close to the estimated ground height, classify it as ground
      point.type = PointType::GROUND;  // Mark as ground point
      updateGndPointInCell(current_cell, point);
    }
  }

  if (
    filter_parameters_dev->use_recheck_ground_cluster && current_cell.num_ground_points > 1 &&
    current_cell.radius_avg > filter_parameters_dev->recheck_start_distance) {
    // Recheck the ground points in the cell
    recheckCell(
      current_cell, classify_points, idx_start_point_of_cell, num_points_of_cell,
      filter_parameters_dev, cell_id);
  }
}
__device__ void SegmentContinuousCell(
  CellCentroid * centroid_cells, ClassifiedPointTypeStruct * classify_points,
  const size_t idx_start_point_of_cell, const size_t num_points_of_cell,
  const FilterParameters * filter_parameters_dev, const int sector_start_cell_index,
  const int cell_idx_in_sector)
{
  // compare point of current cell with previous cell center by local slope angle
  auto gnd_gradient = calcLocalGndGradient(
    centroid_cells, filter_parameters_dev->gnd_cell_buffer_size, sector_start_cell_index,
    cell_idx_in_sector, filter_parameters_dev->global_slope_max_ratio);
  int cell_id = sector_start_cell_index + cell_idx_in_sector;
  auto & current_cell = centroid_cells[cell_id];  // Use reference, not copy
  auto & prev_cell = centroid_cells[cell_id - 1];
  for (size_t i = 0; i < num_points_of_cell; ++i) {
    auto & point = classify_points[idx_start_point_of_cell + i];
    // 1. height is out-of-range
    if (point.z > filter_parameters_dev->detection_range_z_max) {
      point.type = PointType::OUT_OF_RANGE;
      return;  // Skip non-ground points
    }

    auto d_radius = point.radius - prev_cell.radius_avg;
    auto dz = point.z - prev_cell.gnd_avg_z;

    // 2. the angle is exceed the global slope threshold
    if (point.z / point.radius > filter_parameters_dev->global_slope_max_ratio) {
      point.type = PointType::NON_GROUND;
      continue;  // Skip non-ground points
    }

    // 2. the angle is exceed the local slope threshold
    if (dz / d_radius > filter_parameters_dev->local_slope_max_ratio) {
      point.type = PointType::NON_GROUND;
      continue;  // Skip non-ground points
    }

    // 3. height from the estimated ground center estimated by local gradient
    float estimated_ground_z =
      prev_cell.gnd_avg_z + gnd_gradient * filter_parameters_dev->cell_divider_size_m;
    if (point.z > estimated_ground_z + filter_parameters_dev->non_ground_height_threshold) {
      point.type = PointType::NON_GROUND;
      continue;  // Skip non-ground points
    }
    // if (abs(point.z - estimated_ground_z) <= filter_parameters_dev->non_ground_height_threshold)
    // {
    //   continue;  // Mark as ground point
    // }

    if (
      point.z < estimated_ground_z - filter_parameters_dev->non_ground_height_threshold ||
      dz / d_radius < -filter_parameters_dev->local_slope_max_ratio ||
      point.z / point.radius < -filter_parameters_dev->global_slope_max_ratio) {
      // If the point is below the estimated ground height, classify it as non-ground
      point.type = PointType::OUT_OF_RANGE;
      continue;  // Skip non-ground points
    }
    // If the point is close to the estimated ground height, classify it as ground
    point.type = PointType::GROUND;
    updateGndPointInCell(current_cell, point);
  }

  if (
    filter_parameters_dev->use_recheck_ground_cluster && current_cell.num_ground_points > 1 &&
    current_cell.radius_avg > filter_parameters_dev->recheck_start_distance) {
    // Recheck the ground points in the cell
    recheckCell(
      current_cell, classify_points, idx_start_point_of_cell, num_points_of_cell,
      filter_parameters_dev, cell_id);
  }
}

__device__ void SegmentDiscontinuousCell(
  CellCentroid * centroid_cells, ClassifiedPointTypeStruct * classify_points,
  const size_t idx_start_point_of_cell, const size_t num_points_of_cell,
  const FilterParameters * filter_parameters_dev, const int sector_start_cell_index,
  const int cell_idx_in_sector)
{
  auto cell_id = sector_start_cell_index + cell_idx_in_sector;
  auto & current_cell = centroid_cells[cell_id];  // Use reference, not copy
  auto & prev_gnd_cell = centroid_cells[cell_id - 1];
  if (prev_gnd_cell.num_ground_points <= 0) {
    return;
  }

  for (int i = 0; i < num_points_of_cell; ++i) {
    auto & point = classify_points[idx_start_point_of_cell + i];
    // 1. height is out-of-range
    if (point.z - prev_gnd_cell.gnd_avg_z > filter_parameters_dev->detection_range_z_max) {
      point.type = PointType::OUT_OF_RANGE;
      continue;  // Skip non-ground points
    }
    // 2. the angle is exceed the global slope threshold
    auto dz = point.z - prev_gnd_cell.gnd_avg_z;
    auto d_radius = point.radius - prev_gnd_cell.radius_avg;

    if (point.z / point.radius > filter_parameters_dev->global_slope_max_ratio) {
      point.type = PointType::NON_GROUND;
      continue;  // Skip non-ground points
    }
    // 3. local slope
    if (dz / d_radius > filter_parameters_dev->local_slope_max_ratio) {
      point.type = PointType::NON_GROUND;
      continue;  // Skip non-ground points
    }
    if (dz / d_radius > filter_parameters_dev->global_slope_max_ratio) {
      // If the point is above the estimated ground height, classify it as non-ground
      point.type = PointType::NON_GROUND;
      continue;  // Skip non-ground points
    }

    if (dz / d_radius < -filter_parameters_dev->local_slope_max_ratio) {
      // If the point is below the estimated ground height, classify it as non-ground
      point.type = PointType::OUT_OF_RANGE;
      continue;  // Skip non-ground points
    }
    if (point.z / point.radius < -filter_parameters_dev->global_slope_max_ratio) {
      // If the point is below the estimated ground height, classify it as non-ground
      point.type = PointType::OUT_OF_RANGE;
      continue;  // Skip non-ground points
    }
    point.type = PointType::GROUND;  // Mark as ground point
    updateGndPointInCell(current_cell, point);
  }

  if (
    filter_parameters_dev->use_recheck_ground_cluster && current_cell.num_ground_points > 1 &&
    current_cell.radius_avg > filter_parameters_dev->recheck_start_distance) {
    // Recheck the ground points in the cell
    recheckCell(
      current_cell, classify_points, idx_start_point_of_cell, num_points_of_cell,
      filter_parameters_dev, cell_id);
  }
}

__device__ void SegmentBreakCell(
  CellCentroid * centroid_cells, ClassifiedPointTypeStruct * classify_points,
  const size_t idx_start_point_of_cell, const size_t num_points_of_cell,
  const FilterParameters * filter_parameters_dev, const int sector_start_cell_index,
  const int cell_idx_in_sector)
{
  // This function is called when the cell is not continuous with the previous cell
  auto cell_id = sector_start_cell_index + cell_idx_in_sector;
  auto & current_cell = centroid_cells[cell_id];  // Use reference, not copy
  int prev_gnd_cell_idx = cell_idx_in_sector - 1;
  for (int i = cell_idx_in_sector - 1; i > 0; --i) {
    // find the latest cell with ground points
    auto & prev_cell = centroid_cells[sector_start_cell_index + i];
    if (prev_cell.num_ground_points > 0) {
      prev_gnd_cell_idx = i;
      break;
    }
  }
  auto & prev_gnd_cell = centroid_cells[sector_start_cell_index + prev_gnd_cell_idx];
  for (int i = 0; i < num_points_of_cell; ++i) {
    auto & point = classify_points[idx_start_point_of_cell + i];
    // 1. height is out-of-range
    // if (point.z - prev_gnd_cell.gnd_avg_z > filter_parameters_dev->detection_range_z_max) {
    //   point.type = PointType::OUT_OF_RANGE;
    //   continue;  // Skip non-ground points
    // }
    // 2. the angle is exceed the local slope threshold
    auto dx = point.x - prev_gnd_cell.gnd_avg_x;
    auto dy = point.y - prev_gnd_cell.gnd_avg_y;
    auto dz = point.z - prev_gnd_cell.gnd_avg_z;
    auto d_radius = point.radius - prev_gnd_cell.radius_avg;

    // 2. the angle is exceed the local slope threshold
    if (point.z / point.radius > filter_parameters_dev->global_slope_max_ratio) {
      point.type = PointType::NON_GROUND;
      continue;  // Skip non-ground points
    }
    // 3. local slope
    if (dz / d_radius > filter_parameters_dev->local_slope_max_ratio) {
      point.type = PointType::NON_GROUND;
      continue;  // Skip non-ground points
    }

    if (dz / d_radius < -filter_parameters_dev->global_slope_max_ratio) {
      // If the point is below the estimated ground height, classify it as non-ground
      point.type = PointType::OUT_OF_RANGE;
      continue;  // Skip non-ground points
    }
    point.type = PointType::GROUND;
    updateGndPointInCell(current_cell, point);
  }
  if (
    filter_parameters_dev->use_recheck_ground_cluster && current_cell.num_ground_points > 1 &&
    current_cell.radius_avg > filter_parameters_dev->recheck_start_distance) {
    // Recheck the ground points in the cell
    recheckCell(
      current_cell, classify_points, idx_start_point_of_cell, num_points_of_cell,
      filter_parameters_dev, cell_id);
  }
}

__global__ void scanPerSectorGroundReferenceKernel(
  ClassifiedPointTypeStruct * classified_points_dev, const int * num_points_per_cell_dev,
  CellCentroid * cells_centroid_list_dev, const int * cell_start_point_idx_dev,
  const FilterParameters * filter_parameters_dev)
{
  // Implementation of the kernel
  // scan in each sector from cell_index_in_sector = 0 to max_num_cells_per_sector
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= filter_parameters_dev->num_sectors) {
    return;  // Out of bounds
  }
  // For each sector, find the ground reference points if points exist
  // otherwise, use the previous sector ground reference points
  // initialize the previous cell centroid

  // Process the first cell of the sector
  SegmentationMode mode = SegmentationMode::UNINITIALIZED;

  for (int cell_index_in_sector = 0;
       cell_index_in_sector < filter_parameters_dev->max_num_cells_per_sector;
       ++cell_index_in_sector) {
    auto sector_start_cell_index = idx * filter_parameters_dev->max_num_cells_per_sector;
    auto cell_id = sector_start_cell_index + cell_index_in_sector;
    auto num_points_in_cell = num_points_per_cell_dev[cell_id];
    auto index_start_point_current_cell = cell_start_point_idx_dev[cell_id];

    checkSegmentMode(
      cells_centroid_list_dev, cell_index_in_sector, sector_start_cell_index,
      filter_parameters_dev->gnd_cell_buffer_size, mode);

    if (mode == SegmentationMode::UNINITIALIZED) {
      SegmentInitializedCell(
        cells_centroid_list_dev, classified_points_dev, index_start_point_current_cell,
        num_points_in_cell, filter_parameters_dev, sector_start_cell_index, cell_index_in_sector);
    } else if (mode == SegmentationMode::CONTINUOUS) {
      SegmentContinuousCell(
        cells_centroid_list_dev, classified_points_dev, index_start_point_current_cell,
        num_points_in_cell, filter_parameters_dev, sector_start_cell_index, cell_index_in_sector);
    } else if (mode == SegmentationMode::DISCONTINUOUS) {
      SegmentDiscontinuousCell(
        cells_centroid_list_dev, classified_points_dev, index_start_point_current_cell,
        num_points_in_cell, filter_parameters_dev, sector_start_cell_index, cell_index_in_sector);
    } else if (mode == SegmentationMode::BREAK) {
      SegmentBreakCell(
        cells_centroid_list_dev, classified_points_dev, index_start_point_current_cell,
        num_points_in_cell, filter_parameters_dev, sector_start_cell_index, cell_index_in_sector);
    }

    // if the first round of scan
  }
}

__global__ void sortPointsInCellsKernel(
  const int * __restrict__ num_points_per_cell_dev,
  ClassifiedPointTypeStruct * classified_points_dev, const int max_num_cells,
  const int max_num_points_per_cell)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_num_cells) {
    return;  // Out of bounds
  }

  auto * cell_points = classified_points_dev + idx * max_num_points_per_cell;
  int num_points_in_cell = num_points_per_cell_dev[idx];
  if (num_points_in_cell <= 1) {
    return;  // No need to sort if there is 0 or 1 point in the cell
  }
  // Sort the points in the cell by radius using a cub::DeviceRadixSort
  // the points are located in cell_points, and the number of points is num_points_in_cell
}

__global__ void CellsCentroidInitializeKernel(
  CellCentroid * cells_centroid_list_dev, const int max_num_cells)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= max_num_cells) {
    return;  // Out of bounds
  }
  cells_centroid_list_dev[idx].radius_avg = 0.0f;
  cells_centroid_list_dev[idx].height_avg = 0.0f;
  cells_centroid_list_dev[idx].height_max = -FLT_MAX;
  cells_centroid_list_dev[idx].height_min = FLT_MAX;
  cells_centroid_list_dev[idx].num_points = 0;
  cells_centroid_list_dev[idx].cell_id = -1;  // Initialize cell_id to -1
  cells_centroid_list_dev[idx].gnd_avg_z = 0.0f;
  cells_centroid_list_dev[idx].gnd_avg_x = 0.0f;
  cells_centroid_list_dev[idx].gnd_avg_y = 0.0f;
  cells_centroid_list_dev[idx].gnd_max_z = -FLT_MAX;
  cells_centroid_list_dev[idx].gnd_min_z = FLT_MAX;
  cells_centroid_list_dev[idx].num_ground_points = 0;
  cells_centroid_list_dev[idx].start_point_index = 0;
}

__global__ void CellCentroidUpdateKernel(
  const PointTypeStruct * __restrict__ input_points, const size_t num_input_points,
  const float center_x, const float center_y, const float sector_angle_rad, const float max_radius,
  const int max_number_cels_per_sector, const int max_num_cells, const size_t sector_num,
  const float cell_size_m, CellCentroid * cells_centroid_list_dev)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_input_points) {
    return;
  }

  const auto inv_sector_angle_rad = 1.0f / sector_angle_rad;

  // Calculate the angle and distance from the center
  const float dx = input_points[idx].x - center_x;
  const float dy = input_points[idx].y - center_y;
  const float radius = sqrtf(dx * dx + dy * dy);
  const float angle = fastAtan2_0_2Pi(dy, dx);  // replace with approximate atan

  // Determine the radial division index
  auto sector_index = static_cast<int>(angle * inv_sector_angle_rad);
  auto cell_index_in_sector = static_cast<int>(radius / cell_size_m);
  auto cell_id = sector_index * max_number_cels_per_sector + cell_index_in_sector;

  // const auto cell_id = getCellID(
  //   input_points[idx], center_x, center_y, inv_sector_angle_rad, cell_size_m,
  //   max_number_cels_per_sector, max_num_cells);

  if (cell_id < 0 || cell_id >= max_num_cells) {
    return;  // Out of bounds
  }
  // add pointcloud to output grid list
  // Update the existing grid
  auto & cell = cells_centroid_list_dev[cell_id];
  int current_cell_points_num = atomicAdd(&cell.num_points, 1);
  // cell.radius_avg =
  //   (cell.radius_avg * current_cell_points_num + radius) / (current_cell_points_num + 1);
  // cell.height_avg = (cell.height_avg * current_cell_points_num + input_points[idx].z) /
  //                   (current_cell_points_num + 1);
  // cell.height_max = fmaxf(cell.height_max, input_points[idx].z);
  // cell.height_min = fminf(cell.height_min, input_points[idx].z);
}

// Mark obstacle points for point in classified_points_dev
__global__ void markObstaclePointsKernel(
  ClassifiedPointTypeStruct * classified_points_dev, const int max_num_classified_points,
  const size_t num_points, int * __restrict__ flags)
{
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= static_cast<size_t>(max_num_classified_points)) {
    return;
  }
  // check if the classified_points_dev[idx] is existing?
  if (classified_points_dev[idx].radius < 0.0f) {
    return;
  }

  // extract origin index of point
  auto origin_index = classified_points_dev[idx].origin_index;
  auto point_type = classified_points_dev[idx].type;
  if (origin_index > static_cast<size_t>(num_points) || origin_index < 1) {
    return;
  }

  // Mark obstacle points for point in classified_points_dev

  flags[origin_index] = (point_type == PointType::NON_GROUND) ? 1 : 0;
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

void CudaScanGroundSegmentationFilter::scanObstaclePoints(
  const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input_points,
  PointTypeStruct * output_points_dev, size_t * num_output_points,
  CellCentroid * cells_centroid_list_dev)
{
  // Implementation of the function to scan obstacle points
  if (number_input_points_ == 0) {
    *num_output_points = 0;
    return;  // No points to process
  }
}

// ============= Sort points in each cell by radius =============
void CudaScanGroundSegmentationFilter::sortPointsInCells(
  const int * num_points_per_cell_dev, ClassifiedPointTypeStruct * classified_points_dev)
{
  (void)num_points_per_cell_dev;
  (void)classified_points_dev;
}

// ============ Scan per sector to get ground reference =============
void CudaScanGroundSegmentationFilter::scanPerSectorGroundReference(
  ClassifiedPointTypeStruct * classified_points_dev, const int * num_points_per_cell_dev,
  CellCentroid * cells_centroid_list_dev, const int * cell_start_point_idx_dev,
  const FilterParameters * filter_parameters_dev)
{
  const int num_sectors = filter_parameters_.num_sectors;

  dim3 block_dim(num_sectors);
  dim3 grid_dim((num_sectors + block_dim.x - 1) / block_dim.x);
  // Launch the kernel to scan for ground points in each sector
  scanPerSectorGroundReferenceKernel<<<grid_dim, block_dim, 0, ground_segment_stream_>>>(
    classified_points_dev, num_points_per_cell_dev, cells_centroid_list_dev,
    cell_start_point_idx_dev, filter_parameters_dev);
  CHECK_CUDA_ERROR(cudaStreamSynchronize(ground_segment_stream_));
}

// ============= Get obstacle point cloud =============
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

// =========== looping all input pointcloud and update cells ==================
void CudaScanGroundSegmentationFilter::addPointToCells(
  const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input_points,
  CellCentroid * cells_centroid_list_dev)
{
  // Implementation of the function to divide the point cloud into radial divisions
  // Sort the points in each radial division by distance from the center
  // return the indices of the points in each radial division
  if (number_input_points_ == 0) {
    return;  // No points to process
  }
  const auto * input_points_dev =
    reinterpret_cast<const PointTypeStruct *>(input_points->data.get());
  const float center_x = filter_parameters_.center_pcl_shift;
  const float center_y = 0.0f;

  // Launch the kernel to divide the point cloud into radial divisions
  // Each thread will process one point and calculate its angle and distance from the center

  dim3 block_dim(512);
  dim3 grid_dim((number_input_points_ + block_dim.x - 1) / block_dim.x);

  // initialize the list of cells centroid
  CellsCentroidInitializeKernel<<<grid_dim, block_dim, 0, ground_segment_stream_>>>(
    cells_centroid_list_dev, filter_parameters_.max_num_cells);
  CHECK_CUDA_ERROR(cudaGetLastError());

  auto max_cells_num = filter_parameters_.max_num_cells;
  CellCentroidUpdateKernel<<<grid_dim, block_dim, 0, ground_segment_stream_>>>(
    input_points_dev, number_input_points_, center_x, center_y, filter_parameters_.sector_angle_rad,
    filter_parameters_.max_radius, filter_parameters_.max_num_cells_per_sector, max_cells_num,
    filter_parameters_.num_sectors, filter_parameters_.cell_divider_size_m,
    cells_centroid_list_dev);
  CHECK_CUDA_ERROR(cudaGetLastError());

  CHECK_CUDA_ERROR(cudaStreamSynchronize(ground_segment_stream_));
}

// ========== Assign each pointcloud to specific cell =========================
void CudaScanGroundSegmentationFilter::assignPointToClassifyPoint(
  const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input_points,
  const CellCentroid * cells_centroid_list_dev, const FilterParameters * filter_parameters_dev,
  int * cell_counts_dev, const int * num_points_per_cell_dev, const int * cell_start_point_idx_dev,
  ClassifiedPointTypeStruct * classified_points_dev)
{
  // implementation of the function to split point cloud into cells
  if (number_input_points_ == 0) {
    return;  // No points to process
  }
  // Initialize the cells_centroid_list_dev value
  dim3 block_dim(512);
  dim3 grid_dim((number_input_points_ + block_dim.x - 1) / block_dim.x);

  initPoints<<<grid_dim, block_dim, 0, ground_segment_stream_>>>(
    classified_points_dev, number_input_points_);
  CHECK_CUDA_ERROR(cudaGetLastError());

  const auto * input_points_dev =
    reinterpret_cast<const PointTypeStruct *>(input_points->data.get());

  assignPointToClassifyPointKernel<<<grid_dim, block_dim, 0, ground_segment_stream_>>>(
    input_points_dev, number_input_points_, num_points_per_cell_dev, cell_start_point_idx_dev,
    cells_centroid_list_dev, cell_counts_dev, filter_parameters_dev, classified_points_dev);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaStreamSynchronize(ground_segment_stream_));
}

// ============= Extract non-ground points =============
void CudaScanGroundSegmentationFilter::extractNonGroundPoints(
  const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input_points,
  ClassifiedPointTypeStruct * classified_points_dev, const int max_num_cells,
  const int * cell_start_point_idx_dev, PointTypeStruct * output_points_dev,
  size_t * num_output_points_host)
{
  if (number_input_points_ == 0) {
    *num_output_points_host = 0;
    return;  // No points to process
  }
  int * flag_dev = allocateBufferFromPool<int>(
    number_input_points_);  // list flag of Non-Groud pointcloud related to classified_points_dev

  dim3 block_dim(512);
  dim3 grid_dim((number_input_points_ + block_dim.x - 1) / block_dim.x);
  setFlagsKernel<<<grid_dim, block_dim, 0, ground_segment_stream_>>>(
    flag_dev, number_input_points_, 1);
  CHECK_CUDA_ERROR(cudaGetLastError());

  // auto * flag_dev = allocateBufferFromPool<int>(number_input_points_);
  auto * indices_dev = allocateBufferFromPool<int>(number_input_points_);
  void * temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  markObstaclePointsKernel<<<grid_dim, block_dim, 0, ground_segment_stream_>>>(
    classified_points_dev, number_input_points_, number_input_points_, flag_dev);
  // CHECK_CUDA_ERROR(cudaMemset(&flag_dev,1,number_input_points_));
  // CHECK_CUDA_ERROR(cudaGetLastError());

  cub::DeviceScan::ExclusiveSum(
    nullptr, temp_storage_bytes, flag_dev, indices_dev, static_cast<int>(number_input_points_),
    ground_segment_stream_);
  CHECK_CUDA_ERROR(
    cudaMallocFromPoolAsync(&temp_storage, temp_storage_bytes, mem_pool_, ground_segment_stream_));

  cub::DeviceScan::ExclusiveSum(
    temp_storage, temp_storage_bytes, flag_dev, indices_dev, static_cast<int>(number_input_points_),
    ground_segment_stream_);
  CHECK_CUDA_ERROR(
    cudaMallocFromPoolAsync(&temp_storage, temp_storage_bytes, mem_pool_, ground_segment_stream_));

  CHECK_CUDA_ERROR(cudaGetLastError());

  const auto * input_points_dev =
    reinterpret_cast<const PointTypeStruct *>(input_points->data.get());

  scatterKernel<<<grid_dim, block_dim, 0, ground_segment_stream_>>>(
    input_points_dev, flag_dev, indices_dev, number_input_points_, output_points_dev);
  CHECK_CUDA_ERROR(cudaGetLastError());
  // Count the number of valid points
  int last_index = 0;
  int last_flag = 0;
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    &last_index, indices_dev + number_input_points_ - 1, sizeof(int), cudaMemcpyDeviceToHost,
    ground_segment_stream_));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    &last_flag, flag_dev + number_input_points_ - 1, sizeof(int), cudaMemcpyDeviceToHost,
    ground_segment_stream_));
  CHECK_CUDA_ERROR(cudaStreamSynchronize(ground_segment_stream_));

  const size_t num_output_points = static_cast<size_t>(last_flag + last_index);
  *num_output_points_host = num_output_points;

  if (temp_storage) {
    CHECK_CUDA_ERROR(cudaFreeAsync(temp_storage, ground_segment_stream_));
  }
  returnBufferToPool(flag_dev);
  returnBufferToPool(indices_dev);
}

void CudaScanGroundSegmentationFilter::getCellStartPointIndex(
  const FilterParameters * filter_parameters_dev, CellCentroid * cells_centroid_list_dev,
  int * num_points_per_cell_dev, int * max_num_point_dev, int * cell_start_point_idx_dev)
{
  void * d_temp_storage = nullptr;

  size_t temp_storage_bytes = 0;
  int threads = filter_parameters_.num_sectors;
  int blocks = (filter_parameters_.max_num_cells + threads - 1) / threads;
  getCellNumPointsKernel<<<blocks, threads, 0, ground_segment_stream_>>>(
    cells_centroid_list_dev, filter_parameters_.max_num_cells, num_points_per_cell_dev);
  CHECK_CUDA_ERROR(cudaGetLastError());

  // accumulate num_points_per_cell_dev into cell_start_point_idx_dev
  //  Exclusive scan
  // First call: get temporary storage size
  cub::DeviceScan::ExclusiveSum(
    d_temp_storage, temp_storage_bytes, num_points_per_cell_dev, cell_start_point_idx_dev,
    filter_parameters_.max_num_cells, ground_segment_stream_);
  CHECK_CUDA_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));
  cub::DeviceScan::ExclusiveSum(
    d_temp_storage, temp_storage_bytes, num_points_per_cell_dev, cell_start_point_idx_dev,
    filter_parameters_.max_num_cells, ground_segment_stream_);
  CHECK_CUDA_ERROR(cudaGetLastError());
  CHECK_CUDA_ERROR(cudaStreamSynchronize(ground_segment_stream_));

  CHECK_CUDA_ERROR(cudaFree(d_temp_storage));
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

  if (number_input_points_ == 0) {
    filtered_output->width = 0;
    filtered_output->height = 0;

    filtered_output->header = input_points->header;
    filtered_output->height = 1;  // Set height to 1 for unorganized point
    filtered_output->is_bigendian = input_points->is_bigendian;
    filtered_output->point_step = input_points->point_step;
    filtered_output->row_step = static_cast<uint32_t>(num_output_points * sizeof(PointTypeStruct));
    filtered_output->is_dense = input_points->is_dense;
    filtered_output->fields = input_points->fields;
    return filtered_output;  // No points to process
  }

  // Allocate and copy filter parameters to device
  FilterParameters * filter_parameters_dev = allocateBufferFromPool<FilterParameters>(1);
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    filter_parameters_dev, &filter_parameters_, sizeof(FilterParameters), cudaMemcpyHostToDevice,
    ground_segment_stream_));

  // split pointcloud to radial divisions
  // sort points in each radial division by distance from the center
  auto * cells_centroid_list_dev =
    allocateBufferFromPool<CellCentroid>(filter_parameters_.max_num_cells);

  // calculate the centroid of each cell
  addPointToCells(input_points, cells_centroid_list_dev);

  // get maximum of num_points along all cells in cells_centroid_list_dev, on device

  int * num_points_per_cell_dev;  // array of num_points for each cell
  int * cell_start_point_idx_dev;
  int * max_num_point_dev;  // storage for maximum number of points along all cells
  int max_num_cells = filter_parameters_.max_num_cells_per_sector * filter_parameters_.num_sectors;

  CHECK_CUDA_ERROR(cudaMalloc(&num_points_per_cell_dev, max_num_cells * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMalloc(&cell_start_point_idx_dev, max_num_cells * sizeof(int)));
  CHECK_CUDA_ERROR(cudaMalloc(&max_num_point_dev, sizeof(int)));

  getCellStartPointIndex(
    filter_parameters_dev, cells_centroid_list_dev, num_points_per_cell_dev, max_num_point_dev,
    cell_start_point_idx_dev);

  auto * classified_points_dev =
    allocateBufferFromPool<ClassifiedPointTypeStruct>(number_input_points_);

  int * cell_counts_dev;  // array of point index in each cell
  CHECK_CUDA_ERROR(cudaMallocFromPoolAsync(
    &cell_counts_dev, max_num_cells * sizeof(int), mem_pool_, ground_segment_stream_));
  CHECK_CUDA_ERROR(
    cudaMemsetAsync(cell_counts_dev, 0, max_num_cells * sizeof(int), ground_segment_stream_));

  assignPointToClassifyPoint(
    input_points, cells_centroid_list_dev, filter_parameters_dev, cell_counts_dev,
    num_points_per_cell_dev, cell_start_point_idx_dev, classified_points_dev);

  CHECK_CUDA_ERROR(cudaFree(cell_counts_dev));

  // // sort classified point by radius in each cell
  // // The real existing points in each cell are located in cells_centroid_list_dev's num_points
  // // sortPointsInCells(num_points_per_cell_dev, classified_points_dev);

  // // classify points without sorting
  // allocate gnd_grid_buffer_size of CentroidCells memory for  previous cell centroids
  CellCentroid * prev_cell_centroids;
  CHECK_CUDA_ERROR(cudaMallocFromPoolAsync(
    &prev_cell_centroids,
    filter_parameters_.num_sectors * filter_parameters_.gnd_cell_buffer_size * sizeof(CellCentroid),
    mem_pool_, ground_segment_stream_));
  CHECK_CUDA_ERROR(cudaMemsetAsync(
    prev_cell_centroids, 0,
    filter_parameters_.num_sectors * filter_parameters_.gnd_cell_buffer_size * sizeof(CellCentroid),
    ground_segment_stream_));
  scanPerSectorGroundReference(
    classified_points_dev, num_points_per_cell_dev, cells_centroid_list_dev,
    cell_start_point_idx_dev, filter_parameters_dev);

  // Extract obstacle points from classified_points_dev
  extractNonGroundPoints(
    input_points, classified_points_dev, max_num_cells, cell_start_point_idx_dev, output_points_dev,
    &num_output_points);

  // // mark valid points based on height threshold

  CHECK_CUDA_ERROR(cudaFree(num_points_per_cell_dev));
  CHECK_CUDA_ERROR(cudaFree(cell_start_point_idx_dev));
  CHECK_CUDA_ERROR(cudaFree(max_num_point_dev));
  CHECK_CUDA_ERROR(cudaFreeAsync(prev_cell_centroids, ground_segment_stream_));
  // Return the device memory to pool
  returnBufferToPool(filter_parameters_dev);
  returnBufferToPool(cells_centroid_list_dev);
  returnBufferToPool(classified_points_dev);

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

}  // namespace autoware::cuda_pointcloud_preprocessor
