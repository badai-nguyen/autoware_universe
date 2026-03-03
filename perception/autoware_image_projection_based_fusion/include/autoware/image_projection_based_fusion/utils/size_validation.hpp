// Copyright 2026 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef AUTOWARE__IMAGE_PROJECTION_BASED_FUSION__UTILS__SIZE_VALIDATION_HPP_
#define AUTOWARE__IMAGE_PROJECTION_BASED_FUSION__UTILS__SIZE_VALIDATION_HPP_

#include <autoware_perception_msgs/msg/detected_object.hpp>
#include <autoware_perception_msgs/msg/object_classification.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/region_of_interest.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <utility>

namespace autoware::image_projection_based_fusion
{

using autoware_perception_msgs::msg::ObjectClassification;

/**
 * @brief Result of size validation
 */
struct SizeValidationResult
{
  bool is_valid = false;
};

/**
 * @brief Per-class size validation parameters (length = x, width = y, height = z)
 */
struct ClassSizeValidationParams
{
  bool enable = false;
  double min_length = 0.0;
  double max_length = 10.0;
  double min_width = 0.0;
  double max_width = 10.0;
  double min_height = 0.0;
  double max_height = 10.0;
};

/**
 * @brief Calculate the dimensions of a pointcloud cluster (length = x, width = y, height = z)
 * @param cluster PointCloud2 cluster data
 * @return length [m], width [m], height [m] if successful
 */
inline std::optional<std::tuple<double, double, double>> calculateClusterDimensions(
  const sensor_msgs::msg::PointCloud2 & cluster)
{
  if (cluster.data.empty()) {
    return std::nullopt;
  }

  double min_x = std::numeric_limits<double>::max();
  double max_x = std::numeric_limits<double>::lowest();
  double min_y = std::numeric_limits<double>::max();
  double max_y = std::numeric_limits<double>::lowest();
  double min_z = std::numeric_limits<double>::max();
  double max_z = std::numeric_limits<double>::lowest();
  size_t valid_points = 0;

  for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(cluster, "x"), iter_y(cluster, "y"),
       iter_z(cluster, "z");
       iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
    if (!std::isfinite(*iter_x) || !std::isfinite(*iter_y) || !std::isfinite(*iter_z)) {
      continue;
    }
    min_x = std::min(min_x, static_cast<double>(*iter_x));
    max_x = std::max(max_x, static_cast<double>(*iter_x));
    min_y = std::min(min_y, static_cast<double>(*iter_y));
    max_y = std::max(max_y, static_cast<double>(*iter_y));
    min_z = std::min(min_z, static_cast<double>(*iter_z));
    max_z = std::max(max_z, static_cast<double>(*iter_z));
    valid_points++;
  }

  if (valid_points == 0) {
    return std::nullopt;
  }

  const double length = max_x - min_x;
  const double width = max_y - min_y;
  const double height = max_z - min_z;
  if (length <= 0.0 || width <= 0.0 || height <= 0.0) {
    return std::nullopt;
  }
  return std::make_tuple(length, width, height);
}

/**
 * @brief Validate 3D size of a cluster against class-specific constraints (length = x, width = y, height = z)
 * @param cluster PointCloud2 cluster data to validate the size
 * @param params Validation parameters for the object class
 * @return True if the size is valid, false otherwise
 */
inline bool validateObject3DSize(
  const sensor_msgs::msg::PointCloud2 & cluster, const ClassSizeValidationParams & params)
{
  if (!params.enable) {
    return true;
  }
  const auto dimensions = calculateClusterDimensions(cluster);
  if (!dimensions.has_value()) {
    return false;
  }
  const double length = std::get<0>(*dimensions);
  const double width = std::get<1>(*dimensions);
  const double height = std::get<2>(*dimensions);
  if (length < params.min_length || length > params.max_length) {
    return false;
  }
  if (width < params.min_width || width > params.max_width) {
    return false;
  }
  if (height < params.min_height || height > params.max_height) {
    return false;
  }
  return true;
}

}  // namespace autoware::image_projection_based_fusion

#endif  // AUTOWARE__IMAGE_PROJECTION_BASED_FUSION__UTILS__SIZE_VALIDATION_HPP_
