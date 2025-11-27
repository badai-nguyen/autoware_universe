// Copyright 2025 TIER IV, Inc.
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

#include "autoware/pointcloud_preprocessor/concatenate_data/combine_cloud_handler_base.hpp"

#include <tf2/LinearMath/Transform.h>
#include <tf2/convert.h>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#endif

#include <deque>

namespace autoware::pointcloud_preprocessor
{

void CombineCloudHandlerBase::process_twist(
  const geometry_msgs::msg::TwistWithCovarianceStamped::ConstSharedPtr & twist_msg, bool use_imu)
{
  geometry_msgs::msg::TwistStamped msg;
  msg.header = twist_msg->header;
  msg.twist = twist_msg->twist.twist;

  // If use_imu is enabled, replace angular velocity with IMU data
  if (use_imu && !angular_velocity_queue_.empty()) {
    // Find the closest IMU measurement to this twist timestamp
    // Since IMU runs at ~40Hz and twist at ~20Hz, we need to find the temporally closest sample
    const double twist_time = rclcpp::Time(msg.header.stamp).seconds();
    
    auto it_imu = std::lower_bound(
      std::begin(angular_velocity_queue_), std::end(angular_velocity_queue_),
      twist_time,
      [](const geometry_msgs::msg::Vector3Stamped & x, const double t) {
        return rclcpp::Time(x.header.stamp).seconds() < t;
      });

    // it_imu now points to the first IMU sample >= twist_time
    // Check if the previous sample is actually closer in time
    if (it_imu != std::end(angular_velocity_queue_)) {
      if (it_imu != std::begin(angular_velocity_queue_)) {
        auto it_prev = std::prev(it_imu);
        const double time_diff_current = std::abs(rclcpp::Time(it_imu->header.stamp).seconds() - twist_time);
        const double time_diff_prev = std::abs(rclcpp::Time(it_prev->header.stamp).seconds() - twist_time);
        
        // Use the temporally closest IMU sample
        if (time_diff_prev < time_diff_current) {
          it_imu = it_prev;
        }
      }
      
      msg.twist.angular.x = it_imu->vector.x;
      msg.twist.angular.y = it_imu->vector.y;
      msg.twist.angular.z = it_imu->vector.z;
    } else if (!angular_velocity_queue_.empty()) {
      // If all IMU samples are older than twist, use the most recent one
      auto it_last = std::prev(std::end(angular_velocity_queue_));
      msg.twist.angular.x = it_last->vector.x;
      msg.twist.angular.y = it_last->vector.y;
      msg.twist.angular.z = it_last->vector.z;
    }
  }

  // If time jumps backwards (e.g. when a rosbag restarts), clear buffer
  if (!twist_queue_.empty()) {
    if (rclcpp::Time(twist_queue_.front().header.stamp) > rclcpp::Time(msg.header.stamp)) {
      twist_queue_.clear();
    }
  }

  // Twist data in the queue that is older than the current twist by 1 second will be cleared.
  auto cutoff_time = rclcpp::Time(msg.header.stamp) - rclcpp::Duration::from_seconds(1.0);

  while (!twist_queue_.empty()) {
    if (rclcpp::Time(twist_queue_.front().header.stamp) > cutoff_time) {
      break;
    }
    twist_queue_.pop_front();
  }

  twist_queue_.push_back(msg);
}

void CombineCloudHandlerBase::process_odometry(
  const nav_msgs::msg::Odometry::ConstSharedPtr & odometry_msg)
{
  geometry_msgs::msg::TwistStamped msg;
  msg.header = odometry_msg->header;
  msg.twist = odometry_msg->twist.twist;

  // If time jumps backwards (e.g. when a rosbag restarts), clear buffer
  if (!twist_queue_.empty()) {
    if (rclcpp::Time(twist_queue_.front().header.stamp) > rclcpp::Time(msg.header.stamp)) {
      twist_queue_.clear();
    }
  }

  // Twist data in the queue that is older than the current twist by 1 second will be cleared.
  auto cutoff_time = rclcpp::Time(msg.header.stamp) - rclcpp::Duration::from_seconds(1.0);

  while (!twist_queue_.empty()) {
    if (rclcpp::Time(twist_queue_.front().header.stamp) > cutoff_time) {
      break;
    }
    twist_queue_.pop_front();
  }

  twist_queue_.push_back(msg);
}

std::deque<geometry_msgs::msg::TwistStamped> CombineCloudHandlerBase::get_twist_queue()
{
  return twist_queue_;
}

std::deque<geometry_msgs::msg::Vector3Stamped> CombineCloudHandlerBase::get_angular_velocity_queue()
{
  return angular_velocity_queue_;
}

void CombineCloudHandlerBase::process_imu(
  const std::string & base_frame, const sensor_msgs::msg::Imu::ConstSharedPtr & imu_msg)
{
  get_imu_transformation(base_frame, imu_msg->header.frame_id);
  enqueue_imu(imu_msg);
}

void CombineCloudHandlerBase::get_imu_transformation(
  const std::string & base_frame, const std::string & imu_frame)
{
  if (imu_transform_exists_) {
    return;
  }

  Eigen::Matrix4f eigen_imu_to_base_link;
  auto eigen_transform_opt = managed_tf_buffer_->getTransform<Eigen::Matrix4f>(
    base_frame, imu_frame, node_.now(), rclcpp::Duration::from_seconds(1.0), node_.get_logger());
  imu_transform_exists_ = eigen_transform_opt.has_value();
  
  if (imu_transform_exists_) {
    eigen_imu_to_base_link = *eigen_transform_opt;
    
    // Convert Eigen matrix to tf2::Transform
    tf2::Matrix3x3 rotation(
      eigen_imu_to_base_link(0, 0), eigen_imu_to_base_link(0, 1), eigen_imu_to_base_link(0, 2),
      eigen_imu_to_base_link(1, 0), eigen_imu_to_base_link(1, 1), eigen_imu_to_base_link(1, 2),
      eigen_imu_to_base_link(2, 0), eigen_imu_to_base_link(2, 1), eigen_imu_to_base_link(2, 2));
    
    tf2::Quaternion quaternion;
    rotation.getRotation(quaternion);
    
    geometry_imu_to_base_link_ptr_ = std::make_shared<geometry_msgs::msg::TransformStamped>();
    geometry_imu_to_base_link_ptr_->transform.rotation = tf2::toMsg(quaternion);
  }
}

void CombineCloudHandlerBase::enqueue_imu(const sensor_msgs::msg::Imu::ConstSharedPtr & imu_msg)
{
  geometry_msgs::msg::Vector3Stamped angular_velocity;
  angular_velocity.vector = imu_msg->angular_velocity;

  geometry_msgs::msg::Vector3Stamped transformed_angular_velocity;
  
  if (imu_transform_exists_ && geometry_imu_to_base_link_ptr_) {
    tf2::doTransform(
      angular_velocity, transformed_angular_velocity, *geometry_imu_to_base_link_ptr_);
  } else {
    // If no transform, use angular velocity as-is
    transformed_angular_velocity.vector = angular_velocity.vector;
  }
  
  transformed_angular_velocity.header = imu_msg->header;

  // If time jumps backwards (e.g. when a rosbag restarts), clear buffer
  if (!angular_velocity_queue_.empty()) {
    if (
      rclcpp::Time(angular_velocity_queue_.front().header.stamp) >
      rclcpp::Time(imu_msg->header.stamp)) {
      angular_velocity_queue_.clear();
    }
  }

  // IMU data in the queue that is older than the current imu msg by 1 second will be cleared.
  auto cutoff_time = rclcpp::Time(imu_msg->header.stamp) - rclcpp::Duration::from_seconds(1.0);

  while (!angular_velocity_queue_.empty()) {
    if (rclcpp::Time(angular_velocity_queue_.front().header.stamp) > cutoff_time) {
      break;
    }
    angular_velocity_queue_.pop_front();
  }

  angular_velocity_queue_.push_back(transformed_angular_velocity);
}

Eigen::Matrix4f CombineCloudHandlerBase::compute_transform_to_adjust_for_old_timestamp(
  const rclcpp::Time & old_stamp, const rclcpp::Time & new_stamp)
{
  // return identity if no twist is available
  if (twist_queue_.empty()) {
    RCLCPP_WARN_STREAM_THROTTLE(
      node_.get_logger(), *node_.get_clock(), std::chrono::milliseconds(10000).count(),
      "No twist is available. Please confirm twist topic and timestamp. Leaving point cloud "
      "untransformed.");
    return Eigen::Matrix4f::Identity();
  }

  auto old_twist_it = std::lower_bound(
    std::begin(twist_queue_), std::end(twist_queue_), old_stamp,
    [](const geometry_msgs::msg::TwistStamped & x, const rclcpp::Time & t) {
      return rclcpp::Time(x.header.stamp) < t;
    });
  old_twist_it = old_twist_it == twist_queue_.end() ? (twist_queue_.end() - 1) : old_twist_it;

  auto new_twist_it = std::lower_bound(
    std::begin(twist_queue_), std::end(twist_queue_), new_stamp,
    [](const geometry_msgs::msg::TwistStamped & x, const rclcpp::Time & t) {
      return rclcpp::Time(x.header.stamp) < t;
    });
  new_twist_it = new_twist_it == twist_queue_.end() ? (twist_queue_.end() - 1) : new_twist_it;

  auto prev_time = old_stamp;
  double x = 0.0;
  double y = 0.0;
  double yaw = 0.0;
  for (auto twist_it = old_twist_it; twist_it != new_twist_it + 1; ++twist_it) {
    const double dt =
      (twist_it != new_twist_it)
        ? (rclcpp::Time((*twist_it).header.stamp) - rclcpp::Time(prev_time)).seconds()
        : (rclcpp::Time(new_stamp) - rclcpp::Time(prev_time)).seconds();

    if (std::fabs(dt) > 0.1) {
      RCLCPP_WARN_STREAM_THROTTLE(
        node_.get_logger(), *node_.get_clock(), std::chrono::milliseconds(10000).count(),
        "Time difference is too large. Cloud not interpolate. Please confirm twist topic and "
        "timestamp");
      break;
    }

    const double distance = (*twist_it).twist.linear.x * dt;
    yaw += (*twist_it).twist.angular.z * dt;
    x += distance * std::cos(yaw);
    y += distance * std::sin(yaw);
    prev_time = (*twist_it).header.stamp;
  }

  Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();

  float cos_yaw = std::cos(yaw);
  float sin_yaw = std::sin(yaw);

  transformation_matrix(0, 3) = x;
  transformation_matrix(1, 3) = y;
  transformation_matrix(0, 0) = cos_yaw;
  transformation_matrix(0, 1) = -sin_yaw;
  transformation_matrix(1, 0) = sin_yaw;
  transformation_matrix(1, 1) = cos_yaw;

  return transformation_matrix;
}

}  // namespace autoware::pointcloud_preprocessor
