// Copyright 2024 TIER IV, Inc.
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

#include "roi_based_detector/roi_based_detector_node.hpp"

#include "rclcpp/qos.hpp"

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_eigen/tf2_eigen.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#else
#include <tf2_eigen/tf2_eigen.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#endif

namespace roi_based_detector
{
// initialize Constructor
RoiBasedDetectorNode::RoiBasedDetectorNode(const rclcpp::NodeOptions & node_options)
: Node("roi_based_detector_node", node_options)
{
  // create publisher
  rois_pub_ = this->create_publisher<DetectedObjectsWithFeature>("output_rois", 1);
  objects_pub_ = this->create_publisher<DetectedObjects>("output_objects", 1);
  // create subscriber
  roi_sub_ = this->create_subscription<DetectedObjectsWithFeature>(
    "input_rois", 1, std::bind(&RoiBasedDetectorNode::roiCallback, this, std::placeholders::_1));

  camera_info_sub_ = this->create_subscription<CameraInfo>(
    "input_camera_info", rclcpp::QoS(10).reliability(rclcpp::ReliabilityPolicy::BestEffort),
    std::bind(&RoiBasedDetectorNode::cameraInfoCallback, this, std::placeholders::_1));
  transform_listener_ = std::make_shared<TransformListener>(this);
}

Eigen::Matrix4d RoiBasedDetectorNode::transformToHomogeneous(
  const geometry_msgs::msg::Transform & transform)
{
  Eigen::Matrix4d homogeneous = Eigen::Matrix4d::Identity();

  // Extract translation
  homogeneous(0, 3) = transform.translation.x;
  homogeneous(1, 3) = transform.translation.y;
  homogeneous(2, 3) = transform.translation.z;

  // Extract rotation (quaternion to rotation matrix)
  tf2::Quaternion quat(
    transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w);
  tf2::Matrix3x3 rotationMatrix(quat);

  // Convert tf2::Matrix3x3 to Eigen::Matrix3d
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      homogeneous(i, j) = rotationMatrix[i][j];
    }
  }

  return homogeneous;
}

void RoiBasedDetectorNode::cameraInfoCallback(const CameraInfo::ConstSharedPtr & msg)
{
  camera_info_ = *msg;
  RCLCPP_INFO(get_logger(), "camera_info is received");
}
// implement roiCallback
void RoiBasedDetectorNode::roiCallback(const DetectedObjectsWithFeature::ConstSharedPtr & msg)
{
  if (camera_info_ == CameraInfo()) {
    RCLCPP_WARN(get_logger(), "camera_info is not received yet");
    return;
  }
  Eigen::Matrix4f projection;
  projection << camera_info_.p.at(0), camera_info_.p.at(1), camera_info_.p.at(2),
    camera_info_.p.at(3), camera_info_.p.at(4), camera_info_.p.at(5), camera_info_.p.at(6),
    camera_info_.p.at(7), camera_info_.p.at(8), camera_info_.p.at(9), camera_info_.p.at(10),
    camera_info_.p.at(11), 0.0, 0.0, 0.0, 1.0;

  DetectedObjects objects;

  // get transform from camera frame to base_link frame

  try {
    transform_ = transform_listener_->getTransform(
      "base_link", msg->header.frame_id, msg->header.stamp, rclcpp::Duration::from_seconds(0.01));
  } catch (const tf2::TransformException & ex) {
    RCLCPP_ERROR(get_logger(), "Failed to get transform: %s", ex.what());
    objects.header = msg->header;
    objects_pub_->publish(objects);
    return;
  }

  if (transform_ == nullptr) {
    RCLCPP_ERROR_THROTTLE(
      get_logger(), *get_clock(), 5000, "getTransform failed. radar output will be empty.");
    objects.header = msg->header;
    objects_pub_->publish(objects);
    return;
  }

  const Eigen::Matrix4f transform_matrix_cam2base =
    tf2::transformToEigen(transform_->transform).matrix().cast<float>();
  Eigen::Matrix4f inv_project = projection.inverse();
  Eigen::Matrix4f camera2lidar_mul_inv_projection = transform_matrix_cam2base * inv_project;

  for (const auto & obj_with_feature : msg->feature_objects) {
    DetectedObject object;
    object.classification.push_back(obj_with_feature.object.classification.front());
    // object.classification.front().label = obj_with_feature.object.classification.front().label;
    object.existence_probability = obj_with_feature.object.existence_probability;

    const double normalized_projected_x =
      static_cast<double>(obj_with_feature.feature.roi.x_offset);
    const double normalized_projected_y =
      static_cast<double>(obj_with_feature.feature.roi.y_offset);
    auto w_div_projected_z = -(camera2lidar_mul_inv_projection(2, 0) * normalized_projected_x +
                               camera2lidar_mul_inv_projection(2, 1) * normalized_projected_y +
                               camera2lidar_mul_inv_projection(2, 2)) /
                             camera2lidar_mul_inv_projection(2, 3);
    auto projected_z = 1.0 / (camera2lidar_mul_inv_projection(3, 0) * normalized_projected_x +
                              camera2lidar_mul_inv_projection(3, 1) * normalized_projected_y +
                              camera2lidar_mul_inv_projection(3, 2) +
                              camera2lidar_mul_inv_projection(3, 3) * w_div_projected_z);
    auto w = w_div_projected_z * projected_z;
    Eigen::Vector4f projected_point = Eigen::Vector4f(
      normalized_projected_x * projected_z, normalized_projected_y * projected_z, projected_z, w);

    Eigen::Vector4f point_lidar = camera2lidar_mul_inv_projection * projected_point;
    geometry_msgs::msg::PoseStamped pose_stamped{};
    pose_stamped.pose.position.x = point_lidar.x();
    pose_stamped.pose.position.y = point_lidar.y();
    pose_stamped.pose.position.z = point_lidar.z();
    object.shape.dimensions.x = 0.5;
    object.shape.dimensions.y = 0.5;
    object.shape.dimensions.z = 0.0;
    geometry_msgs::msg::Point32 point;
    point.x = point_lidar.x() - 0.5 / 2;
    point.y = point_lidar.y() - 0.5 / 2;
    point.z = point_lidar.z();
    object.shape.footprint.points.push_back(point);
    point.x = point_lidar.x() + 0.5 / 2;
    point.y = point_lidar.y() - 0.5 / 2;
    point.z = point_lidar.z();
    object.shape.footprint.points.push_back(point);
    point.x = point_lidar.x() + 0.5 / 2;
    point.y = point_lidar.y() + 0.5 / 2;
    point.z = point_lidar.z();
    object.shape.footprint.points.push_back(point);
    point.x = point_lidar.x() - 0.5 / 2;
    point.y = point_lidar.y() + 0.5 / 2;
    point.z = point_lidar.z();
    object.shape.footprint.points.push_back(point);
    
    object.kinematics.pose_with_covariance.pose = pose_stamped.pose;
    objects.objects.push_back(object);
    continue;
  }
  rois_pub_->publish(*msg);
  objects.header = msg->header;
  objects.header.frame_id = "base_link";
  objects_pub_->publish(objects);
}

void RoiBasedDetectorNode::convertRoiToObjects(
  const DetectedObjectWithFeature & roi, const CameraInfo & camera_info, DetectedObject & object)
{
  (void)roi;
  (void)camera_info;
  (void)object;
}

}  // namespace roi_based_detector

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(roi_based_detector::RoiBasedDetectorNode)
