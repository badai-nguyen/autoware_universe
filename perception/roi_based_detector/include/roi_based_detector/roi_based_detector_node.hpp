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

#ifndef ROI_BASED_DETECTOR__ROI_BASED_DETECTOR_NODE_HPP_
#define ROI_BASED_DETECTOR__ROI_BASED_DETECTOR_NODE_HPP_

#include "autoware/universe_utils/ros/transform_listener.hpp"

#include <rclcpp/rclcpp.hpp>

#include <autoware_perception_msgs/msg/detected_objects.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <tier4_perception_msgs/msg/detected_objects_with_feature.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#ifdef ROS_DISTRO_GALACTIC
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#else
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#endif

#include <memory>
#include <string>
#include <vector>

namespace roi_based_detector
{
class RoiBasedDetectorNode : public rclcpp::Node
{
public:
  explicit RoiBasedDetectorNode(const rclcpp::NodeOptions & node_options);

private:
  void roiCallback(
    const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr & msg);
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg);
  void convertRoiToObjects(
    const tier4_perception_msgs::msg::DetectedObjectWithFeature & roi,
    const sensor_msgs::msg::CameraInfo & camera_info,
    autoware_perception_msgs::msg::DetectedObject & object);

  rclcpp::Publisher<tier4_perception_msgs::msg::DetectedObjectsWithFeature>::SharedPtr rois_pub_;
  rclcpp::Publisher<autoware_perception_msgs::msg::DetectedObjects>::SharedPtr objects_pub_;
  rclcpp::Subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>::SharedPtr roi_sub_;
  // camera_info sub
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  sensor_msgs::msg::CameraInfo camera_info_;
  // sync subscriber
  // message_filters::Subscriber<tier4_perception_msgs::msg::DetectedObjectsWithFeature> roi_sub_;
  // message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;
  // using SyncPolicy = message_filters::sync_policies::ApproximateTime<
  //   tier4_perception_msgs::msg::DetectedObjectsWithFeature, sensor_msgs::msg::CameraInfo>;
  // using Sync = message_filters::Synchronizer<SyncPolicy>;
  // // std::shared_ptr<Sync> sync_ptr_;
  // Sync sync_;
  // tf2_ros::Buffer tf_buffer_;
  // tf2_ros::TransformListener tf_listener_;

  std::shared_ptr<autoware::universe_utils::TransformListener> transform_listener_;
  geometry_msgs::msg::TransformStamped::ConstSharedPtr transform_;
};
}  // namespace roi_based_detector

#endif  // ROI_BASED_DETECTOR__ROI_BASED_DETECTOR_NODE_HPP_
