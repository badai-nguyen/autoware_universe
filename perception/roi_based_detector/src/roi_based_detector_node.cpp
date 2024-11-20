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

namespace roi_based_detector
{
// initialize Constructor
RoiBasedDetectorNode::RoiBasedDetectorNode(const rclcpp::NodeOptions & node_options)
: Node("roi_based_detector_node", node_options)
{
  // create publisher
  rois_pub_ = this->create_publisher<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
    "output_rois", 1);
  objects_pub_ =
    this->create_publisher<autoware_perception_msgs::msg::DetectedObjects>("output_objects", 1);
  // create subscriber
  roi_sub_ = this->create_subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>(
    "input_rois", 1, std::bind(&RoiBasedDetectorNode::roiCallback, this, std::placeholders::_1));

  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "input_camera_info", rclcpp::QoS(10).reliability(rclcpp::ReliabilityPolicy::BestEffort),
    std::bind(&RoiBasedDetectorNode::cameraInfoCallback, this, std::placeholders::_1));
}

void RoiBasedDetectorNode::cameraInfoCallback(
  const sensor_msgs::msg::CameraInfo::ConstSharedPtr & msg)
{
  camera_info_ = *msg;
  RCLCPP_INFO(get_logger(), "camera_info is received");
}
// implement roiCallback
void RoiBasedDetectorNode::roiCallback(const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr & msg)
{
  // publish message subscribed msg to object_pub_
  // (void)msg;
  autoware_perception_msgs::msg::DetectedObjects objects;
  for (const auto & roi : msg->feature_objects) {

    autoware_perception_msgs::msg::DetectedObject object;
    //object.classification.push_back(roi.object.classification.front().label);
    object.classification.front().label = roi.object.classification.front().label;
    object.existence_probability = roi.object.existence_probability;
    // object.roi = roi.feature.roi;
    objects.objects.push_back(object);
    continue;
  }
  rois_pub_->publish(*msg);
  objects.header = msg->header;
  objects_pub_->publish(objects);

}

void RoiBasedDetectorNode::convertRoiToObjects(const tier4_perception_msgs::msg::DetectedObjectWithFeature & roi,
                           autoware_perception_msgs::msg::DetectedObject & object)
{
  object.classification.front().label = roi.object.classification.front().label;
  object.classification.front().probability = roi.object.classification.front().probability;
  auto x_offset = roi.feature.roi.x_offset;
  auto y_offset = roi.feature.roi.y_offset;
  auto width = roi.feature.roi.width;
  auto height = roi.feature.roi.height;
  
  object.kinematics.pose_with_covariance.pose.position.z = 0.0;
  object.kinematics.pose_with_covariance.pose.orientation.x = 0.0;
  object.kinematics.pose_with_covariance.pose.orientation.y = 0.0;
  object.existence_probability = roi.object.existence_probability;
   
}

}  // namespace roi_based_detector

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(roi_based_detector::RoiBasedDetectorNode)