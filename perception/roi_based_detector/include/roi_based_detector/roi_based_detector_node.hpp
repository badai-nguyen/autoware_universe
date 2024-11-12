// create node to estimate object from rois

#ifndef ROI_BASED_DETECTOR__ROI_BASED_DETECTOR_NODE_HPP_
#define ROI_BASED_DETECTOR__ROI_BASED_DETECTOR_NODE_HPP_

#include <rclcpp/rclcpp.hpp>

#include <tier4_perception_msgs/msg/detected_objects_with_feature.hpp>
#include <autoware_perception_msgs/msg/detected_objects.hpp>

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
  void roiCallback(const tier4_perception_msgs::msg::DetectedObjectsWithFeature::ConstSharedPtr & msg);
  void convertRoiToObjects(const tier4_perception_msgs::msg::DetectedObjectWithFeature & roi,
                           autoware_perception_msgs::msg::DetectedObject & object);

  rclcpp::Publisher<tier4_perception_msgs::msg::DetectedObjectsWithFeature>::SharedPtr rois_pub_;
  rclcpp::Publisher<autoware_perception_msgs::msg::DetectedObjects>::SharedPtr objects_pub_;
  rclcpp::Subscription<tier4_perception_msgs::msg::DetectedObjectsWithFeature>::SharedPtr roi_sub_;
  // camera_info sub
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
};
}  // namespace roi_based_detector

#endif  // ROI_BASED_DETECTOR__ROI_BASED_DETECTOR_NODE_HPP_