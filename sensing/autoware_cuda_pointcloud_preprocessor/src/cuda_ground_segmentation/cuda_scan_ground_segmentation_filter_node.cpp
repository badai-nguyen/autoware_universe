
#include "autoware/cuda_pointcloud_preprocessor/cuda_ground_segmentation/cuda_scan_ground_segmentation_filter_node.hpp"

#include "autoware/pointcloud_preprocessor/utility/memory.hpp"

#include <autoware/vehicle_info_utils/vehicle_info_utils.hpp>
#include <autoware_utils/math/unit_conversion.hpp>

namespace autoware::cuda_pointcloud_preprocessor
{

using autoware::vehicle_info_utils::VehicleInfoUtils;
using autoware_utils::deg2rad;
CudaScanGroundSegmentationFilterNode::CudaScanGroundSegmentationFilterNode(
  const rclcpp::NodeOptions & options)
: Node("cuda_scan_ground_segmentation_filter_node", options)
{
  // Declare parameters
  FilterParameters filter_parameters;

  // common parameters
  filter_parameters.radial_divider_angle_rad =
    static_cast<float>(deg2rad(declare_parameter<double>("radial_divider_angle_deg")));
  filter_parameters.radial_dividers_num =
    std::ceil(2.0 * M_PI / filter_parameters.radial_divider_angle_rad);

  // common thresholds
  filter_parameters.global_slope_max_angle_rad =
    static_cast<float>(deg2rad(declare_parameter<double>("global_slope_max_angle_deg")));
  filter_parameters.local_slope_max_angle_rad =
    static_cast<float>(deg2rad(declare_parameter<double>("local_slope_max_angle_deg")));
  filter_parameters.global_slope_max_ratio = std::tan(filter_parameters.global_slope_max_angle_rad);
  filter_parameters.local_slope_max_ratio = std::tan(filter_parameters.local_slope_max_angle_rad);
  filter_parameters.split_points_distance_tolerance =
    static_cast<float>(declare_parameter<double>("split_points_distance_tolerance"));

  // vehicle info
  filter_parameters.vehicle_info = VehicleInfoUtils(*this).getVehicleInfo();

  // non-grid parameters
  filter_parameters.use_virtual_ground_point = declare_parameter<bool>("use_virtual_ground_point");
  filter_parameters.split_height_distance =
    static_cast<float>(declare_parameter<double>("split_height_distance"));

  // grid mode parameters
  filter_parameters.use_recheck_ground_cluster =
    declare_parameter<bool>("use_recheck_ground_cluster");
  filter_parameters.recheck_start_distance =
    static_cast<float>(declare_parameter<double>("recheck_start_distance"));
  filter_parameters.use_lowest_point = declare_parameter<bool>("use_lowest_point");
  filter_parameters.detection_range_z_max =
    static_cast<float>(declare_parameter<double>("detection_range_z_max"));
  filter_parameters.low_priority_region_x =
    static_cast<float>(declare_parameter<double>("low_priority_region_x"));
  filter_parameters.center_pcl_shift =
    static_cast<float>(declare_parameter<double>("center_pcl_shift"));
  filter_parameters.non_ground_height_threshold =
    static_cast<float>(declare_parameter<double>("non_ground_height_threshold"));

  // grid parameters
  filter_parameters.grid_size_m = static_cast<float>(declare_parameter<double>("grid_size_m"));
  filter_parameters.grid_mode_switch_radius =
    static_cast<float>(declare_parameter<double>("grid_mode_switch_radius"));
  filter_parameters.gnd_grid_buffer_size = declare_parameter<int>("gnd_grid_buffer_size");
  filter_parameters.virtual_lidar_z = filter_parameters.vehicle_info.vehicle_height_m;

  int64_t max_mem_pool_size_in_byte =
    declare_parameter<int64_t>("max_mem_pool_size_in_byte", 1e9);  // 1 GB
  // Initialize CUDA blackboard subscriber
  sub_ =
    std::make_shared<cuda_blackboard::CudaBlackboardSubscriber<cuda_blackboard::CudaPointCloud2>>(
      *this, "~/input/pointcloud",
      std::bind(
        &CudaScanGroundSegmentationFilterNode::cudaPointCloudCallback, this,
        std::placeholders::_1));

  // Initialize CUDA blackboard publisher
  pub_ =
    std::make_unique<cuda_blackboard::CudaBlackboardPublisher<cuda_blackboard::CudaPointCloud2>>(
      *this, "~/output/pointcloud");

  cuda_ground_segmentation_filter_ = std::make_unique<CudaScanGroundSegmentationFilter>(
    filter_parameters, max_mem_pool_size_in_byte);
}

void CudaScanGroundSegmentationFilterNode::cudaPointCloudCallback(
  const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & msg)
{
  // Process the incoming point cloud message using the CUDA ground segmentation filter
  auto non_ground_pointcloud_ptr_ = cuda_ground_segmentation_filter_->classifyPointcloud(msg);

  // Publish the filtered point cloud message
  pub_->publish(std::move(non_ground_pointcloud_ptr_));
}

}  // namespace autoware::cuda_pointcloud_preprocessor

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(
  autoware::cuda_pointcloud_preprocessor::CudaScanGroundSegmentationFilterNode)
