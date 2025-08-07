
#include "autoware/cuda_pointcloud_preprocessor/cuda_ground_segmentation/cuda_scan_ground_segmentation_filter_node.hpp"

#include "autoware/pointcloud_preprocessor/utility/memory.hpp"

namespace autoware::cuda_pointcloud_preprocessor
{
CudaScanGroundSegmentationFilterNode::CudaScanGroundSegmentationFilterNode(
  const rclcpp::NodeOptions & options)
: Node("cuda_scan_ground_segmentation_filter_node", options)
{
  float height_threshold = static_cast<float>(declare_parameter<double>("height_threshold", 0.0));
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

  cuda_ground_segmentation_filter_ =
    std::make_unique<CudaScanGroundSegmentationFilter>(height_threshold, max_mem_pool_size_in_byte);
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
