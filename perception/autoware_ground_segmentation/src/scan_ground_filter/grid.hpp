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

#ifndef SCAN_GROUND_FILTER__GRID_HPP_
#define SCAN_GROUND_FILTER__GRID_HPP_

#include <autoware_utils/geometry/geometry.hpp>
#include <autoware_utils/math/normalization.hpp>
#include <autoware_utils/system/time_keeper.hpp>

#include <algorithm>
#include <cmath>
#include <memory>
#include <utility>
#include <vector>

namespace
{

float pseudoArcTan2(const float y, const float x)
{
  // lightweight arc tangent

  // avoid divide-by-zero
  if (y == 0.0f) {
    if (x >= 0.0f) return 0.0f;
    return M_PIf;
  }
  if (x == 0.0f) {
    if (y >= 0.0f) return M_PI_2f;
    return -M_PI_2f;
  }

  const float x_abs = std::abs(x);
  const float y_abs = std::abs(y);

  // divide to 8 zones
  constexpr float M_2PIf = 2.0f * M_PIf;
  constexpr float M_3PI_2f = 3.0f * M_PI_2f;
  if (x_abs > y_abs) {
    const float ratio = y_abs / x_abs;
    const float angle = ratio * M_PI_4f;
    if (y >= 0.0f) {
      if (x >= 0.0f) return angle;  // 1st zone
      return M_PIf - angle;         // 2nd zone
    } else {
      if (x >= 0.0f) return M_2PIf - angle;  // 4th zone
      return M_PIf + angle;                  // 3rd zone
    }
  } else {
    const float ratio = x_abs / y_abs;
    const float angle = ratio * M_PI_4f;
    if (y >= 0.0f) {
      if (x >= 0.0f) return M_PI_2f - angle;  // 1st zone
      return M_PI_2f + angle;                 // 2nd zone
    } else {
      if (x >= 0.0f) return M_3PI_2f + angle;  // 4th zone
      return M_3PI_2f - angle;                 // 3rd zone
    }
  }
}

}  // namespace

namespace autoware::ground_segmentation
{
using autoware_utils::ScopedTimeTrack;

struct Point
{
  size_t index;
  float distance;
  float height;
};

// Concentric Zone Model (CZM) based polar grid
class Cell
{
public:
  // list of point indices
  std::vector<Point> point_list_;  // point index and distance

  // method to check if the cell is empty
  inline bool isEmpty() const { return point_list_.empty(); }

  // index of the cell
  int grid_idx_;
  int radial_idx_;
  int azimuth_idx_;
  int next_grid_idx_;
  int prev_grid_idx_;

  int scan_grid_root_idx_;

  // geometric properties of the cell
  float center_radius_;
  float center_azimuth_;
  float radial_size_;
  float azimuth_size_;

  // ground statistics of the points in the cell
  float avg_height_;
  float max_height_;
  float min_height_;
  float avg_radius_;
  float gradient_;
  float intercept_;

  // process flags
  bool is_processed_ = false;
  bool is_ground_initialized_ = false;
  bool has_ground_ = false;
};

class Grid
{
public:
  Grid(const float origin_x, const float origin_y) : origin_x_(origin_x), origin_y_(origin_y) {}
  ~Grid() = default;

  void setTimeKeeper(std::shared_ptr<autoware_utils::TimeKeeper> time_keeper_ptr)
  {
    time_keeper_ = std::move(time_keeper_ptr);
  }

  void initialize(
    const float grid_dist_size, const float sector_azimuth_size, const float grid_radial_limit)
  {
    grid_dist_size_ = grid_dist_size;
    sector_azimuth_size_ = sector_azimuth_size;

    grid_radial_limit_ = grid_radial_limit;
    grid_dist_size_inv_ = 1.0f / grid_dist_size_;
    sector_grid_max_num_ = std::ceil(grid_radial_limit * grid_dist_size_inv_) + 1;
    sector_azimuth_size_inv_ = 1.0f / sector_azimuth_size_;
    radial_sector_num_ = std::ceil(2.0f * M_PIf * sector_azimuth_size_inv_) + 1;
    std::cout<<"sector_grid_max_num_: "<<sector_grid_max_num_<<", radial_sector_num_: "<<radial_sector_num_<<std::endl;
    RCLCPP_INFO(rclcpp::get_logger("Grid"), "initialize: sector_grid_max_num_: %d, radial_sector_num_: %d", sector_grid_max_num_, radial_sector_num_);

    // generate grid geometry
    // setGridBoundaries();

    // initialize and resize cells
    cells_.clear();
    cells_.resize(static_cast<size_t>(sector_grid_max_num_ * radial_sector_num_));

    // set cell geometry
    setCellGeometry();

    // set initialized flag
    is_initialized_ = true;
  }

  bool isInitialized() const { return is_initialized_; }

  // method to add a point to the grid
  void addPoint(const float x, const float y, const float z, const size_t point_idx)
  {
    const float x_fixed = x - origin_x_;
    const float y_fixed = y - origin_y_;
    const float radius = std::sqrt(x_fixed * x_fixed + y_fixed * y_fixed);
    const float azimuth = pseudoArcTan2(y_fixed, x_fixed);
    if (radius >= grid_radial_limit_) {
      return;
    }
    if (azimuth < 0.0f || azimuth >= 2.0f * M_PIf) {
      return;
    }

    // calculate the grid id: 
    // azimuth sector 0 indexing from 0 -> sector_grid_max_num_ -1
    // azimuth sector 1 indexing from sector_grid_max_num_ -> 2*sector_grid_max_num_ -1
    // ...

    const int grid_idx = getGridIdx(radius, azimuth);

    // check if the point is within the grid
    if (grid_idx < 0) {
      return;
    }
    const size_t grid_idx_idx = static_cast<size_t>(grid_idx);

    // check bounds to prevent memory corruption
    if (grid_idx_idx >= cells_.size()) {
      // log grid_idx_idx: " << grid_idx_idx << ", cells_.size(): " << cells_.size();
      RCLCPP_INFO(rclcpp::get_logger("Grid"), "invalid index: out of bounds %zu %zu", grid_idx_idx, cells_.size());

      const int radial_idx = getRadialIdx(radius);
      const int azimuth_idx = getAzimuthSectorIdx(azimuth);
      std::cout<<"grid_idx: "<<grid_idx<<", radial_idx: "<<radial_idx<<", azimuth_idx: "<<azimuth_idx<<std::endl;
      return;
    }

    // add the point to the cell
    cells_[grid_idx_idx].point_list_.emplace_back(Point{point_idx, radius, z});
  }

  int getGridRadialMaxNum() const { return sector_grid_max_num_; }
  int getSectorAzimuthMaxNum() const { return radial_sector_num_; }

  size_t getGridSize() const { return cells_.size(); }

  // method to get the cell
  inline Cell & getCell(const int grid_idx)
  {
    if (grid_idx < 0) {
      throw std::out_of_range("Invalid grid index: negative");
    }
    const size_t idx = static_cast<size_t>(grid_idx);
    if (idx >= cells_.size()) {
      throw std::out_of_range("Invalid grid index: out of bounds");
    }
    return cells_[idx];
  }

  void resetCells()
  {
    std::unique_ptr<ScopedTimeTrack> st_ptr;
    if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

    for (auto & cell : cells_) {
      cell.point_list_.clear();
      cell.is_processed_ = false;
      cell.is_ground_initialized_ = false;
      cell.has_ground_ = false;
    }
  }

  void setGridConnections()
  {
    std::unique_ptr<ScopedTimeTrack> st_ptr;
    if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

    // Example for one azimuth sector:
    // point number :       0 1 2 0 4 5 6 0 8 0
    // radial_index:        0 1 2 3 4 5 6 7 8 9
    // cell_idx:            0 1 2 3 4 5 6 7 8 9
    // prev_grid_idx:       0 0 1 2 3 4 5 6 7 8
    // scan_grid_root_idx_: 0 0 1 2 2 4 5 6 6 7
    for (int azimuth_idx = 0; azimuth_idx < radial_sector_num_; ++azimuth_idx) {
      const int sector_base_idx = azimuth_idx * sector_grid_max_num_;
      
      // Initialize first cell in the sector (radial_idx = 0)
      auto & first_cell = cells_[sector_base_idx];
      first_cell.scan_grid_root_idx_ = sector_base_idx;
      
      // Process remaining cells in radial direction
      for (int radial_idx = 1; radial_idx < sector_grid_max_num_; ++radial_idx) {
        const int current_cell_idx = sector_base_idx + radial_idx;
        const int prev_cell_idx = sector_base_idx + (radial_idx - 1);
        
        auto & current_cell = cells_[current_cell_idx];
        const auto & prev_cell = cells_[prev_cell_idx];
        
        // If previous cell has points, it becomes the root; otherwise inherit its root
        if (!prev_cell.isEmpty()) {
          current_cell.scan_grid_root_idx_ = prev_cell_idx;
        } else {
          current_cell.scan_grid_root_idx_ = prev_cell.scan_grid_root_idx_;
        }
      }
    }
  }

private:
  // given parameters
  float origin_x_;
  float origin_y_;
  float grid_dist_size_ = 1.0f;      // meters
  float sector_azimuth_size_ = 0.01f;  // radians

  // calculated parameters
  float grid_dist_size_inv_ = 0.0f;  // inverse of the grid size in meters
  float sector_azimuth_size_inv_ = 0.0f;  // inverse of the grid size in radians
  bool is_initialized_ = false;

  // configured parameters
  float grid_radial_limit_ = 200.0f;  // meters
  int sector_grid_max_num_ = 0;
  int radial_sector_num_ = 0;

  // array of grid boundaries
  std::vector<float> grid_radial_boundaries_;
  std::vector<float> azimuth_sector_boundaries_;

  // list of cells
  std::vector<Cell> cells_;

  // debug information
  std::shared_ptr<autoware_utils::TimeKeeper> time_keeper_;

  // Generate grid geometry
  // the grid is cylindrical mesh grid
  // azimuth interval: constant angle
  // radial interval: constant distance within mode switch radius
  //                  constant elevation angle outside mode switch radius
  void setGridBoundaries()
  {
    std::unique_ptr<ScopedTimeTrack> st_ptr;
    if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

    // radial boundaries
    {
      // constant distance
      for (int i = 0; i < sector_grid_max_num_; i++) {
        grid_radial_boundaries_.push_back(i * grid_dist_size_);
      }
    }

    //  azimuth sector boundaries
    {
      for (int azimuth_sector_idx = 0; azimuth_sector_idx < radial_sector_num_;
           azimuth_sector_idx++) {
        azimuth_sector_boundaries_.push_back(azimuth_sector_idx * sector_azimuth_size_);
      }
    }
    
  }


  // int getAzimuthSectorIdx(const int & radial_idx, const float & azimuth) const
  // {
  //   const int azimuth_grid_num = azimuth_grids_per_radial_[radial_idx];

  //   int azimuth_grid_idx =
  //     static_cast<int>(std::floor(azimuth / azimuth_interval_per_radial_[radial_idx]));
  //   if (azimuth_grid_idx == azimuth_grid_num) {
  //     // loop back to the first grid
  //     azimuth_grid_idx = 0;
  //   }
  //   // constant azimuth interval
  //   return azimuth_grid_idx;
  // }

  int getRadialIdx(const float & radius) const
  {
    // check if the point is within the grid
    if (radius >= grid_radial_limit_) {
      return -1;
    }
    if (radius < 0) {
      return -1;
    }
    return static_cast<int>(radius * grid_dist_size_inv_);
  }

  int getAzimuthSectorIdx(const float azimuth) const 
  {
    if (azimuth < 0.0f || azimuth >= 2.0f * M_PIf) {
      return -1;
    }
    return static_cast<int>(azimuth * sector_azimuth_size_inv_);
    
  }

  // method to determine the grid id of a point
  // -1 means out of range
  // range limit is horizon angle
  int getGridIdx(const float & radius, const float & azimuth) const
  {
    const int grid_rad_idx = getRadialIdx(radius);
    if (grid_rad_idx < 0) {
      return -1;
    }

    // azimuth sector id
    const int sector_az_idx = getAzimuthSectorIdx(azimuth);
    if (sector_az_idx < 0) {
      return -1;
    }

    return sector_az_idx * sector_grid_max_num_ + grid_rad_idx;
  }

  void getRadialAzimuthIdxFromCellIdx(const int cell_id, int & radial_idx, int & azimuth_idx) const
  {
    radial_idx = -1;
    azimuth_idx = -1;

    azimuth_idx = static_cast<int>(cell_id / sector_grid_max_num_);
    radial_idx = cell_id % sector_grid_max_num_;
  }

  void setCellGeometry()
  {
    std::unique_ptr<ScopedTimeTrack> st_ptr;
    if (time_keeper_) st_ptr = std::make_unique<ScopedTimeTrack>(__func__, *time_keeper_);

    for (int azimuth_sector_idx = 0; azimuth_sector_idx < radial_sector_num_; azimuth_sector_idx++) {
      for (int radial_idx = 0; radial_idx < sector_grid_max_num_; radial_idx++) {
         auto idx = azimuth_sector_idx * sector_grid_max_num_ + radial_idx;
        Cell & cell = cells_[idx];

        cell.grid_idx_ = idx;
        cell.radial_idx_ = radial_idx;
        cell.azimuth_idx_ = azimuth_sector_idx;

        // set width of the cell

        cell.azimuth_size_ = sector_azimuth_size_;
        cell.radial_size_ = grid_dist_size_;

        // set center of the cell
        cell.center_radius_ =  (static_cast<float>(radial_idx) + 0.5f) * cell.radial_size_;
        cell.center_azimuth_ = (static_cast<float>(azimuth_sector_idx) + 0.5f) * cell.azimuth_size_;

        // set next grid id, which is radially next
        int next_grid_idx = -1;
        if( radial_idx + 1 < sector_grid_max_num_ ) {
          next_grid_idx = idx + 1;
        } else {
          next_grid_idx = idx;  // no next grid
        }
        cell.next_grid_idx_ = next_grid_idx;

        // set previous grid id, which is radially previous
        int prev_grid_idx = -1;
        if( radial_idx - 1 > 0 ) {
          prev_grid_idx = idx - 1;
        }
        cell.prev_grid_idx_ = prev_grid_idx;
      }
    }
  }
};

}  // namespace autoware::ground_segmentation

#endif  // SCAN_GROUND_FILTER__GRID_HPP_
