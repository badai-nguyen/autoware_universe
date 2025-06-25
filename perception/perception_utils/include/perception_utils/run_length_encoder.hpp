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

#ifndef PERCEPTION_UTILS__RUN_LENGTH_ENCODER_HPP_

#define PERCEPTION_UTILS__RUN_LENGTH_ENCODER_HPP_
#include <opencv2/opencv.hpp>

#include <utility>
#include <vector>

namespace perception_utils
{
struct RLEEntry {
  uint8_t label_value;
  int run_length;
  std::string label_name;
};

std::vector<uint8_t> runLengthEncoder(const cv::Mat & mask, 
                                      const std::vector<std::string> & label_names);
std::vector<uint8_t> serializeRLEEntry(const std::vector<RLEEntry> & rle_data);
cv::Mat runLengthDecoder(const std::vector<uint8_t> & rle_data, const int rows, const int cols, 
                         std::vector<std::string> & label_names);
}  // namespace perception_utils

#endif  // PERCEPTION_UTILS__RUN_LENGTH_ENCODER_HPP_
