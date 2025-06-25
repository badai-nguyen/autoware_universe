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

#include "perception_utils/run_length_encoder.hpp"

#include <vector>

namespace perception_utils
{
std::vector<uint8_t> serializeRLEEntry(const std::vector<RLEEntry> & rle_data)
{
  std::vector<uint8_t> buffer;
  for (const auto & entry : rle_data) {
    // 1. pushback label value
    buffer.push_back(entry.label_value);
    // 2. pushback run length (int)
    int run = entry.run_length;
    uint8_t * run_ptr = reinterpret_cast<uint8_t *>(&run);
    buffer.insert(buffer.end(), run_ptr, run_ptr + sizeof(int));
    // 3. pushback label name length
    int label_name_length = entry.label_name.size();
    uint8_t * label_name_length_ptr = reinterpret_cast<uint8_t *>(&label_name_length);
    buffer.insert(buffer.end(), label_name_length_ptr, label_name_length_ptr + sizeof(int));
    // 4. pushback label name
    buffer.insert(buffer.end(), entry.label_name.begin(), entry.label_name.end());
  }
  return buffer;
}

std::vector<uint8_t> runLengthEncoder(
  const cv::Mat & image, const std::vector<std::string> & label_names)
{
  std::vector<RLEEntry> compressed_data;
  const int rows = image.rows;
  const int cols = image.cols;
  uint8_t current_value = image.at<uint8_t>(0, 0);
  RLEEntry entry;
  entry.label_value = current_value;
  entry.run_length = 0;
  entry.label_name = current_value < label_names.size() ? label_names[current_value] : "unknown";
  compressed_data.push_back(
    {current_value, 0,
     current_value < label_names.size() ? label_names[current_value] : "unknown"});
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      current_value = image.at<uint8_t>(i, j);
      if (compressed_data.back().label_value == current_value) {
        ++compressed_data.back().run_length;
      } else {
        compressed_data.push_back(
          {current_value, 1,
           current_value < label_names.size() ? label_names[current_value] : "unknown"});
      }
    }
  }
  return serializeRLEEntry(compressed_data);
}

cv::Mat runLengthDecoder(
  const std::vector<uint8_t> & rle_data, const int rows, const int cols,
  std::vector<std::string> & label_names)
{
  label_names.clear();
  cv::Mat mask(rows, cols, CV_8UC1, cv::Scalar(0));
  size_t offset = 0;
  int row = 0;
  int col = 0;
  while (offset + sizeof(uint8_t) + sizeof(int) + sizeof(int) < rle_data.size()) {
    uint8_t label_value;
    int length;
    int label_name_length;
    std::string label_name;

    std::memcpy(&label_value, &rle_data[offset], sizeof(uint8_t));
    offset += sizeof(uint8_t);
    std::memcpy(&length, &rle_data[offset], sizeof(int));
    offset += sizeof(int);
    std::memcpy(&label_name_length, &rle_data[offset], sizeof(int));
    offset += sizeof(int);
    if (label_name_length < 0 || offset + label_name_length > rle_data.size()) {
      throw std::runtime_error("Invalid RLE data: label name length exceeds data size.");
    }
    label_name =
      std::string(rle_data.begin() + offset, rle_data.begin() + offset + label_name_length);
    offset += label_name_length;
    label_names.push_back(label_name);
    for (int i = 0; i < length; ++i) {
      if (row >= rows || col >= cols) {
        throw std::runtime_error("RLE data exceeds mask dimensions.");
      }
      mask.at<uint8_t>(row, col) = label_value;
      ++col;
      if (col >= cols) {
        col = 0;
        ++row;
      }
    }
  }
  return mask;
}

}  // namespace perception_utils
