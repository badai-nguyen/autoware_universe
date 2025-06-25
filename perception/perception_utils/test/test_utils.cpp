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

#include <opencv2/opencv.hpp>

#include <gtest/gtest.h>

#include <vector>

// Test case 1: Test if the decoded image is the same as the original image
TEST(UtilsTest, runLengthEncoderDecoderTest)
{
  int height = 10;
  int width = 20;
  uint8_t number_cls = 16;
  // Create an image as below
  // 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  // 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  // 2 2 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  // 3 3 3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  // 4 4 4 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  // 5 5 5 5 5 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  // 6 6 6 6 6 6 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  // 7 7 7 7 7 7 7 0 0 0 0 0 0 0 0 0 0 0 0 0
  // 8 8 8 8 8 8 8 8 0 0 0 0 0 0 0 0 0 0 0 0
  // 9 9 9 9 9 9 9 9 9 0 0 0 0 0 0 0 0 0 0 0
  cv::Mat image = cv::Mat::zeros(10, 20, CV_8UC1);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      if (j < i) {
        image.at<uint8_t>(i, j) = i % number_cls;
      } else {
        image.at<uint8_t>(i, j) = 0;
      }
    }
  }
  // Compress the image
  std::vector<std::string> encoded_label_names = {
    "class_0", "class_1", "class_2",  "class_3",  "class_4",  "class_5",  "class_6",  "class_7",
    "class_8", "class_9", "class_10", "class_11", "class_12", "class_13", "class_14", "class_15"};
  std::vector<uint8_t> compressed_data =
    perception_utils::runLengthEncoder(image, encoded_label_names);
  // Decompress the image
  std::vector<std::string> decoded_label_names;
  cv::Mat decoded_image =
    perception_utils::runLengthDecoder(compressed_data, height, width, decoded_label_names);
  // Compare the original image and the decoded image
  ASSERT_EQ(image.rows, decoded_image.rows);
  ASSERT_EQ(image.cols, decoded_image.cols);
  bool image_eq = true;
  for (int i = 0; i < image.rows; ++i) {
    for (int j = 0; j < image.cols; ++j) {
      if (image.at<uint8_t>(i, j) != decoded_image.at<uint8_t>(i, j)) {
        image_eq = false;
        break;
      }
    }
  }
  EXPECT_EQ(image_eq, true);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
