//
// Created by amaan on 4/6/2021.
//
#include <core/dataset.h>

using naivebayes::core::Pixel;
using std::vector;
using std::string;

namespace naivebayes {
namespace core {

Pixel Dataset::ParsePixel(char value) {
  if (value == kWhite) {
    return kWhite;
  } else if (value == kBlack || value == kGray) {
    return kBlack;
  } else {
    throw std::invalid_argument("Invalid pixel");
  }
}
std::istream& operator>>(std::istream& is, Dataset& dataset) {
  string line;
  size_t img_row = 0;

  while (std::getline(is, line)) { // get next line from input stream
    if (line.length() == 1) { // start of a new data pair
      dataset._labels.push_back(std::stoi(line));
      Image temp_img = Image(dataset._input_height, dataset._input_width);
      dataset._images.push_back(temp_img);

//      // check that the num rows is same as height that was provided TODO: fix this implementation later! --> and the one on line 38
//      if (!dataset._labels.empty() && img_row != dataset._input_height - 1) {
//        throw std::invalid_argument("Invalid image size provided");
//      }
      img_row = 0;
    } else {
//      if (line.length() != dataset._input_width - 1) {
//        throw std::invalid_argument("Invalid image size provided");
//      }
      for (size_t column = 0; column < line.length(); column++) {
        Pixel pixel = dataset.ParsePixel(line.at(column));
        dataset._images.back().SetPixel(pixel, img_row, column);
      }
      img_row++;
    }
  }

  return is;
}
Dataset::Dataset(size_t input_height, size_t input_width) {
  _input_height = input_height;
  _input_width = input_width;
}
const Image& Dataset::GetImage(size_t index) {
  return _images[index];
}
int Dataset::GetLabel(size_t index) {
  return _labels[index];
}
const std::vector<Image>& Dataset::GetImages() {
  return _images;
}
const std::vector<int>& Dataset::GetLabels() {
  return _labels;
}

}
}  // namespace naivebayes
