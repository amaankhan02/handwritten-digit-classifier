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
      dataset._labels.push_back(static_cast<int>(line[0]));
      dataset._images.push_back(Image(dataset._input_height, dataset._input_width));
      img_row = 0;
    } else {
      for (size_t column = 0; column < line.length(); column++) {
        Pixel pixel = dataset.ParsePixel(line[column]);
        dataset._images.back().SetPixel(pixel, img_row, column); // TODO: throw error if #columns > width
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

}
}  // namespace naivebayes
