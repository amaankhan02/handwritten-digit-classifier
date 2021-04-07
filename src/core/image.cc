//
// Created by amaan on 4/6/2021.
//
#include <core/image.h>
#include <vector>

using std::vector;
using naivebayes::core::Pixel;

namespace naivebayes {
namespace core {

Image::Image(size_t height, size_t width, Pixel default_pixel) {
  _height = height;
  _width = width;
  _image = vector<vector<Pixel>>(height);

  // construct a blank image with all default pixels
  for (size_t row = 0; row < height; row++) {
    for (size_t column = 0; column < width; column++) {
      _image[row][column] = default_pixel;
    }
  }
}
size_t Image::GetHeight() {
  return _height;
}
size_t Image::GetWidth() {
  return _width;
}
const Pixel& Image::GetPixel(size_t row, size_t column) const {
  return _image[row][column];
}
bool Image::SetPixel(const Pixel &new_pixel, size_t row, size_t column) {
  if (row < _height && column < _width) { // check if valid indices
    _image[row][column] = new_pixel;
    return true;
  }
  return false;
}
const std::vector<std::vector<Pixel>>& Image::GetImageAsVector() {
  return _image;
}
//void Image::AddRow(std::vector<char> row) {
//  // make sure 1 more row won't exceed the height limit
//  if ()
//}
}
}