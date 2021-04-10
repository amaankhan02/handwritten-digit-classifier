//
// Created by amaan on 4/6/2021.
//
#include <core/image.h>

#include <vector>

using naivebayes::core::Pixel;
using std::vector;

namespace naivebayes {
namespace core {

Image::Image(size_t height, size_t width, Pixel default_pixel) {
  _height = height;
  _width = width;
  _image = vector<vector<Pixel>>(height);

  // construct a blank image with all default pixels
  for (size_t row = 0; row < height; row++) {
    for (size_t column = 0; column < width; column++) {
      _image[row].push_back(default_pixel);
    }
  }
}

Image::Image(std::vector<std::vector<Pixel>>& image_vector) {
  _image = image_vector;
  _height = _image.size();
  _width = _image[0].size();
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
bool Image::SetPixel(const Pixel& new_pixel, size_t row, size_t column) {
  if (row < _height && column < _width) {  // check if valid indices
    _image[row][column] = new_pixel;
    return true;
  }
  return false;
}
const std::vector<std::vector<Pixel>>& Image::GetImageAsVector() {
  return _image;
}

}  // namespace core
}  // namespace naivebayes