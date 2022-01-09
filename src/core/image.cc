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
  height_ = height;
  width_ = width;
  image_ = vector<vector<Pixel>>(height);
  default_pixel_ = default_pixel;

  // construct a blank image with all default pixels
  for (size_t row = 0; row < height; row++) {
    for (size_t column = 0; column < width; column++) {
      image_[row].push_back(default_pixel);
    }
  }
}

Image::Image(std::vector<std::vector<Pixel>>& image_vector, Pixel default_pixel) {
  image_ = image_vector;
  height_ = image_.size();
  width_ = image_[0].size();
  default_pixel_ = default_pixel;
}

size_t Image::GetHeight() {
  return height_;
}
size_t Image::GetWidth() {
  return width_;
}
const Pixel& Image::GetPixel(size_t row, size_t column) const {
  return image_[row][column];
}
bool Image::SetPixel(const Pixel& new_pixel, size_t row, size_t column) {
  if (row < height_ && column < width_) {  // check if valid indices
    image_[row][column] = new_pixel;
    return true;
  }
  return false;
}
const std::vector<std::vector<Pixel>>& Image::GetImageAsVector() {
  return image_;
}
void Image::Clear() {
  for (size_t row = 0; row < height_; row++) {
    for (size_t column = 0; column < width_; column++) {
      image_[row][column] = default_pixel_;
    }
  }
}

}  // namespace core
}  // namespace naivebayes