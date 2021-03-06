//
// Created by amaan on 4/6/2021.
//
#include <core/pixel.h>
#include <string>
#include <vector>

#ifndef NAIVE_BAYES_IMAGE_H
#define NAIVE_BAYES_IMAGE_H

namespace naivebayes {
namespace core {
class Image {
 public:
  /**
   * Initialize a blank image with all pixels equal
   * to the default pixel that is passed in
   *
   * @param height          height of the image
   * @param width           width of the image
   * @param default_pixel   the default pixel that you want every pixel in the
   *                        image initialize to by default
   */
  Image(size_t height, size_t width, Pixel default_pixel = kWhite);

  /**
   * Initialize image with existing vector of pixels
   * @param image_vector    vector of pixels representing image
   */
  Image(std::vector<std::vector<Pixel>> &image_vector, Pixel default_pixel = kWhite);

  size_t GetHeight();
  size_t GetWidth();
  const Pixel& GetPixel(size_t row, size_t column) const;
  const std::vector<std::vector<Pixel>>& GetImageAsVector();

  /**
   * Set a new value to the pixel in the image at coordinates [row, column]
   * @param new_pixel   new value of the pixel
   * @param row         the row the pixel is locatd at
   * @param column      the column the pixel is located at
   * @return            false if the row and/or column do not exist (out of bounds)
   *                    true if it does, hence the new pixel was set
   */
  bool SetPixel(const Pixel &new_pixel, size_t row, size_t column);

  /**
   * Clear the image by setting all pixels to the default pixel that was
   * assigned during construction of the image
   */
  void Clear();

 private:
  std::vector<std::vector<Pixel>> image_;
  size_t height_;
  size_t width_;
  Pixel default_pixel_;

};
}  // namespace core

}  // namespace naivebayes
#endif  // NAIVE_BAYES_IMAGE_H
