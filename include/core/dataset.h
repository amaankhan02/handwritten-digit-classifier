//
// Created by amaan on 4/6/2021.
//
#include <core/image.h>
#include <core/pixel.h>

#include <string>
#include <vector>

#ifndef NAIVE_BAYES_DATASET_H
#define NAIVE_BAYES_DATASET_H

namespace naivebayes {
namespace core {

class Dataset {
 public:
  Dataset(size_t input_height, size_t input_width); // TODO: make this automatic later --> automatically read from file
  friend std::istream &operator>>(std::istream &is, Dataset &dataset);

  const Image& GetImage(size_t index);
  int GetLabel(size_t index);

 private:
  size_t _input_height;
  size_t _input_width;
  std::vector<Image> _images;
  std::vector<int> _labels;

  /**
   * Parses the char value into a Pixel.
   * NOTE: Gray and Black pixels are converted to Black
   * @param value
   * @return
   */
  Pixel ParsePixel(char value);

};

}  // namespace core
}  // namespace naivebayes
#endif  // NAIVE_BAYES_DATASET_H
