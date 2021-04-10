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

/**
 * Class to store input and output vector of data for a machine learning model
 */
class Dataset {
 public:
  /**
   * Initialize dataset with the input height and width that is required
   * for each data
   * @param input_height    input height
   * @param input_width     input width
   */
  Dataset(size_t input_height, size_t input_width);

  /**
   * Load dataset from input stream
   * @param is  input stream
   * @param dataset     dataset object to load into
   * @return    input stream
   */
  friend std::istream &operator>>(std::istream &is, Dataset &dataset);

  const Image& GetImage(size_t index);
  int GetLabel(size_t index);

  const std::vector<Image>& GetImages();
  const std::vector<int>& GetLabels();

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
