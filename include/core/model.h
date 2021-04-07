//
// Created by amaan on 4/6/2021.
//

#ifndef NAIVE_BAYES_MODEL_H
#define NAIVE_BAYES_MODEL_H
#include <core/dataset.h>

#include <map>

namespace naivebayes {
namespace core {

/**
 * A Naive Bayes model to train and predict
 */
class Model {
 public:
  /**
   * Initialize model
   * @param input_dim_width             width of input image
   * @param input_dim_height            height of input image
   * @param label_types                 the possible output types from the model
   * @param laplace_smoothing_constant  laplace smoothing constant
   */
  Model(size_t input_dim_width, size_t input_dim_height,
        std::vector<int> label_types, float laplace_smoothing_constant);

  /**
   * Train Naive Bayes Model on the training data provided. Calculate
   * the feature and prior probabilities for the model.
   * @param imgs    training images (inputs)
   * @param labels  training labels (outputs)
   */
  void Train(std::vector<Image> imgs, std::vector<int> labels);

  /**
   * Save model probabilities to stream
   * @param os output stream
   * @param model model
   * @return ostream
   */
  friend std::ostream &operator<<(std::ostream &os, const Model &model);
  /**
   * Load model probabibilites from file to Model object
   * @param is      input stream
   * @param model   model object to load into
   * @return    input stream
   */
  friend std::istream &operator>>(std::istream &is, Model &model);

  /**
   * Get feature probabilities.
   * The vector dimensionality has [label_type, row, col, pixel_shade]
   * @return
   */
  const std::vector<std::vector<std::vector<std::vector<float>>>>
      &GetFeatureProbabilities();
  const std::vector<int> &GetPriorProbabilities();

 private:
  const std::map<Pixel, size_t> kPixelTypes = {{kWhite, 0}, {kBlack, 1}};

  /** Used for saving values to output stream **/
  const char kPixelTypeProbDelimiter = ',';

  /** Used for saving values to output stream -- from each pixel (which contains
   * the different probabilties like shaded prob and unshaded prob) **/
  const char kPixelDelimiter = '|';

  /** Used for saving values to output stream -- splits each feature map
   * for each label. i.e., signifies when it should go to the new index for the
   * outer most dimension**/
  const char kLabelFeatureMapDelimiter = '.';

  /** Splits the prior probabilities and the Feature probabilties **/
  const char kPriorAndFeatureProbDelimiter = '*';

  float _laplace_smooth_constant;
  std::vector<Image> _imgs;
  std::vector<int> _labels;
  std::vector<int> _label_types;
  size_t _input_dim_width;
  size_t _input_dim_height;

  /** dimension types are [label_type, row, col, pixel_type]**/
  std::vector<std::vector<std::vector<std::vector<float>>>> _feature_probs;
  std::vector<float> _prior_probs;

  void CalculatePriorProbabilities();
  void CalculateFeatureProbabilities();
  size_t GetCountForFeatures(int label, Pixel shade, size_t row, size_t column);
  void InitializeFeatureProbs();

  /**
   * Split string off delimeter into vector of strings
   * Method derived from
   * https://www.techiedelight.com/split-string-cpp-using-delimiter/
   * @param str
   * @param delim
   * @param out
   */
  void SplitString(const std::string &str, const char delim,
                   std::vector<std::string> &out);
};
}  // namespace core
}  // namespace naivebayes
#endif  // NAIVE_BAYES_MODEL_H
