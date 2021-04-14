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
   * @param images  training images (inputs)
   * @param labels  training labels (outputs)
   */
  void Train(std::vector<Image> images, std::vector<int> labels);

  /**
   * Model predicts on all the images passed in and computes the accuracy by
   * calculating the ratio of the number of correct predictions over the
   * number of total predictions.
   *
   * @param images          vector of images to predict on and compute accuracy
   *                        of
   * @param correct_labels  the correct labels corresponding to the the vector
   *                        of input images
   * @return
   */
  float ComputeAccuracy(std::vector<Image> images,
                        std::vector<int> correct_labels);

  /**
   * Given an image, model predicts/classifies what the output
   * should be by finding the likelihood scores and returning the
   * most probable label.
   *
   * @param input_img   Image to classify/predict on
   * @return            A predicted label of the image
   */
  int Predict(Image input_img);

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
   * @return    feature probabilities
   */
  const std::vector<std::vector<std::vector<std::vector<float>>>>
      &GetFeatureProbabilities();

  /**
   * Get prior probabilities
   * @return prior probabiltiies
   */
  const std::vector<float> &GetPriorProbabilities();

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

  float laplace_smooth_constant_;
  std::vector<Image> imgs_;
  std::vector<int> labels_;
  std::vector<int> label_types_;
  size_t input_dim_width_;
  size_t input_dim_height_;

  /** dimension types are [label_type, row, col, pixel_type]**/
  std::vector<std::vector<std::vector<std::vector<float>>>> feature_probs_;
  std::vector<float> prior_probs_;

  void CalculatePriorProbabilities();
  void CalculateFeatureProbabilities();
  size_t GetCountForFeatures(int label, Pixel shade, size_t row, size_t column);
  void InitializeFeatureProbabilities();
  void InitializePriorProbilities();
  void LoadPriorProbilities(std::istream& in_stream, Model& model, std::string& line);
  void LoadFeatureProbilities(std::istream& in_stream, Model& model, std::string& line);


  /**
   * Split string off delimiter into vector of strings
   * Method derived from
   * https://www.techiedelight.com/split-string-cpp-using-delimiter/
   * @param str
   * @param delim
   * @param out
   */
  void SplitString(const std::string &str, const char delim,
                   std::vector<std::string> &out);
  int GetMaxIndex(std::vector<float> vec);
};
}  // namespace core
}  // namespace naivebayes
#endif  // NAIVE_BAYES_MODEL_H
