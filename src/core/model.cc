//
// Created by amaan on 4/6/2021.
//
#include "core/model.h"

#include <math.h>

#include <sstream>
#include <string>

using std::string;
using std::vector;

namespace naivebayes {
namespace core {

Model::Model(size_t input_dim_width, size_t input_dim_height,
             vector<int> label_types, float laplace_smoothing_constant) {
  laplace_smooth_constant_ = laplace_smoothing_constant;
  labels_ = vector<int>();  // init to empty vector
  label_types_ = label_types;
  imgs_ = vector<Image>();  // init to empty vector
  input_dim_width_ = input_dim_width;
  input_dim_height_ = input_dim_height;

  InitializePriorProbilities();
  InitializeFeatureProbabilities();
}

void Model::Train(std::vector<Image> images, std::vector<int> labels) {
  if (images.size() != labels.size()) {
    throw std::invalid_argument("Invalid data: sizes do not match");
  }
  imgs_ = images;
  labels_ = labels;

  CalculatePriorProbabilities();
  CalculateFeatureProbabilities();
}

int Model::Predict(Image input_img) {
  if (input_img.GetHeight() != input_dim_height_ ||
      input_img.GetWidth() != input_dim_width_) {
    throw std::invalid_argument("Invalid image dimensions");
  }
  vector<float> likelihood_scores(label_types_.size());

  for (auto class_label : label_types_) {
    likelihood_scores[class_label] = log(prior_probs_[class_label]);
    for (size_t row = 0; row < input_img.GetHeight(); row++) {
      for (size_t column = 0; column < input_img.GetWidth(); column++) {
        // get type of pixel (i.e. black or white) and its respective index
        size_t shaded_index = kPixelTypes.at(input_img.GetPixel(row, column));
        likelihood_scores[class_label] +=
            log(feature_probs_[class_label][row][column][shaded_index]);
      }
    }
  }

  // index = class_label SO index of max likelihood = most likely class label
  return GetMaxIndex(likelihood_scores);
}

float Model::ComputeAccuracy(std::vector<Image> images,
                             std::vector<int> correct_labels) {
  if (images.size() != correct_labels.size()) {
    throw std::invalid_argument(
        "Cannot compute accuracy because size of images and correct_labels are "
        "not same");
  } else if (images.empty()) {
    // if size==0 then divide by 0 occurs when calculating percent on last line
    // of this method so prevent a return of nan
    throw std::invalid_argument(
        "Cannot compute accuracy on empty vector of images/labels");
  }
  size_t num_correct = 0;  // number of correct predictions
  for (size_t i = 0; i < images.size(); i++) {
    num_correct += (Predict(images[i]) == correct_labels[i]) ? 1 : 0;
  }

  return static_cast<float>(num_correct) / images.size();
}

// save model
std::ostream& operator<<(std::ostream& os, const Model& model) {
  // store prior probabilties
  for (auto prob : model.prior_probs_) {
    os << prob << std::endl;
  }

  os << model.kPriorAndFeatureProbDelimiter << std::endl;;

  // store feature probabilities
  for (auto label : model.label_types_) {
    for (size_t row = 0; row < model.input_dim_height_; row++) {
      for (size_t column = 0; column < model.input_dim_width_; column++) {
        for (size_t pixel_type_i = 0; pixel_type_i < model.kPixelTypes.size();
             pixel_type_i++) {
          float value = model.feature_probs_[label][row][column][pixel_type_i];
          if (pixel_type_i <
              model.kPixelTypes.size() - 1) {  // if not the last pixel type
            os << value << model.kPixelTypeProbDelimiter;
          } else {  // last pixel type value to output for this iteration
            os << value << model.kPixelDelimiter;
          }
        }
      }
      os << std::endl;  // add new line after each row
    }
    os << model.kLabelFeatureMapDelimiter
       << std::endl;  // delimiter to signify new feature map for a new label
  }

  return os;
}

// load model
std::istream& operator>>(std::istream& is, Model& model) {
  string line;
  model.LoadPriorProbilities(is, model, line);
  model.LoadFeatureProbilities(is, model, line);
  return is;
}

void Model::CalculatePriorProbabilities() {
  for (int label_type : label_types_) {
    float numerator = laplace_smooth_constant_ +
                      std::count(labels_.begin(), labels_.end(), label_type);
    float denominator =
        label_types_.size() * laplace_smooth_constant_ + labels_.size();
    prior_probs_[label_type] = numerator / denominator;
  }
}

size_t Model::GetCountForFeatures(int label, Pixel shade, size_t row,
                                  size_t column) {
  size_t count = 0;

  // data_i = index for the current image label pair
  for (size_t data_i = 0; data_i < imgs_.size(); data_i++) {
    if (imgs_[data_i].GetPixel(row, column) == shade &&
        labels_[data_i] == label) {
      count++;
    }
  }

  return count;
}

void Model::CalculateFeatureProbabilities() {
  size_t num_shades = kPixelTypes.size();
  // calculate feature probabilities
  for (int label_type : label_types_) {
    for (size_t row = 0; row < input_dim_height_; row++) {
      for (size_t column = 0; column < input_dim_width_; column++) {
        for (const auto& pixel : kPixelTypes) {
          size_t pixel_index = pixel.second;  // returns value in key-value pair
          float numerator =
              laplace_smooth_constant_ +
              GetCountForFeatures(label_type, pixel.first, row, column);
          float denominator =
              num_shades * laplace_smooth_constant_ +
              std::count(labels_.begin(), labels_.end(), label_type);

          feature_probs_[label_type][row][column][pixel_index] =
              (numerator / denominator);
        }
      }
    }
  }
}

void Model::SplitString(const string& str, const char delim,
                        vector<std::string>& out) {
  // construct a stream from the string
  std::stringstream ss(str);

  std::string s;
  while (std::getline(ss, s, delim)) {
    out.push_back(s);
  }
}

void Model::InitializeFeatureProbabilities() {
  feature_probs_ = vector<vector<vector<vector<float>>>>();

  // resize 4d vector to correct dimensions
  for (size_t label_i = 0; label_i < label_types_.size(); label_i++) {
    feature_probs_.emplace_back();
    for (size_t row = 0; row < input_dim_height_; row++) {
      feature_probs_[label_i].emplace_back();
      for (size_t col = 0; col < input_dim_width_; col++) {
        feature_probs_[label_i][row].emplace_back();
        for (size_t pixel_shade = 0; pixel_shade < kPixelTypes.size();
             pixel_shade++) {
          feature_probs_[label_i][row][col].emplace_back();
        }
      }
    }
  }
}

const std::vector<std::vector<std::vector<std::vector<float>>>>&
Model::GetFeatureProbabilities() {
  return feature_probs_;
}

const std::vector<float>& Model::GetPriorProbabilities() {
  return prior_probs_;
}

int Model::GetMaxIndex(std::vector<float> vec) {
  size_t max_index = 0;
  for (size_t i = 1; i < vec.size(); i++) {
    if (vec[i] > vec[max_index]) {
      max_index = i;
    }
  }

  return max_index;
}

void Model::InitializePriorProbilities() {
  prior_probs_ = vector<float>(label_types_.size(), 0.0f);  // init with 0s
}

void Model::LoadPriorProbilities(std::istream& in_stream, Model& model, std::string& line) {
  size_t prior_prob_index = 0;
  while (std::getline(in_stream, line)) {
    if (line.length() == 1 && line[0] == model.kPriorAndFeatureProbDelimiter) {
      break;
    }
    model.prior_probs_[prior_prob_index] = std::stof(line);
    prior_prob_index++;
  }
}
void Model::LoadFeatureProbilities(std::istream& in_stream, Model& model,
                                   string& line) {
  size_t label_type = 0;
  size_t row = 0;
  while (std::getline(in_stream, line)) {
    if (line[0] == model.kLabelFeatureMapDelimiter) {
      label_type++;
      row = 0;
      continue;
    }

    vector<string> pixels = vector<string>();
    model.SplitString(line, model.kPixelDelimiter, pixels);

    size_t column = 0;
    for (string pixel : pixels) {
      vector<string> shade_probabilities = vector<string>();
      model.SplitString(pixel, model.kPixelTypeProbDelimiter,
                        shade_probabilities);

      size_t pixel_shade_i = 0;
      for (string prob_string : shade_probabilities) {
        float prob_float = std::stof(prob_string);
        model.feature_probs_[label_type][row][column][pixel_shade_i] =
            prob_float;
        pixel_shade_i++;
      }
      column++;
    }
    row++;
  }
}

}  // namespace core
}  // namespace naivebayes