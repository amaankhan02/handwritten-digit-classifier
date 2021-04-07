//
// Created by amaan on 4/6/2021.
//
#include "core/model.h"

#include <string>
#include <sstream>


using std::vector;
using std::string;

namespace naivebayes {
namespace core {

Model::Model(size_t input_dim_width, size_t input_dim_height,
             vector<int> label_types, float laplace_smoothing_constant) {
  _laplace_smooth_constant = laplace_smoothing_constant;
  _feature_probs =
      vector<vector<vector<vector<float>>>>();  // TODO: init this with 0s -
                                                // ERROR would occur otherwise
  InitializeFeatureProbs(); // initialize with default values
  _prior_probs = vector<float>(label_types.size(), 0.0f);  // init with 0s
  _label_types = label_types;
  _imgs = vector<Image>();  // init to empty vector
  _labels = vector<int>();  // init to empty vector
  _input_dim_width = input_dim_width;
  _input_dim_height = input_dim_height;
}
void Model::CalculatePriorProbabilities() {
  for (int label_type : _label_types) {
    float numerator =
        _laplace_smooth_constant +
        std::count(_labels.begin(), _labels.end(), label_type) / _labels.size();
    float denominator =
        _label_types.size() * _laplace_smooth_constant + _labels.size();
    _prior_probs[label_type] = numerator / denominator;
  }
}
void Model::Train(std::vector<Image> imgs, std::vector<int> labels) {
  if (imgs.size() != labels.size()) {
    throw std::invalid_argument("Invalid data: sizes do not match");
  }
  _imgs = imgs;
  _labels = labels;

  CalculateFeatureProbabilities();
  CalculatePriorProbabilities();
}

size_t Model::GetCountForFeatures(int label, Pixel shade, size_t row,
                                  size_t column) {
  size_t count = 0;

  // data_i = index for the current image label pair
  for (size_t data_i = 0; data_i < _imgs.size(); data_i++) {
    if (_imgs[data_i].GetPixel(row, column) == shade &&
        _labels[data_i] == label) {
      count++;
    }
  }

  return count;
}
void Model::CalculateFeatureProbabilities() {
  size_t num_shades = kPixelTypes.size();
  // calculate feature probabilities
  for (int label_type : _label_types) {
    for (size_t row = 0; row < _input_dim_height; row++) {
      for (size_t column = 0; column < _input_dim_width; column++) {
        for (const auto& pixel : kPixelTypes) {
          size_t pixel_index = pixel.second;  // returns value in key-value pair
          float numerator =
              _laplace_smooth_constant +
              GetCountForFeatures(label_type, pixel.first, row, column);
          float denominator =
              num_shades * _laplace_smooth_constant +
              std::count(_labels.begin(), _labels.end(), label_type);

          _feature_probs[label_type][row][column][pixel_index] =
              numerator / denominator;
        }
      }
    }
  }
}
std::ostream& operator<<(std::ostream& os, const Model& model) {
  // store prior probabilties
  for (auto prob : model._prior_probs) {
    os << prob << std::endl;
  }

  os << model.kPriorAndFeatureProbDelimiter << std::endl;

  // store feature probabilities
  for (auto label : model._label_types) {
    for (size_t row = 0; row < model._input_dim_height; row++) {
      for (size_t column = 0; column < model._input_dim_width; column++) {
        for (size_t pixel_type_i = 0; pixel_type_i < model.kPixelTypes.size();
             pixel_type_i++) {
          float value = model._feature_probs[label][row][column][pixel_type_i];
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
    os << model.kLabelFeatureMapDelimiter << std::endl;  // delimiter to signify new feature map for a new label
  }

  return os;
}
std::istream& operator>>(std::istream& is, Model& model) {
  string line;
  // read prior probabilities
  while (std::getline(is, line)) {
    if (line.length() == 1 && line[0] == model.kPriorAndFeatureProbDelimiter) {
      break;
    }
    model._prior_probs.push_back(std::stoi(line));
  }

  // read feature probabilties
  size_t label_type = 0;
  size_t row = 0;
  while (std::getline(is, line)) {
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
      model.SplitString(pixel, model.kPixelTypeProbDelimiter, shade_probabilities);

      size_t pixel_shade_i = 0;
      for (string prob_string : shade_probabilities) {
        float prob_float = std::stof(prob_string);
        model._feature_probs[label_type][row][column][pixel_shade_i] = prob_float;
        pixel_shade_i++;
      }
      column++;
    }
    row++;
  }
  return is;
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
void Model::InitializeFeatureProbs() {
  for (size_t label_i = 0; label_i < _label_types.size(); label_i++) {
    _feature_probs.emplace_back();
    for (size_t row = 0; row < _input_dim_height; row++) {
      _feature_probs[label_i].emplace_back();
      for (size_t col = 0; col < _input_dim_width; col++) {
        _feature_probs[label_i][row].emplace_back();
        for (size_t pixel_shade = 0; pixel_shade < kPixelTypes.size(); pixel_shade++) {
          _feature_probs[label_i][row][col].emplace_back();
        }
      }
    }
  }
}

}  // namespace core
}  // namespace naivebayes