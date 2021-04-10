#include <fstream>
#include <iostream>
#include <string>
#include <tuple>

#include <core/model.h>

using naivebayes::core::Dataset;
using naivebayes::core::Model;

int main() {
  // load train_dataset from file
  Dataset train_dataset = Dataset(28, 28);
  std::string input_path;
  std::ifstream input_file(input_path);
  if (input_file.is_open()) {
    input_file >> train_dataset;
  }

  // construct model and train // TODO: remove these magic numbers!
  Model model = Model(28, 28, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 1);
  model.Train(train_dataset.GetImages(), train_dataset.GetLabels());

  // save model features to file
  std::string output_path;
  std::ofstream output_file(output_path);
  if (output_file.is_open()) {
    output_file << model;
  }

  // load test data
  std::string test_data_path;
  std::ifstream test_data_file(test_data_path);
  Dataset test_dataset = Dataset(28, 28);
  if (test_data_file.is_open()) {
    test_data_file >> test_dataset;
  }

  float accuracy = model.ComputeAccuracy(test_dataset.GetImages(), test_dataset.GetLabels());
  std::cout << "Model Accuracy: " << accuracy;

  return 0;
}
