#include <fstream>
#include <iostream>
#include <string>
#include <tuple>

//#include <core/dataset.h>
#include <core/model.h>

using naivebayes::core::Dataset;
using naivebayes::core::Model;

int main() {
  // load dataset from file
  Dataset dataset = Dataset(28, 28);
  std::string input_path;
  std::ifstream input_file(input_path);
  if (input_file.is_open()) {
    input_file >> dataset;
  }

  // construct model and train
  Model model = Model(28, 28, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 1);
  model.Train(dataset.GetImages(), dataset.GetLabels());

  // save model features to file
  std::string output_path;
  std::ofstream output_file(output_path);
  if (output_file.is_open()) {
    output_file << model;
  }

  return 0;
}
