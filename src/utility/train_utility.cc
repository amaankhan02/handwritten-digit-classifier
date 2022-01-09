//
// Created by amaan on 4/13/2021.
//
#include <utility/train_utility.h>

#include <fstream>
#include <iostream>
#include <string>
#include <tuple>

namespace naivebayes {
namespace utility {

void LoadData(Dataset& dataset, const std::string& file_path) {
  std::ifstream input_file(file_path);
  if (input_file.is_open()) {
    input_file >> dataset;
  }
}

void SaveModel(Model& model, const std::string& output_file_path) {
  std::ofstream output_file(output_file_path);
  if (output_file.is_open()) {
    output_file << model;
  }
}

void PrintLine(const std::string& msg) {
  std::cout << msg << std::endl;
}

void PrintLine(const std::string& msg, float value) {
  std::cout << msg << value << std::endl;
}
}  // namespace utility
}  // namespace naivebayes