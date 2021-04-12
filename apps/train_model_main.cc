#include <core/model.h>

#include <fstream>
#include <iostream>
#include <string>
#include <tuple>

using naivebayes::core::Dataset;
using naivebayes::core::Model;

// TODO: should these be in a separate file?
void LoadData(Dataset& dataset, const std::string& file_path);
void SaveModel(Model& model, const std::string& output_file_path);
void PrintLine(const std::string& msg);
void PrintLine(const std::string& msg, float value);

// TODO: is it fine to define these values outside the method
const auto train_data_path = "../data/trainingimagesandlabels.txt";
const auto test_data_path = "../data/testimagesandlabels.txt";
const auto model_save_file = "../data/modelsave.txt";
const auto class_labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
const auto img_height = 28;
const auto img_width = 28;
const auto laplace_constant = 1e-4f;  // laplace smoothing constant for Model

int main() {
  Dataset train_dataset = Dataset(img_height, img_width);
  Dataset test_dataset = Dataset(img_height, img_width);
  Model model = Model(img_width, img_height, class_labels, laplace_constant);

  PrintLine("Loading Data...");
  LoadData(train_dataset, train_data_path);
  LoadData(test_dataset, test_data_path);

  PrintLine("Training Model...");
  model.Train(train_dataset.GetImages(), train_dataset.GetLabels());

  PrintLine("Saving Model...");
  SaveModel(model, model_save_file);

  PrintLine("Computing Accuracies...");
  float test_accuracy = model.ComputeAccuracy(test_dataset.GetImages(),
                                              test_dataset.GetLabels());
  float train_accuracy = model.ComputeAccuracy(train_dataset.GetImages(),
                                               train_dataset.GetLabels());

  PrintLine("***********************");
  PrintLine("Testing Accuracy:  ", test_accuracy);
  PrintLine("Training Accuracy: ", train_accuracy);

  return 0;
}

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
