#include <core/model.h>

#include <fstream>
#include <iostream>
#include <string>
#include <tuple>

using naivebayes::core::Dataset;
using naivebayes::core::Model;

void LoadDataIntoDataset(Dataset& dataset, const std::string& file_path);
void SaveModel(Model& model, const std::string& output_file_path);

int main() {
  // define values to be used in constructing Datasets and Models
  const std::string train_data_path = "../data/trainingimagesandlabels.txt";
  const std::string test_data_path = "../data/testimagesandlabels.txt";
  const std::string model_save_file = "....";
  const int img_height = 28;
  const int img_width = 28;
  const std::vector<int> class_labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const auto laplace_constant = 0.0f;  // laplace smoothing constant for Model

  // load data from file
  std::cout << "Loading Data..." << std::endl;
  Dataset train_dataset = Dataset(img_height, img_width);
  Dataset test_dataset = Dataset(img_height, img_width);
  LoadDataIntoDataset(train_dataset, train_data_path);
  LoadDataIntoDataset(test_dataset, test_data_path);

  // construct model, train, and save features to file
  Model model = Model(img_width, img_height, class_labels, laplace_constant);

  std::cout << "Training Model..." << std::endl;
  model.Train(train_dataset.GetImages(), train_dataset.GetLabels());

  std::cout << "Saving Model..." << std::endl;
  SaveModel(model, model_save_file);

  // compute & display accuracies
  std::cout << "Computing Accuracies..." << std::endl;
  float test_accuracy = model.ComputeAccuracy(test_dataset.GetImages(),
                                              test_dataset.GetLabels());
  float train_accuracy = model.ComputeAccuracy(train_dataset.GetImages(),
                                               train_dataset.GetLabels());

  std::cout << std::endl;
  std::cout << "Testing Accuracy: " << test_accuracy << std::endl;
  std::cout << "Training Accuracy: " << train_accuracy << std::endl;

  return 0;
}

void LoadDataIntoDataset(Dataset& dataset, const std::string& file_path) {
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
