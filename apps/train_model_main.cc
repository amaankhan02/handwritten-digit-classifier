#include <core/model.h>
#include <utility/train_utility.h>

using naivebayes::core::Dataset;
using naivebayes::core::Model;
using naivebayes::utility::LoadData;
using naivebayes::utility::PrintLine;
using naivebayes::utility::SaveModel;

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

  PrintLine("Training & Saving Model...");
  model.Train(train_dataset.GetImages(), train_dataset.GetLabels());
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