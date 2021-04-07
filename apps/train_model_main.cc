#include <iostream>
#include <fstream>
#include <string>
#include <tuple>

//#include <core/dataset.h>
#include <core/model.h>

//using naivebayes::core::Dataset;
using naivebayes::core::Model;

int main() {
  // and saves the trained model to a file

  std::ifstream input_file(R"(C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\data\trainingimagesandlabels.txt)");
  if (input_file.is_open()) {

  }
//  Model model = Model();

//  Dataset training_data = Dataset();
//  std::string line;
//  if (input_file.is_open()) {
//    input_file >> training_data;
//    std::getline(input_file, line);
//    std::cout << line.length() << std::endl;
//    std::cout << line << std::endl;
//    std::getline(input_file, line);
//    std::cout << line.length() << std::endl;
//    std::cout << "line contents " << line << std::endl;
//  }

//  auto ret = training_data[0];
//  std::cout << std::get<1>(ret) << std::endl;
//  std::cout << training_data.size();

  return 0;

}
