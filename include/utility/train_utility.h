//
// Created by amaan on 4/13/2021.
//
#include <core/dataset.h>
#include <core/model.h>

using naivebayes::core::Dataset;
using naivebayes::core::Model;

#ifndef NAIVE_BAYES_TRAIN_UTILITY_H
#define NAIVE_BAYES_TRAIN_UTILITY_H
namespace naivebayes {
namespace utility { // utility functions for train_model_main.cc

/**
 * Load data from a file to the passed in Dataset object
 * @param dataset       dataset to load data into
 * @param file_path     path to the file that has the data
 */
void LoadData(Dataset& dataset, const std::string& file_path);

/**
 * Saves model to the file path that is passed in
 * @param model             model to save to file
 * @param output_file_path  path to file to save data into
 */
void SaveModel(Model& model, const std::string& output_file_path);

/**
 * Prints a line of message onto console
 * @param msg   message to print to console
 */
void PrintLine(const std::string& msg);

/**
 * Prints a line of message onto console with a float value to be appended
 * to the end of the message string (does not add space between msg and value)
 * @param msg   message to print
 * @param value value to be appended to the message to be printed
 */
void PrintLine(const std::string& msg, float value);

}  // namespace utility
}  // namespace naivebayes
#endif  // NAIVE_BAYES_TRAIN_UTILITY_H
