#include <core/dataset.h>
#include <core/model.h>

#include <catch2/catch.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using naivebayes::core::Dataset;
using naivebayes::core::Image;
using naivebayes::core::Model;
using naivebayes::core::Pixel;
using std::string;
using std::vector;

TEST_CASE("Test train") {
  Model model = Model(2, 2, {0, 1}, 1);
  vector<vector<Pixel>> img1 = {{Pixel::kWhite, Pixel::kWhite},
                                {Pixel::kWhite, Pixel::kBlack}};
  vector<vector<Pixel>> img2 = {{Pixel::kBlack, Pixel::kWhite},
                                {Pixel::kWhite, Pixel::kWhite}};
  vector<vector<Pixel>> img3 = {{Pixel::kBlack, Pixel::kBlack},
                                {Pixel::kWhite, Pixel::kWhite}};
  model.Train({img1, img2, img3}, {1, 1, 0});

  SECTION("Check prior probabilities are stored correctly") {
    auto prior_probs = model.GetPriorProbabilities();
    REQUIRE(prior_probs[0] == Approx(0.40f).epsilon(0.01));
    REQUIRE(prior_probs[1] == Approx(0.60f).epsilon(0.01));
  }

  SECTION("Check feature probabilities are correct") {
    auto feature_probs = model.GetFeatureProbabilities();
    REQUIRE(feature_probs[0][0][0][0] == Approx(0.333f).epsilon(0.01));
    REQUIRE(feature_probs[0][0][0][1] == Approx(0.667f).epsilon(0.01));
    REQUIRE(feature_probs[0][0][1][0] == Approx(0.333f).epsilon(0.01));
    REQUIRE(feature_probs[0][0][1][1] == Approx(0.667f).epsilon(0.01));
    REQUIRE(feature_probs[0][1][0][0] == Approx(0.667f).epsilon(0.01));
    REQUIRE(feature_probs[0][1][0][1] == Approx(0.333f).epsilon(0.01));
    REQUIRE(feature_probs[0][1][1][0] == Approx(0.667f).epsilon(0.01));
    REQUIRE(feature_probs[0][1][1][1] == Approx(0.333f).epsilon(0.01));

    REQUIRE(feature_probs[1][0][0][0] == Approx(0.50f).epsilon(0.01));
    REQUIRE(feature_probs[1][0][0][1] == Approx(0.50f).epsilon(0.01));
    REQUIRE(feature_probs[1][0][1][0] == Approx(0.75f).epsilon(0.01));
    REQUIRE(feature_probs[1][0][1][1] == Approx(0.25f).epsilon(0.01));
    REQUIRE(feature_probs[1][1][0][0] == Approx(0.75f).epsilon(0.01));
    REQUIRE(feature_probs[1][1][0][1] == Approx(0.25f).epsilon(0.01));
    REQUIRE(feature_probs[1][1][1][0] == Approx(0.50f).epsilon(0.01));
    REQUIRE(feature_probs[1][1][1][1] == Approx(0.50f).epsilon(0.01));
  }
}

TEST_CASE("Test saving: << operator") {
  vector<vector<Pixel>> img1 = {{Pixel::kWhite, Pixel::kWhite},
                                {Pixel::kWhite, Pixel::kBlack}};
  vector<vector<Pixel>> img2 = {{Pixel::kBlack, Pixel::kWhite},
                                {Pixel::kWhite, Pixel::kWhite}};
  vector<vector<Pixel>> img3 = {{Pixel::kBlack, Pixel::kBlack},
                                {Pixel::kWhite, Pixel::kWhite}};

  SECTION("Check that outputted values are correct - 3 label types") {
    Model model = Model(2, 2, {0, 1, 2}, 1);
    model.Train({img1, img2, img3}, {1, 0, 2});
    string save_path =
        R"(C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\tests\model_saves_test\testmodelsave1.txt)";
    std::ofstream output_file(save_path);
    if (output_file.is_open()) {
      output_file << model;  // save model to file
    }
    string expected_str =
        "0.333333\n"
        "0.333333\n"
        "0.333333\n"
        "*\n"
        "0.333333,0.666667|0.666667,0.333333|\n"
        "0.666667,0.333333|0.666667,0.333333|\n"
        ".\n"
        "0.666667,0.333333|0.666667,0.333333|\n"
        "0.666667,0.333333|0.333333,0.666667|\n"
        ".\n"
        "0.333333,0.666667|0.333333,0.666667|\n"
        "0.666667,0.333333|0.666667,0.333333|\n"
        ".";

    // open saved file and evaluate values are correct line by line
    std::istringstream expected_stream(
        expected_str);  // convert expected_str to a stream to read it line by
                        // line
    std::ifstream actual_stream(save_path);
    string expected_line;
    string actual_line;

    while (std::getline(actual_stream, actual_line)) {
      std::getline(expected_stream, expected_line);
      REQUIRE(actual_line == expected_line);
    }
  }

  SECTION("Check that outputted values are correct - 0 label types") {
    Model model = Model(2, 2, {}, 1);
    model.Train({}, {});
    string save_path =
        R"(C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\tests\model_saves_test\testmodelsave2.txt)";
    std::ofstream output_file(save_path);
    if (output_file.is_open()) {
      output_file << model;  // save model to file
    }
    string expected_str = "*\n";

    // open saved file and evaluate values are correct line by line
    std::istringstream expected_stream(
        expected_str);  // convert expected_str to a stream to read it line by
                        // line
    std::ifstream actual_stream(save_path);
    string expected_line;
    string actual_line;

    while (std::getline(actual_stream, actual_line)) {
      std::getline(expected_stream, expected_line);
      REQUIRE(actual_line == expected_line);
    }
  }
}

TEST_CASE("Test loading: >> operator") {
  SECTION("Load file with ZERO label types. (i.e. no probabilities)") {
    Model model = Model(2, 2, {}, 1);
    std::string path =
        R"(C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\tests\model_loads_test\testmodelload2.txt)";
    std::ifstream input_file(path);

    if (input_file.is_open()) {
      input_file >> model;
    }

    REQUIRE(model.GetFeatureProbabilities().empty());
    REQUIRE(model.GetPriorProbabilities().empty());
  }

  SECTION("Load file with THREE label types saved") {
    Model model = Model(2, 2, {0, 1, 2}, 1);
    std::string path =
        R"(C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\tests\model_loads_test\testmodelload1.txt)";
    std::ifstream input_file(path);
    if (input_file.is_open()) {
      input_file >> model;  // load file into model
    }

    vector<float> expected_prior_probs = {0.333333f, 0.333333f, 0.333333f};
    vector<vector<vector<vector<float>>>> expected_feature_probs = {
        {{{.333333f, .666667f}, {0.666667f, .333333f}},
         {{.666667f, .333333f}, {.666667f, .333333f}}},
        {{{.666667f, .333333f}, {.666667f, .333333f}},
         {{.666667f, .333333f}, {.333333f, .666667f}}},
        {{{.333333f, .666667f}, {.333333f, .666667f}},
         {{.666667f, .333333f}, {.666667f, .333333f}}}};

    REQUIRE(model.GetPriorProbabilities() == expected_prior_probs);
    REQUIRE(model.GetFeatureProbabilities() == expected_feature_probs);
  }
}

TEST_CASE("Test Predict()") {
  Model model = Model(2, 2, {0, 1}, 1);
  vector<vector<Pixel>> train_img1 = {{Pixel::kWhite, Pixel::kWhite},
                                      {Pixel::kWhite, Pixel::kBlack}};
  vector<vector<Pixel>> train_img2 = {{Pixel::kBlack, Pixel::kBlack},
                                      {Pixel::kBlack, Pixel::kBlack}};
  vector<vector<Pixel>> train_img3 = {{Pixel::kBlack, Pixel::kBlack},
                                      {Pixel::kWhite, Pixel::kWhite}};
  model.Train({train_img1, train_img2, train_img3}, {1, 1, 0});

  SECTION("Check predictions are correct for all label types") {
    Image test_img1 = Image(2, 2);
    test_img1.SetPixel(Pixel::kBlack, 0, 1);
    Image test_img2 = Image(2, 2, Pixel::kBlack);

    REQUIRE(0 == model.Predict(test_img1));
    REQUIRE(1 == model.Predict(test_img2));
  }

  SECTION("Predict on Invalid Image Dimensions") {
    // model is trained on 2x2 imgs, so Predict(4x4 img) should throw error
    Image test_img = Image(4, 4);
    REQUIRE_THROWS_AS(model.Predict(test_img), std::invalid_argument);
  }
}

TEST_CASE("Test ComputeAccuracy()") {
  Model model = Model(2, 2, {0, 1}, 1);
  vector<vector<Pixel>> train_img1 = {{Pixel::kWhite, Pixel::kWhite},
                                      {Pixel::kWhite, Pixel::kBlack}};
  vector<vector<Pixel>> train_img2 = {{Pixel::kBlack, Pixel::kBlack},
                                      {Pixel::kBlack, Pixel::kBlack}};
  vector<vector<Pixel>> train_img3 = {{Pixel::kBlack, Pixel::kBlack},
                                      {Pixel::kWhite, Pixel::kWhite}};
  model.Train({train_img1, train_img2, train_img3}, {1, 1, 0});

  SECTION("Check Accuracy with 3 test images") {
    Image test_img1 = Image(2, 2);                 // all white img
    Image test_img2 = Image(2, 2, Pixel::kBlack);  // all black image
    Image test_img3 = Image(2, 2);
    test_img3.SetPixel(Pixel::kBlack, 0, 0);
    test_img3.SetPixel(Pixel::kBlack, 1, 1);
    auto correct_labels = {0, 1, 0};

    auto actual_acc = model.ComputeAccuracy({test_img1, test_img2, test_img3},
                                            correct_labels);
    REQUIRE(actual_acc == Approx(0.667f).epsilon(0.01));
  }

  SECTION("Check accuracy with 0 images (divide by zero - nan return)") {
    REQUIRE_THROWS_AS(model.ComputeAccuracy({}, {}), std::invalid_argument);
  }

  SECTION("Check error thrown for number of images != number of labels") {
    Image test_img1 = Image(2, 2);
    auto correct_labels = {0, 1};  // 2 labels but only 1 image

    REQUIRE_THROWS_AS(model.ComputeAccuracy({test_img1}, correct_labels),
                      std::invalid_argument);
  }
}