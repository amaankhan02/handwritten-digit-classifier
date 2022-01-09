#include <core/dataset.h>
#include <core/model.h>

#include <catch2/catch.hpp>
#include <fstream>
#include <iostream>
#include <string>

using naivebayes::core::Dataset;
using naivebayes::core::Model;
using naivebayes::core::Image;
using naivebayes::core::Pixel;
using std::vector;

TEST_CASE("Test train") {
  Model model = Model(2, 2, {1, 2}, 1);
  vector<vector<Pixel>> img1 = {
      {Pixel::kWhite, Pixel::kWhite},
      {Pixel::kWhite, Pixel::kBlack}
  };  // make sure gray is parsed as black
  vector<vector<Pixel>> img2 = {
      {Pixel::kBlack, Pixel::kWhite},
      {Pixel::kWhite, Pixel::kWhite}
  };
  vector<vector<Pixel>> img3 = {
      {Pixel::kBlack, Pixel::kBlack},
      {Pixel::kWhite, Pixel::kWhite}
  };
  model.Train({img1, img2, img3}, {1, 1, 2});

  SECTION("Check prior probabilities are stored correctly") {
    auto prior_probs = model.GetPriorProbabilities();
    REQUIRE(prior_probs[0] == Approx(0.60f).epsilon(0.01));
    REQUIRE(prior_probs[1] == Approx(0.40f).epsilon(0.01));
  }

  SECTION("Check feature probabilities are correct") {
    auto feature_probs = model.GetFeatureProbabilities();
    REQUIRE(feature_probs[0][0][0][0] == Approx(0.50f).epsilon(0.01));
    REQUIRE(feature_probs[0][0][0][1] == Approx(0.50f).epsilon(0.01));
    REQUIRE(feature_probs[0][0][1][0] == Approx(0.75f).epsilon(0.01));
    REQUIRE(feature_probs[0][0][1][1] == Approx(0.25f).epsilon(0.01));
    REQUIRE(feature_probs[0][1][0][0] == Approx(0.75f).epsilon(0.01));
    REQUIRE(feature_probs[0][1][0][1] == Approx(0.25f).epsilon(0.01));
    REQUIRE(feature_probs[0][1][1][0] == Approx(0.50f).epsilon(0.01));
    REQUIRE(feature_probs[0][1][1][1] == Approx(0.50f).epsilon(0.01));

    REQUIRE(feature_probs[1][0][0][0] == Approx(0.33f).epsilon(0.01));
    REQUIRE(feature_probs[1][0][0][1] == Approx(0.67f).epsilon(0.01));
    REQUIRE(feature_probs[1][0][1][0] == Approx(0.33f).epsilon(0.01));
    REQUIRE(feature_probs[1][0][1][1] == Approx(0.67f).epsilon(0.01));
    REQUIRE(feature_probs[1][1][0][0] == Approx(0.67f).epsilon(0.01));
    REQUIRE(feature_probs[1][1][0][1] == Approx(0.33f).epsilon(0.01));
    REQUIRE(feature_probs[1][1][1][0] == Approx(0.67f).epsilon(0.01));
    REQUIRE(feature_probs[1][1][1][1] == Approx(0.33f).epsilon(0.01));
  }
}

TEST_CASE("Test saving - << operator") {
  std::ofstream output_file("/model_saves_test/model_probs.txt");
  Model model = Model(3, 3, {1, 2}, 1);
  vector<vector<Pixel>> img1 = {
      {Pixel::kWhite, Pixel::kWhite, Pixel::kWhite},
      {Pixel::kBlack, Pixel::kBlack,
                                     Pixel::kBlack},  // make sure gray is parsed as black
      {Pixel::kBlack, Pixel::kBlack, Pixel::kBlack}};
  vector<vector<Pixel>> img2 = {
      {Pixel::kBlack, Pixel::kBlack, Pixel::kBlack},
      {Pixel::kWhite, Pixel::kWhite, Pixel::kBlack},
      {Pixel::kBlack, Pixel::kBlack, Pixel::kBlack}};
  vector<vector<Pixel>> img3 = {
      {Pixel::kBlack, Pixel::kBlack, Pixel::kBlack},
      {Pixel::kWhite, Pixel::kWhite, Pixel::kBlack},
      {Pixel::kBlack, Pixel::kBlack, Pixel::kBlack}};
  vector<int> labels = {1, 1, 2};
  vector<Image> imgs = {Image(img1), Image(img2), Image(img3)};

  model.Train(imgs, labels);

  if (output_file.is_open()) {
    output_file << model;
    output_file.close();
  }

}

TEST_CASE("Test model loading from file - >> operator") {

}