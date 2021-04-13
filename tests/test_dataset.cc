#include <core/dataset.h>

#include <catch2/catch.hpp>
#include <fstream>
#include <iostream>
#include <string>

using naivebayes::core::Dataset;
using naivebayes::core::Image;
using naivebayes::core::Pixel;
using std::vector;

TEST_CASE("Test >> Operator with 3 images") {
  std::ifstream input_file(R"(test_data/test_dataset_data.txt)");
  Dataset dataset = Dataset(3, 3);
  if (input_file.is_open()) {
    input_file >> dataset;
    input_file.close();
  }
  Image img1 = dataset.GetImage(0);
  Image img2 = dataset.GetImage(1);
  Image img3 = dataset.GetImage(2);

  SECTION("Check pixel values are correct (parsed correctly)") {
    vector<vector<Pixel>> expected_img12 = {
        {Pixel::kWhite, Pixel::kWhite, Pixel::kWhite},
        {Pixel::kBlack, Pixel::kBlack, Pixel::kBlack},
        {Pixel::kBlack, Pixel::kBlack, Pixel::kBlack}};
    vector<vector<Pixel>> expected_img3 = {
        {Pixel::kBlack, Pixel::kBlack, Pixel::kBlack},
        {Pixel::kWhite, Pixel::kWhite, Pixel::kBlack},
        {Pixel::kBlack, Pixel::kBlack, Pixel::kBlack}};

    REQUIRE(img1.GetImageAsVector() == expected_img12);
    REQUIRE(img2.GetImageAsVector() == expected_img12);
    REQUIRE(img3.GetImageAsVector() == expected_img3);
  }

  SECTION("Check image size is correct") {
    REQUIRE(img1.GetHeight() == 3);
    REQUIRE(img1.GetWidth() == 3);

    REQUIRE(img2.GetHeight() == 3);
    REQUIRE(img2.GetWidth() == 3);

    REQUIRE(img3.GetHeight() == 3);
    REQUIRE(img3.GetWidth() == 3);
  }

  SECTION("Check labels are assigned correctly") {
    REQUIRE(dataset.GetLabel(0) == 1);
    REQUIRE(dataset.GetLabel(1) == 1);
    REQUIRE(dataset.GetLabel(2) == 2);
  }
}

TEST_CASE("Test >> Operator with images invalid size/inputs") {
  SECTION("Check it throws error for num COLS don't match the input width") {
    std::ifstream input_file(
        R"(test_data/test_dataset_invalidcolssizedata.txt)");
    Dataset dataset = Dataset(3, 3);
    if (input_file.is_open()) {
      REQUIRE_THROWS_AS(input_file >> dataset, std::invalid_argument);
    }
  }

  SECTION("Check it throws error for num ROWS don't match the input height") {
    std::ifstream input_file(
        R"(test_data/test_dataset_invalidrowssizedata.txt)");
    Dataset dataset = Dataset(3, 3);
    if (input_file.is_open()) {
      REQUIRE_THROWS_AS(input_file >> dataset, std::invalid_argument);
    }
  }

  SECTION("Check it throws error for invalid CHARS in the image") {
    std::ifstream input_file(R"(test_data/test_dataset_data_invalidchars.txt)");
    Dataset dataset = Dataset(3, 3);
    if (input_file.is_open()) {
      REQUIRE_THROWS_AS(input_file >> dataset, std::invalid_argument);
    }
  }
}

TEST_CASE(
    "Test >> Operator with 5x5 image size and different image count in file") {
  std::ifstream input_file(R"(test_data/test_dataset_data_5x5.txt)");
  Dataset dataset = Dataset(5, 5);
  if (input_file.is_open()) {
    input_file >> dataset;
    input_file.close();
  }
  Image img1 = dataset.GetImage(0);

  SECTION("Check pixel values are correct (parsed correctly)") {
    vector<vector<Pixel>> expected_img1 = {
        {Pixel::kWhite, Pixel::kWhite, Pixel::kWhite, Pixel::kWhite,
         Pixel::kWhite},
        {Pixel::kBlack, Pixel::kBlack, Pixel::kBlack, Pixel::kWhite,
         Pixel::kWhite},
        {Pixel::kBlack, Pixel::kBlack, Pixel::kBlack, Pixel::kBlack,
         Pixel::kBlack},
        {Pixel::kBlack, Pixel::kBlack, Pixel::kBlack, Pixel::kWhite,
         Pixel::kWhite},
        {Pixel::kBlack, Pixel::kBlack, Pixel::kBlack, Pixel::kWhite,
         Pixel::kWhite}};

    REQUIRE(img1.GetImageAsVector() == expected_img1);
  }

  SECTION("Check image size is correct") {
    REQUIRE(img1.GetHeight() == 5);
    REQUIRE(img1.GetWidth() == 5);
  }

  SECTION("Check labels are assigned correctly") {
    REQUIRE(dataset.GetLabel(0) == 1);
  }
}