#pragma once

#include "cinder/app/App.h"
#include "cinder/app/RendererGl.h"
#include "cinder/gl/gl.h"
#include "sketchpad.h"
#include <core/model.h>

namespace naivebayes {

namespace visualizer {

/**
 * Allows a user to draw a digit on a sketchpad and uses Naive Bayes to
 * classify it.
 */
class NaiveBayesApp : public ci::app::App {
 public:
  NaiveBayesApp();

  void draw() override;
  void mouseDown(ci::app::MouseEvent event) override;
  void mouseDrag(ci::app::MouseEvent event) override;
  void keyDown(ci::app::KeyEvent event) override;

  // TODO: Delete this comment. Feel free to play around with these variables
  // provided that you can see the entire UI on your screen.
  const double kWindowSize = 1600;
  const double kMargin = 100;
  const size_t kImageDimension = 28;
  const std::vector<int> kClassLabels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const float kLaplaceSmoothingConstant = 1e-4f;
  const std::string kModelFeaturesFile = R"(C:\Users\amaan\CppLibraries\Cinder\my-projects\naive-bayes\data\modelsave.txt)";

 private:
  Sketchpad sketchpad_;
  naivebayes::core::Model model_; // Naive bayes model to classify digits
  int current_prediction_ = -1;

  void LoadModelFeatures();
};

}  // namespace visualizer

}  // namespace naivebayes
