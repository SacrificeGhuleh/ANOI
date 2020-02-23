//
// Created by zvone on 23-Feb-20.
//

#ifndef DZO2019_CLASSIFIER_H
#define DZO2019_CLASSIFIER_H


#include <cstdint>
#include <vector>
#include <opencv2/core/types.hpp>

class Object;

class Classifier {
public:
  Classifier(std::string &inputImagePath);
  
  //template<class T_MAT_TYPE>
  void checkPoint(const cv::Point &point, cv::Mat_<uint8_t> &filterImg, int16_t currentPoint = -1);
  
  void preprocess();
  
  void classify();
  
  void show();
  
  void print();
  
  void update();
  
  void prepareVisualization();
  
  ~Classifier();

private:
  constexpr static uint8_t threshold = 128;
  constexpr static uint8_t noData = 0;
  constexpr static uint8_t toSegmentData = 255;
  
  cv::Mat_<uint8_t> srcImg;
  cv::Mat_<uint8_t> filterImg;
  cv::Mat_<cv::Vec3b> filterColorImg;
  cv::Mat_<cv::Vec3b> classificationColorImg;
  
  uint16_t foundObjects = 0;
  std::vector<cv::Point> toSegmentIndexes;
  
  std::vector<Object *> objects;
};


#endif //DZO2019_CLASSIFIER_H
