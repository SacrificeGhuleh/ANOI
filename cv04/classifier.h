//
// Created by zvone on 23-Feb-20.
//

#ifndef DZO2019_CLASSIFIER_H
#define DZO2019_CLASSIFIER_H


#include <cstdint>
#include <vector>
#include <opencv2/core/types.hpp>
#include <opencv2/core/mat.hpp>

class ObjectClass;

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
  
  void assignObjects(const std::vector<ObjectClass *> &objClasses);
  
  void autoAssign(const unsigned int numberOfCategories, std::vector<ObjectClass *> &objectClasses);
  
  ~Classifier();

private:
  constexpr static uint8_t threshold = 128;
  constexpr static uint8_t noData = 0;
  constexpr static uint8_t toSegmentData = 255;
  
  cv::Mat_<uint8_t> srcImg;
  cv::Mat_<uint8_t> filterImg;
  cv::Mat_<cv::Vec3b> filterColorImg;
  cv::Mat_<cv::Vec3b> classificationColorImg;
  cv::Mat_<cv::Vec3b> groupsImg;
  
  uint16_t foundObjects = 0;
  std::vector<cv::Point> toSegmentIndexes;
  
  std::vector<Object *> objects;
public:
  const std::vector<Object *> &getObjects() const;
};


#endif //DZO2019_CLASSIFIER_H
