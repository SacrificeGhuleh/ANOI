//
// Created by zvone on 23-Feb-20.
//

#ifndef DZO2019_OBJECT_H
#define DZO2019_OBJECT_H


#include <opencv2/core/types.hpp>

class Object {
public:
  explicit Object(unsigned int index, cv::Mat_<uint8_t> *filterImg);
  
  unsigned int getArea() const;
  
  cv::Point computeCenterOfMass();
  
  cv::Point getCenterOfMass() const;
  
  unsigned int computeCircumference();
  unsigned int getCircumference() const;
  
  std::string toString() const;
  
  void computePerimeter();
  
  unsigned int getPerimeter() const;
  
  void recomputeTraits();
  
  double computeMoment(int p, int q) const;
  double computeMomentToCenter(int p, int q) const;
  
  double computeFeatureOne();
  double computeFeatureTwo();
  
  double getFeatureOne() const;
  double getFeatureTwo() const;
  
  double featureOne_;
  double featureTwo_;
  unsigned int perimeter_;
//private:
  unsigned int index_;
  std::vector<cv::Point> indexes_;
  cv::Point centerOfMass_;
  std::vector<cv::Point> perimeterPoints_;
  cv::Mat_<uint8_t> *filterImg_;
  cv::Vec3b color_;
  unsigned int circumference_;
};


#endif //DZO2019_OBJECT_H
