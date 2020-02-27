//
// Created by zvone on 23-Feb-20.
//

#ifndef DZO2019_OBJECT_H
#define DZO2019_OBJECT_H


#include <opencv2/core/types.hpp>

class ObjectClass;

class Object {
public:
  explicit Object(unsigned int index, cv::Mat_<uint8_t> *filterImg);
  
  unsigned int getArea() const;
  
  cv::Point computeCenterOfMass();
  
  cv::Point getCenterOfMass() const;
  
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
  
  void addPoint(const cv::Point &point);
  
  const cv::Vec3b &getColor() const;
  
  unsigned int getIndex() const;
  
  const cv::Point &getEthalon() const;
  
  const cv::Point &computeEthalon();
  
  const std::vector<cv::Point> &getIndexes() const;
  
  const std::vector<cv::Point> &getPerimeterPoints() const;
  
  cv::Mat_<uint8_t> *getFilterImg() const;
  
  ObjectClass *getObjectClass() const;
  
  void setObjectClass(ObjectClass *objectClass);

private:
  ObjectClass *objectClass_;
  double featureOne_;
  double featureTwo_;
  unsigned int perimeter_;
  unsigned int index_;
  std::vector<cv::Point> indexes_;
  cv::Point centerOfMass_;
  cv::Point ethalon_;
  std::vector<cv::Point> perimeterPoints_;
  cv::Mat_<uint8_t> *filterImg_;
  cv::Vec3b color_;
};


#endif //DZO2019_OBJECT_H
