//
// Created by zvone on 27-Feb-20.
//

#ifndef DZO2019_OBJECTCLASS_H
#define DZO2019_OBJECTCLASS_H

#include <vector>
#include <string>
#include <opencv2/core/matx.hpp>

class Object;

class ObjectClass {
public:
  ObjectClass();
  
  ObjectClass(const std::string &className, const cv::Vec3b& color);
  
  void addObject(Object *object);
  
  void printEtalonVector();
  
  const cv::Vec2d &getEtalon() const;
  
  const cv::Vec3b &getColor() const;
  
  void setColor(const cv::Vec3b &color);

  int getId();
private:
  cv::Vec3b color_;
  
  cv::Vec2d etalon_;
  std::vector<Object *> trainingObjects_;
  std::string className_;
  int id;
  static int counter;
};


#endif //DZO2019_OBJECTCLASS_H
