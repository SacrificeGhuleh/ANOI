//
// Created by zvone on 27-Feb-20.
//

#include "pch.h"
#include "objectclass.h"
#include "object.h"

int ObjectClass::counter = 0;

ObjectClass::ObjectClass(const std::string &className, const cv::Vec3b &color) : className_(className), color_(color) {
  id = counter++;
}

ObjectClass::ObjectClass() {
  id = counter;
  char name[20] = {0};
  sprintf(name, "Obj class %d", counter++);
  
  className_ = name;
  color_ = cv::Vec3b(rand() % 255, rand() % 255, rand() % 255);
}

void ObjectClass::addObject(Object *object) {
  trainingObjects_.emplace_back(object);
  
  object->setObjectClass(this);
  
  //recompute etalon
  
  etalon_ = cv::Vec2d(0, 0);
  for (const Object *obj : trainingObjects_) {
    etalon_[0] += obj->getFeatureOne();
    etalon_[1] += obj->getFeatureTwo();
  }
  etalon_[0] /= trainingObjects_.size();
  etalon_[1] /= trainingObjects_.size();
}

void ObjectClass::printEtalonVector() {
  std::cout << className_ << "  [" << etalon_[0] << "," << etalon_[1] << "]\n";
}

const cv::Vec2d &ObjectClass::getEtalon() const {
  return etalon_;
}

const cv::Vec3b &ObjectClass::getColor() const {
  return color_;
}

void ObjectClass::setColor(const cv::Vec3b &color) {
  color_ = color;
}

int ObjectClass::getId() {
  return id;
}
