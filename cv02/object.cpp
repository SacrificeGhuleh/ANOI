//
// Created by zvone on 23-Feb-20.
//
#include "pch.h"
#include <sstream>
#include "object.h"
#include "colors.h"

Object::Object(unsigned int index, cv::Mat_<uint8_t> *filterImg) : index_(index),
                                                                   perimeter_(0),
                                                                   filterImg_(filterImg),
                                                                   featureOne_(0) {
  color_ = colors[(index_ - 1) % colorsSize];
}

unsigned int Object::getArea() const {
  return indexes_.size();
}

cv::Point Object::computeCenterOfMass() {
  centerOfMass_.x = computeMoment(1, 0) / computeMoment(0, 0);
  centerOfMass_.y = computeMoment(0, 1) / computeMoment(0, 0);
  return centerOfMass_;
}

cv::Point Object::getCenterOfMass() const {
  return centerOfMass_;
}


std::string Object::toString() const {
  std::stringstream stringStream;
  stringStream <<
               "Object : " << index_ << "\n" <<
               "  color: " << (int) color_[0] << " " << (int) color_[1] << " " << (int) color_[2] << "\n" <<
               "  area: " << getArea() << "\n" <<
               "  center of mass [" << centerOfMass_.x << "," << centerOfMass_.y << "]\n" <<
               "  perimeter: " << perimeter_ << "\n" <<
               "  feature one: " << featureOne_ << "\n" <<
               "  feature two: " << featureTwo_ << "\n";
  return stringStream.str();
}

void Object::computePerimeter() {
  perimeter_ = 0;
  perimeterPoints_.clear();
  
  for (const cv::Point &point : indexes_) {
    if (point.x < 1 || point.y < 1 || point.x >= filterImg_->cols || point.y >= filterImg_->rows)
      continue;
    
    const uint8_t locPix = filterImg_->at<uint8_t>(point);
    
    cv::Point point1(point.x + 1, point.y);
    cv::Point point2(point.x - 1, point.y);
    cv::Point point3(point.x, point.y + 1);
    cv::Point point4(point.x, point.y - 1);
    
    const uint8_t locPix1 = filterImg_->at<uint8_t>(point1);
    const uint8_t locPix2 = filterImg_->at<uint8_t>(point2);
    const uint8_t locPix3 = filterImg_->at<uint8_t>(point3);
    const uint8_t locPix4 = filterImg_->at<uint8_t>(point4);
    
    if (locPix != locPix1 ||
        locPix != locPix2 ||
        locPix != locPix3 ||
        locPix != locPix4) {
      perimeterPoints_.emplace_back(point);
      perimeter_++;
    }
  }
}

unsigned int Object::getPerimeter() const {
  return perimeter_;
}

void Object::recomputeTraits() {
  computePerimeter();
  computeCenterOfMass();
  computeFeatureOne();
  computeFeatureTwo();
}


double Object::computeMoment(int p, int q) const {
  double moment = 0.0;
  for (const cv::Point &point : indexes_) {
    moment += pow(point.x, p) * pow(point.y, q);
  }
  return moment;
}


double Object::computeMomentToCenter(int p, int q) const {
  double moment = 0.0;
  for (const cv::Point &point : indexes_) {
    moment += pow(point.x - centerOfMass_.x, p) * pow(point.y - centerOfMass_.y, q);
  }
  return moment;
}

double Object::computeFeatureOne() {
  featureOne_ = (perimeter_ * perimeter_) / (100. * getArea());
  return featureOne_;
}


double Object::computeFeatureTwo() {
  double mi20 = computeMomentToCenter(2, 0);
  double mi02 = computeMomentToCenter(0, 2);
  double mi11 = computeMomentToCenter(1, 1);
  
  double maxCenterMoment = 0.5 * (mi20 + mi02) + 0.5 * sqrt(4 * SQR(mi11) + SQR(mi20 - mi02));
  double minCenterMoment = 0.5 * (mi20 + mi02) - 0.5 * sqrt(4 * SQR(mi11) + SQR(mi20 - mi02));
  featureTwo_ = minCenterMoment / maxCenterMoment;
  return featureTwo_;
}

double Object::getFeatureOne() const {
  return featureOne_;
}

double Object::getFeatureTwo() const {
  return featureTwo_;
}