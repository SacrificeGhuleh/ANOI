//
// Created by zvone on 23-Feb-20.
//
#include "pch.h"
#include "object.h"
#include "classifier.h"
#include "colors.h"

//template<class T_MAT_TYPE>
void Classifier::checkPoint(const cv::Point &point, cv::Mat_<uint8_t> &filterImg, int16_t currentPoint) {
  //Check bounds
  if (point.x < 0 || point.y < 0 || point.x >= filterImg.cols || point.y >= filterImg.rows)
    return;
  
  //
//  T_MAT_TYPE locPix = filterImg.template at<T_MAT_TYPE>(point.x, point.y);
  uint8_t locPix = filterImg.at<uint8_t>(point);
  
  // Already segmented data
  if (locPix != toSegmentData) return;
  
  //If called from main
  if (currentPoint < 0) {
    
    currentPoint = ++foundObjects;
    objects.emplace_back(new Object(currentPoint, &filterImg));
  }
  objects.at(currentPoint - 1)->indexes_.emplace_back(point);
  //Add pixel to group
  filterImg.at<uint8_t>(point) = currentPoint;
  
  // Get neighbouring 8 pixels (4 are not enough)
  cv::Point point1(point.x + 1, point.y);
  cv::Point point2(point.x - 1, point.y);
  cv::Point point3(point.x, point.y + 1);
  cv::Point point4(point.x, point.y - 1);
  cv::Point point5(point.x + 1, point.y + 1);
  cv::Point point6(point.x - 1, point.y + 1);
  cv::Point point7(point.x + 1, point.y - 1);
  cv::Point point8(point.x - 1, point.y - 1);
  
  //search only in indexes, no need to search in whole image
  for (const cv::Point &newPoint : toSegmentIndexes) {
    if (newPoint == point1 ||
        newPoint == point2 ||
        newPoint == point3 ||
        newPoint == point4 ||
        newPoint == point5 ||
        newPoint == point6 ||
        newPoint == point7 ||
        newPoint == point8) {
      // Recursively, if neighbour is in index array, check it
      checkPoint(newPoint, filterImg, currentPoint);
    }
  }
}

void Classifier::classify() {
  for (const cv::Point &point : toSegmentIndexes) {
    checkPoint(point, filterImg);
  }
}

void Classifier::preprocess() {
  for (int row = 0; row < filterImg.rows; row++) {
    for (int col = 0; col < filterImg.cols; col++) {
      uint8_t loc_pix = filterImg.at<uint8_t>(row, col);
      
      // Background for segmented image
      filterColorImg.at<cv::Vec3b>(row, col) = cv::Vec3b(noData, noData, noData);
      classificationColorImg.at<cv::Vec3b>(row, col) = cv::Vec3b(255, 255, 255);
      
      if (loc_pix < threshold) {
        loc_pix = noData;
      } else {
        toSegmentIndexes.emplace_back(col, row);
        loc_pix = toSegmentData;
      }
      filterImg.at<uint8_t>(row, col) = loc_pix;
    }
  }
}

void Classifier::show() {
  cv::imshow("TrainImage", srcImg);
  cv::imshow("FilterColorImage", filterColorImg);
  cv::imshow("ClassificationColorImage", classificationColorImg);
  
  cv::waitKey();
}

Classifier::~Classifier() {
  for (Object *obj : objects) {
    delete obj;
    obj = nullptr;
  }
}

void Classifier::print() {
  printf("Found segments: %d\n", foundObjects);
  for (Object *obj : objects) {
    printf("%s\n", obj->toString().c_str());
  }
}

void Classifier::update() {
  for (Object *obj : objects) {
    obj->recomputeTraits();
  }
}

void Classifier::prepareVisualization() {
  for (Object *obj : objects) {
    for (const cv::Point &point : obj->indexes_) {
      uint8_t locPix = filterImg.at<uint8_t>(point);
      cv::Vec3b locPix3;
      if (locPix == noData || locPix == toSegmentData) {
        locPix3 = cv::Vec3b(locPix, locPix, locPix);
      } else {
        locPix3 = colors[(locPix - 1) % colorsSize];
      }
      filterColorImg.at<cv::Vec3b>(point) = locPix3;
      for (const cv::Point &perimeterPoint : obj->perimeterPoints_) {
        
        classificationColorImg.at<cv::Vec3b>(perimeterPoint) = obj->color_;
      }
      
      classificationColorImg.at<cv::Vec3b>(obj->getCenterOfMass()) = obj->color_;
      
      cv::putText(classificationColorImg,
                  std::to_string(obj->index_),
                  cv::Point(obj->getCenterOfMass().x + 10, obj->getCenterOfMass().y - 10), // Coordinates
                  cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                  0.5, // Scale. 2.0 = 2x bigger
                  cv::Scalar(0, 0, 0), // BGR Color
                  1); // Line Thickness (Optional)
      
      cv::putText(classificationColorImg,
                  std::to_string(obj->getArea()),
                  cv::Point(obj->getCenterOfMass().x + 10, obj->getCenterOfMass().y), // Coordinates
                  cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                  0.5, // Scale. 2.0 = 2x bigger
                  cv::Scalar(0, 0, 0), // BGR Color
                  1); // Line Thickness (Optional)
      
      cv::putText(classificationColorImg,
                  std::to_string(obj->getPerimeter()),
                  cv::Point(obj->getCenterOfMass().x + 10, obj->getCenterOfMass().y + 10), // Coordinates
                  cv::FONT_HERSHEY_COMPLEX_SMALL, // Font
                  0.5, // Scale. 2.0 = 2x bigger
                  cv::Scalar(0, 0, 0), // BGR Color
                  1); // Line Thickness (Optional)
      
    }
  }
}

Classifier::Classifier(std::string &inputImagePath) {
  srcImg = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
  if (srcImg.empty()) {
    throw std::invalid_argument("Image not loaded correctly");
  }
  filterImg = srcImg.clone();
  filterColorImg = cv::Mat_<cv::Vec3b>(filterImg.rows, filterImg.cols);
  classificationColorImg = cv::Mat_<cv::Vec3b>(filterImg.rows, filterImg.cols);
}
