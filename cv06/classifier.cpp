//
// Created by zvone on 23-Feb-20.
//
#include <random>
#include "pch.h"
#include "object.h"
#include "classifier.h"
#include "colors.h"
#include "objectclass.h"
#include "utils.h"

#define FEATURES_MAT_SIZE 128

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
//  objects.at(currentPoint - 1)->indexes_.emplace_back(point);
  objects.at(currentPoint - 1)->addPoint(point);
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
      groupsImg.at<cv::Vec3b>(row, col) = cv::Vec3b(255, 255, 255);
      
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
  cv::imshow("GroupsImage", groupsImg);
  
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
    for (const cv::Point &point : obj->getIndexes()) {
      uint8_t locPix = filterImg.at<uint8_t>(point);
      cv::Vec3b locPix3;
      if (locPix == noData || locPix == toSegmentData) {
        locPix3 = cv::Vec3b(locPix, locPix, locPix);
      } else {
        locPix3 = colors[(locPix - 1) % colorsSize];
      }
      filterColorImg.at<cv::Vec3b>(point) = locPix3;
      for (const cv::Point &perimeterPoint : obj->getPerimeterPoints()) {
        classificationColorImg.at<cv::Vec3b>(perimeterPoint) = obj->getColor();
      }
      
      classificationColorImg.at<cv::Vec3b>(obj->getCenterOfMass()) = obj->getColor();
      if (obj->getObjectClass() != nullptr) {
        groupsImg.at<cv::Vec3b>(point) = obj->getObjectClass()->getColor();
      } else {
        groupsImg.at<cv::Vec3b>(point) = cv::Vec3b(0, 0, 0);
      }
      
      cv::putText(classificationColorImg,
                  std::to_string(obj->getIndex()),
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
  groupsImg = cv::Mat_<cv::Vec3b>(filterImg.rows, filterImg.cols);
}

const std::vector<Object *> &Classifier::getObjects() const {
  return objects;
}

void Classifier::assignObjects(const std::vector<ObjectClass *> &objClasses) {
  for (Object *object: objects) {
    double distance = 9999999.;
    ObjectClass *objClassToSet;
    for (ObjectClass *objectClass : objClasses) {
      double localDistance = sqrt(SQR(object->getFeatureOne() - objectClass->getEtalon()[0]) +
                                  SQR(object->getFeatureTwo() - objectClass->getEtalon()[1]));
      if (localDistance < distance) {
        distance = localDistance;
        objClassToSet = objectClass;
      }
    }
    objClassToSet->addObject(object);
  }
  
}

struct Cluster {
  Cluster(int clusterId, const cv::Point2d &clusterCentroid) : id(clusterId), centroid(clusterCentroid) {}
  
  int id = 0;
  cv::Point2d centroid = {0, 0};
};

void Classifier::autoAssign(const unsigned int numberOfCategories, std::vector<ObjectClass *> &objectClasses) {
  std::vector<Cluster *> clusters;
  
  for (int i = 0; i < numberOfCategories; i++) {
    clusters.emplace_back(new Cluster(i, cv::Point2d(random(0, 1), random(0, 1))));
  }
  
  int exchangesPerformed = 0;
  const int maxLoops = 100;
  int currentLoop = 0;
  
  std::vector<std::pair<Object *, Cluster *>> data;
  
  #define SIZE 512
  
  cv::Mat_<cv::Vec3b> visualizeKMeansMat(SIZE, SIZE);
  
  for (Object *obj: objects) {
    obj->computeFeaturesVector();
  }
  
  // Initialize centroids to randomly selected points from the input
  for(Cluster* cluster : clusters){
    cluster->centroid = objects.at(random(0, objects.size()))->getFeaturesVector();
  }
  
  // Compute Euclidean distance from each centroid to all input datapoints.
  // Assign each input data to the closest centroid.
  for (Object *obj: objects) {
    data.emplace_back(std::make_pair(obj, clusters.at(random(0, numberOfCategories))));
    
    for (auto &dat : data) {
      for (Cluster *c : clusters) {
        float newDistance = cv::norm(dat.first->getFeaturesVector() - c->centroid);
        float currentDistance = cv::norm(dat.first->getFeaturesVector() - dat.second->centroid);
        if (newDistance < currentDistance) {
          exchangesPerformed++;
          dat.second = c;
        }
      }
    }
    
  }
  
  while (true) {
    std::vector<cv::Point2d> pointSum;
    std::vector<float> clusterCount;
    
    for (int i = 0; i < numberOfCategories; i++) {
      pointSum.emplace_back(cv::Point2d(0, 0));
      clusterCount.emplace_back(0);
    }
    
    for (auto &dat : data) {
      int idx = 0;
      for (Cluster *c : clusters) {
        if (dat.second == c) {
          pointSum.at(idx) += dat.first->getFeaturesVector();
          clusterCount.at(idx)++;
          //break;
        }
        idx++;
      }
    }
    //Recompute centroid
    for (int i = 0; i < numberOfCategories; i++) {
      clusters.at(i)->centroid = pointSum.at(i) / clusterCount.at(i);
    }
  
    for (int row = 0; row < visualizeKMeansMat.rows; row++) {
      for (int col = 0; col < visualizeKMeansMat.cols; col++) {
        visualizeKMeansMat.at<cv::Vec3b>(row, col) = cv::Vec3b(255, 255, 255);
      }
    }
  
    for (const auto &objPair: data) {
      cv::circle(visualizeKMeansMat, objPair.first->getFeaturesVector() * SIZE, 2,
                 colors[(objPair.second->id + 1) % colorsSize], 1);
    }
  
    for (Cluster *c : clusters) {
      cv::circle(visualizeKMeansMat, c->centroid * SIZE, 3, colors[(c->id + 1) % colorsSize], -1);
    }
  
    cv::imshow("K means visualization", visualizeKMeansMat);
    cv::waitKey(0);
    
    exchangesPerformed = 0;
    for (auto &dat : data) {
      for (Cluster *c : clusters) {
        float newDistance = cv::norm(dat.first->getFeaturesVector() - c->centroid);
        float currentDistance = cv::norm(dat.first->getFeaturesVector() - dat.second->centroid);
        if (newDistance < currentDistance) {
          exchangesPerformed++;
          dat.second = c;
        }
      }
    }
    
    currentLoop++;
    if (exchangesPerformed == 0) break;
    if (currentLoop > maxLoops) break;
  }
  
  for (Cluster *cluster : clusters) {
    std::stringstream ss;
    ss << "Object class " << std::to_string(cluster->id);
    ObjectClass *objClass = new ObjectClass(ss.str(), colors[(cluster->id + 1) % colorsSize]);
    
    for (auto &dat : data) {
      if (dat.second == cluster) {
        objClass->addObject(dat.first);
      }
    }
    objectClasses.emplace_back(objClass);
  }
  
  
  for (Cluster *cluster : clusters) {
    delete cluster;
  }
}
