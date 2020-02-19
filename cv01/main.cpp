#include "pch.h"


/**
 * @file main.cpp
 * @brief Template file for openCV
 * */


constexpr uint8_t threshold = 128;
constexpr uint8_t noData = 0;
constexpr uint8_t toSegmentData = 255;
uint16_t foundObjects = 0;
std::vector<cv::Point> toSegmentIndexes;

// List of 20 Simple, Distinct Colors
// https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
cv::Vec3b colors[] = {
    cv::Vec3b(230, 25, 75),
    cv::Vec3b(60, 180, 75),
    cv::Vec3b(255, 225, 25),
    cv::Vec3b(0, 130, 200),
    cv::Vec3b(245, 130, 48),
    cv::Vec3b(145, 30, 180),
    cv::Vec3b(70, 240, 240),
    cv::Vec3b(240, 50, 230),
    cv::Vec3b(210, 245, 60),
    cv::Vec3b(250, 190, 190),
    cv::Vec3b(0, 128, 128),
    cv::Vec3b(230, 190, 255),
    cv::Vec3b(170, 110, 40),
    cv::Vec3b(255, 250, 200),
    cv::Vec3b(128, 0, 0),
    cv::Vec3b(170, 255, 195),
    cv::Vec3b(128, 128, 0),
    cv::Vec3b(255, 215, 180),
    cv::Vec3b(0, 0, 128),
    cv::Vec3b(128, 128, 128)};


//template<class T_MAT_TYPE>
void checkPoint(const cv::Point &point, cv::Mat_<uint8_t> &filterImg, int16_t currentPoint = -1) {
  //Check bounds
  if (point.x < 0 || point.y < 0 || point.x >= filterImg.cols || point.y >= filterImg.rows)
    return;
  
  //
//  T_MAT_TYPE locPix = filterImg.template at<T_MAT_TYPE>(point.x, point.y);
  uint8_t locPix = filterImg.at<uint8_t>(point);
  
  // Already segmented data
  if (locPix != toSegmentData) return;
  
  //If called from main
  if (currentPoint < 0)
    currentPoint = ++foundObjects;
  
  //Add pixel to group
  filterImg.at<uint8_t>(point) = currentPoint;
  
  // fixme: Tohle je fakt hnus, predelej mne :)
  // Get neighbouring 8 pixels (4 are not enough)
  cv::Point point1 = point;
  point1.x += 1;
  
  cv::Point point2 = point;
  point2.x -= 1;
  
  cv::Point point3 = point;
  point3.y += 1;
  
  cv::Point point4 = point;
  point4.y -= 1;
  
  cv::Point point5 = point;
  point5.x += 1;
  point5.y += 1;
  
  cv::Point point6 = point;
  point6.x -= 1;
  point6.y -= 1;
  
  cv::Point point7 = point;
  point7.x -= 1;
  point7.y += 1;
  
  cv::Point point8 = point;
  point8.x += 1;
  point8.y -= 1;
  
  //search only in indexes, no need to search in whole image
  for (const cv::Point &newPoint : toSegmentIndexes) {
    // fixme
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


int main(void) {
  
  /**
   * Load image.
   */
  const cv::Mat_<uint8_t> srcImg = cv::imread("./images/train.png", cv::IMREAD_GRAYSCALE);
  cv::Mat_<uint8_t> filterImg = srcImg.clone();
  cv::Mat_<cv::Vec3b> filterColorImg(filterImg.rows, filterImg.cols);
  
  const uint16_t colorsSize = sizeof(colors) / sizeof(cv::Vec3b);
  
  for (int row = 0; row < filterImg.rows; row++) {
    for (int col = 0; col < filterImg.cols; col++) {
      uint8_t loc_pix = filterImg.at<uint8_t>(row, col);
      
      // Background for segmented image
      filterColorImg.at<cv::Vec3b>(row, col) = cv::Vec3b(noData, noData, noData);
      
      if (loc_pix < threshold) {
        loc_pix = noData;
      } else {
        toSegmentIndexes.emplace_back(col, row);
        loc_pix = toSegmentData;
      }
      filterImg.at<uint8_t>(row, col) = loc_pix;
    }
  }
  
  for (const cv::Point &point : toSegmentIndexes) {
    checkPoint(point, filterImg);
    
    uint8_t locPix = filterImg.at<uint8_t>(point);
    cv::Vec3b locPix3;
    if (locPix == noData || locPix == toSegmentData) {
      locPix3 = cv::Vec3b(locPix, locPix, locPix);
    } else {
      locPix3 = colors[(locPix - 1) % colorsSize];
    }
    
    filterColorImg.at<cv::Vec3b>(point) = locPix3;
    
  }

//  for (int row = 0; row < filterImg.rows; row++) {
//    for (int col = 0; col < filterImg.cols; col++) {
//      uint8_t locPix = filterImg.at<uint8_t>(row, col);
//      //could be moved into loop upwards, could save time
//      cv::Vec3b locPix3;
//      if (locPix == noData || locPix == toSegmentData) {
//        locPix3 = cv::Vec3b(locPix, locPix, locPix);
//      } else {
//        locPix3 = colors[(locPix - 1) % colorsSize];
//      }
//
//      filterColorImg.at<cv::Vec3b>(row, col) = locPix3;
//    }
//  }
  
  printf("Found segments: %d\n", foundObjects);
  
  cv::imshow("TrainImage", srcImg);
  cv::imshow("FilterImage", filterImg);
  cv::imshow("FilterColorImage", filterColorImg);
  cv::waitKey();
}