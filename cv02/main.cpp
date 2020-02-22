#include "pch.h"
#include "colors.h"
#include "cli.h"


/**
 * @file main.cpp
 * @brief Template file for openCV
 * */


constexpr uint8_t threshold = 128;
constexpr uint8_t noData = 0;
constexpr uint8_t toSegmentData = 255;
uint16_t foundObjects = 0;
std::vector<cv::Point> toSegmentIndexes;


class Object {
public:
  explicit Object(uint16_t index) : index_(index), perimeter_(0) {
    color_ = colors[(index_ - 1) % colorsSize];
  }
  
  uint16_t getArea() const {
    return indexes_.size();
  }
  
  cv::Point getCenterOfMass() const {
    cv::Point centerOfMass(0, 0);
    for (const cv::Point &point : indexes_) {
      centerOfMass += point;
    }
    centerOfMass /= getArea();
    return centerOfMass;
  }
  
  uint16_t getCircumference() const {
    uint16_t circumference = 0;
    cv::Point centerOfMass = getCenterOfMass();
    for (const cv::Point &point : indexes_) {
      circumference = std::max<uint16_t>(std::abs(point.x - centerOfMass.x), circumference);
      circumference = std::max<uint16_t>(std::abs(point.y - centerOfMass.y), circumference);
    }
    return circumference;
  }
  
  std::string toString() const {
    std::stringstream stringStream;
    cv::Point centerOfmass = getCenterOfMass();
    stringStream <<
                 "Object : " << index_ << "\n" <<
                 "  color: " << (int) color_[0] << " " << (int) color_[1] << " " << (int) color_[2] << "\n" <<
                 "  area: " << getArea() << "\n" <<
                 "  center of mass [" << centerOfmass.x << "," << centerOfmass.y << "]\n" <<
                 "  circumference: " << getCircumference() << "\n" <<
                 "  perimeter: " << perimeter_;
    return stringStream.str();
  }
  
  void computePerimeter(cv::Mat_<uint8_t> &filterImg) {
    perimeter_ = 0;
    perimeterPoints_.clear();
    
    for (const cv::Point &point : indexes_) {
      if (point.x < 1 || point.y < 1 || point.x >= filterImg.cols || point.y >= filterImg.rows)
        continue;
      
      const uint8_t locPix = filterImg.at<uint8_t>(point);
      
      cv::Point point1(point.x + 1, point.y);
      cv::Point point2(point.x - 1, point.y);
      cv::Point point3(point.x, point.y + 1);
      cv::Point point4(point.x, point.y - 1);
      
      const uint8_t locPix1 = filterImg.at<uint8_t>(point1);
      const uint8_t locPix2 = filterImg.at<uint8_t>(point2);
      const uint8_t locPix3 = filterImg.at<uint8_t>(point3);
      const uint8_t locPix4 = filterImg.at<uint8_t>(point4);
      
      if (locPix != locPix1 ||
          locPix != locPix2 ||
          locPix != locPix3 ||
          locPix != locPix4) {
        perimeterPoints_.emplace_back(point);
        perimeter_++;
      }
    }
  }
  
  uint16_t getPerimeter() const {
    return perimeter_;
  }
  
  uint16_t perimeter_;
//private:
  uint16_t index_;
  std::vector<cv::Point> indexes_;
  std::vector<cv::Point> perimeterPoints_;
  cv::Vec3b color_;
};

std::vector<Object *> objects;

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
  if (currentPoint < 0) {
    
    currentPoint = ++foundObjects;
    objects.emplace_back(new Object(currentPoint));
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

void printUsage() {
  std::cout << "Image Segmentation" << std::endl;
  std::cout << "Usage: [-f input]" << std::endl;
}

int main(int argc, char **argv) {
  
  if (argc == 1) {
    printUsage();
    return 1;
  }
  std::string inputImagePath = "";
  try {
    ArgParser args(argc, argv);
    while (args.hasNext()) {
      std::string opt = args.getNextOpt();
      if (opt == "f") {
        inputImagePath = args.getNextValue();
      } else if (opt == "h" || opt == "help") {
        printUsage();
        return 1;
      } else
        throw std::invalid_argument("invalid argument");
    }
    
    const cv::Mat_<uint8_t> srcImg = cv::imread(inputImagePath, cv::IMREAD_GRAYSCALE);
    cv::Mat_<uint8_t> filterImg = srcImg.clone();
    cv::Mat_<cv::Vec3b> filterColorImg(filterImg.rows, filterImg.cols);
    cv::Mat_<cv::Vec3b> classificationColorImg(filterImg.rows, filterImg.cols);
    
    
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
    
    for (const cv::Point &point : toSegmentIndexes) {
      checkPoint(point, filterImg);
    }
    
    for (Object *obj : objects) {
      //Recompute perimeter
      obj->computePerimeter(filterImg);
      
      printf("%s\n", obj->toString().c_str());
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
    
    printf("Found segments: %d\n", foundObjects);
    
    cv::imshow("TrainImage", srcImg);
    cv::imshow("FilterColorImage", filterColorImg);
    cv::imshow("ClassificationColorImage", classificationColorImg);
    
    for (Object *obj : objects) {
      delete obj;
      obj = nullptr;
    }
    
    cv::waitKey();
  }
  catch (std::exception &e) {
    std::cout << "Error: " << e.what() << std::endl;
    return 1;
  }
}