#include "pch.h"
#include "cli.h"
#include "object.h"
#include "classifier.h"
#include "objectclass.h"
#include "colors.h"


/**
 * @file main.cpp
 * @brief Template file for openCV
 * */

void printUsage() {
  std::cout << "Image Segmentation" << std::endl;
  std::cout << "Usage: [-f input]" << std::endl;
}

int main(int argc, char **argv) {
  
  if (argc == 1) {
    printUsage();
    return 1;
  }
  try {
//    std::string trainImagePath = "";
//    std::string inputImagePath = "";
  
  
    std::string trainImagePath = "images/train.png";
    std::string inputImagePath = "images/test01.png";
    std::string input2ImagePath = "images/test02.png";
    
//    ArgParser args(argc, argv);
//    while (args.hasNext()) {
//      std::string opt = args.getNextOpt();
//      if (opt == "i") {
//        inputImagePath = args.getNextValue();
//      }
//      if (opt == "t") {
//        trainImagePath = args.getNextValue();
//      } else if (opt == "h" || opt == "help") {
//        printUsage();
//        return 1;
//      } else
//        throw std::invalid_argument("invalid argument");
//    }
    
    
    Classifier trainingClassifier(trainImagePath);
    trainingClassifier.preprocess();
    trainingClassifier.classify();
    trainingClassifier.update();
    trainingClassifier.print();
    trainingClassifier.prepareVisualization();
    trainingClassifier.show();
    
    ObjectClass squareObjectClass("Square", Color::Red);
    ObjectClass rectangleObjectClass("Rectangle", Color::Green);
    ObjectClass starObjectClass("Star", Color::Blue);
    
    // Squares
    squareObjectClass.addObject(trainingClassifier.getObjects().at(0));
    squareObjectClass.addObject(trainingClassifier.getObjects().at(1));
    squareObjectClass.addObject(trainingClassifier.getObjects().at(2));
    squareObjectClass.addObject(trainingClassifier.getObjects().at(3));
    
    // Stars
    starObjectClass.addObject(trainingClassifier.getObjects().at(4));
    starObjectClass.addObject(trainingClassifier.getObjects().at(5));
    starObjectClass.addObject(trainingClassifier.getObjects().at(6));
    starObjectClass.addObject(trainingClassifier.getObjects().at(7));
    
    // Rectangles
    rectangleObjectClass.addObject(trainingClassifier.getObjects().at(8));
    rectangleObjectClass.addObject(trainingClassifier.getObjects().at(9));
    rectangleObjectClass.addObject(trainingClassifier.getObjects().at(10));
    rectangleObjectClass.addObject(trainingClassifier.getObjects().at(11));
    
    squareObjectClass.printEtalonVector();
    starObjectClass.printEtalonVector();
    rectangleObjectClass.printEtalonVector();
    
    std::vector<ObjectClass *> objectClasses;
    objectClasses.emplace_back(&squareObjectClass);
    objectClasses.emplace_back(&starObjectClass);
    objectClasses.emplace_back(&rectangleObjectClass);
    
    {
      Classifier classifier(inputImagePath);
      classifier.preprocess();
      classifier.classify();
      classifier.update();
      classifier.print();
      classifier.assignObjects(objectClasses);
      classifier.prepareVisualization();
      classifier.show();
    }
  
    {
      Classifier classifier(input2ImagePath);
      classifier.preprocess();
      classifier.classify();
      classifier.update();
      classifier.print();
      classifier.assignObjects(objectClasses);
      classifier.prepareVisualization();
      classifier.show();
    }
  }
  catch (std::exception &e) {
    std::cout << "Error: " << e.what() << std::endl;
    return 1;
  }
}