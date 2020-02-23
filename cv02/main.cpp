#include "pch.h"
#include "cli.h"
#include "object.h"
#include "classifier.h"


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
    std::string inputImagePath = "";
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
    
    
    Classifier classifier(inputImagePath);
    classifier.preprocess();
    classifier.classify();
    classifier.update();
    classifier.print();
    classifier.prepareVisualization();
    classifier.show();
    
  }
  catch (std::exception &e) {
    std::cout << "Error: " << e.what() << std::endl;
    return 1;
  }
}