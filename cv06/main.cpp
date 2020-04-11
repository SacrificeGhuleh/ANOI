#include "pch.h"
#include "cli.h"
#include "object.h"
#include "classifier.h"
#include "objectclass.h"
#include "colors.h"
#include "backprop.h"

/**
 * @file main.cpp
 * @brief Template file for openCV
 * */

void trainData(NN *nn, const std::vector<Object *> &objVector);

int getObjClass(NN *nn, const Object *obj);

void assignObjects(NN *nn, const std::vector<Object *> &objVector, std::vector<ObjectClass *> &objClasses);

int main(int argc, char **argv) {

//  if (argc == 1) {
//    printUsage();
//    return 1;
//  }
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
//      } else if (opt == "t") {
//        trainImagePath = args.getNextValue();
//      } else if (opt == "h" || opt == "help") {
//        printUsage();
//        return 1;
//      } else
//        throw std::invalid_argument("invalid argument");
//    }
//
    
    Classifier trainingClassifier(trainImagePath);
    trainingClassifier.preprocess();
    trainingClassifier.classify();
    trainingClassifier.update();
    trainingClassifier.print();
    
    std::vector<ObjectClass *> objectClasses;
    trainingClassifier.autoAssign(3, objectClasses);
    for (ObjectClass *objClass : objectClasses) {
      objClass->printEtalonVector();
    }
    
    trainingClassifier.prepareVisualization();
    trainingClassifier.show();
    
    NN nn(2, 4, 3);
    trainData(&nn, trainingClassifier.getObjects());
    
    {
      Classifier classifier(inputImagePath);
      classifier.preprocess();
      classifier.classify();
      classifier.update();
      assignObjects(&nn, classifier.getObjects(), objectClasses);
      classifier.prepareVisualization();
      classifier.show();
    }
    
    for (ObjectClass *objClass : objectClasses) {
      delete objClass;
    }
    
  }
  catch (std::exception &e) {
    std::cout << "Error: " << e.what() << std::endl;
    return 1;
  }
}


void trainData(NN *nn, const std::vector<Object *> &objVector) {
  int n = objVector.size();
  double **trainingSet = new double *[n];
  
  int i = 0;
  for (const Object *obj : objVector) {
    trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];
    for (int j = 0; j < nn->n[0]; j++) {
      if (j % nn->n[0] == 0)
        trainingSet[i][j] = obj->getFeatureOne();
      if (j % nn->n[0] == 1)
        trainingSet[i][j] = obj->getFeatureTwo();
    }
    
    for (int numClasses = 0; numClasses < nn->n[2]; numClasses++) {
      trainingSet[i][nn->n[0] + numClasses] = 0.0;
      if (numClasses == obj->getObjectClass()->getId()) {
        trainingSet[i][nn->n[0] + numClasses] = 1.0;
      }
    }
    
    i++;
  }
  
  i = 0;
  double error = 1.0;
  while (error > 0.01) {
    int k = i % n;
    nn->setInput(trainingSet[k]);
    nn->feedforward();
    error = nn->backpropagation(&trainingSet[i % n][nn->n[0]]);
    if (i % 1000 == 0) {
      printf("  Iteration: %d  err=%0.3f\n", i, error);
    }
    i++;
    
  }
  printf(" (%i iterations) result error: %f\n", i, error);
  
  for (i = 0; i < n; i++) {
    delete[] trainingSet[i];
  }
  delete[] trainingSet;
}

void assignObjects(NN *nn, const std::vector<Object *> &objVector, std::vector<ObjectClass *> &objClasses) {
  for (Object *obj : objVector) {
    int id = getObjClass(nn, obj);
    for (ObjectClass *objClass : objClasses) {
      if (id == objClass->getId()) {
        obj->setObjectClass(objClass);
      }
    }
  }
}

int getObjClass(NN *nn, const Object *obj) {
  double *in = new double[nn->n[0]];
  for (int j = 0; j < nn->n[0]; j++) {
    if (j % nn->n[0] == 0)
      in[j] = obj->getFeatureOne();
    if (j % nn->n[0] == 1)
      in[j] = obj->getFeatureTwo();
  }
  
  nn->setInput(in);
  nn->feedforward();
  int output = nn->getOutput();
  
  delete[] in;
  
  return output;
}