//
// Created by zvone on 27-Feb-20.
//

#ifndef DZO2019_TRAINEDFEATURES_H
#define DZO2019_TRAINEDFEATURES_H

#include <vector>

class Classifier_;
class Object;

class TrainedFeatures {
  
  Classifier_ *classifier_;
  std::vector<Object*> objects;
};


#endif //DZO2019_TRAINEDFEATURES_H
