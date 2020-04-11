#include "pch.h"
#include "cli.h"
#include "backprop.h"
#include "utils.h"


void train(NN *nn);

void test(NN *nn, int num_samples = 10);

/**
 * @file main.cpp
 * @brief Template file for openCV
 * */

int main(int argc, char **argv) {
  NN *nn = createNN(2, 4, 2);
  train(nn);
  
  //getchar();
  
  test(nn, 100);
  
  //getchar();
  
  releaseNN(nn);
  
  return 0;
}


void train(NN *nn) {
  printf("Training\n");
  int n = 1000;
  double **trainingSet = new double *[n];
  for (int i = 0; i < n; i++) {
    trainingSet[i] = new double[nn->n[0] + nn->n[nn->l - 1]];
    
    bool classA = i % 2;
    
    for (int j = 0; j < nn->n[0]; j++) {
      if (classA) {
        trainingSet[i][j] = 0.1 * random(0, 1) + 0.6;
      } else {
        trainingSet[i][j] = 0.1 * random(0, 1) + 0.2;
      }
    }
    
    trainingSet[i][nn->n[0]] = (classA) ? 1.0 : 0.0;
    trainingSet[i][nn->n[0] + 1] = (classA) ? 0.0 : 1.0;
  }
  
  double error = 1.0;
  unsigned long long i = 0;
  while (error > 0.001) {
    setInput(nn, trainingSet[i % n]);
    feedforward(nn);
    error = backpropagation(nn, &trainingSet[i % n][nn->n[0]]);
    i++;
    if (i % 1000 == 0) {
      printf("  Iteration: %d  err=%0.3f\n", i, error);
    }
  }
  printf("\n (%d iterations)\n", i);
  
  for (i = 0; i < n; i++) {
    delete[] trainingSet[i];
  }
  delete[] trainingSet;
}

void test(NN *nn, int num_samples) {
  double *in = new double[nn->n[0]];
  
  int num_err = 0;
  for (int n = 0; n < num_samples; n++) {
    bool classA = (random(0, 1) > 0.5);
    
    for (int j = 0; j < nn->n[0]; j++) {
      if (classA) {
        in[j] = 0.1 * random(0, 1) + 0.6;
      } else {
        in[j] = 0.1 * random(0, 1) + 0.2;
      }
    }
    printf("predicted: %d\n", !classA);
    setInput(nn, in, true);
    
    feedforward(nn);
    int output = getOutput(nn, true);
    if (output == classA) {
      printf("error\n");
      num_err++;
    }
    printf("\n");
  }
  double err = (double) num_err / num_samples;
  printf("test error: %.2f\n", err);
}
