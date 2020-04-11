#include "pch.h"
#include "backprop.h"
#include "utils.h"

#define LAMBDA 1.0
#define ETA 0.1

void randomize(double *p, int n);

/**
 * @brief Creates neural network
 * @param n Number of input layer neurons
 * @param h Number of hidden layer neurons
 * @param o Number of output layer neurons
 */
NN::NN(int n, int h, int o) {
  this->n = new int[3];
  this->n[0] = n;
  this->n[1] = h;
  this->n[2] = o;
  this->l = 3;
  
  this->w = new double **[this->l - 1];
  
  
  for (int k = 0; k < this->l - 1; k++) {
    this->w[k] = new double *[this->n[k + 1]];
    for (int j = 0; j < this->n[k + 1]; j++) {
      this->w[k][j] = new double[this->n[k]];
      randomize(this->w[k][j], this->n[k]);
      // BIAS
      //this->w[k][j] = new double[this->n[k] + 1];
      //randomize( this->w[k][j], this->n[k] + 1 );
    }
  }
  
  this->y = new double *[this->l];
  for (int k = 0; k < this->l; k++) {
    this->y[k] = new double[this->n[k]];
    memset(this->y[k], 0, sizeof(double) * this->n[k]);
  }
  
  this->in = this->y[0];
  this->out = this->y[this->l - 1];
  
  this->d = new double *[this->l];
  for (int k = 0; k < this->l; k++) {
    this->d[k] = new double[this->n[k]];
    memset(this->d[k], 0, sizeof(double) * this->n[k]);
  }
}

NN::~NN() {
  for (int k = 0; k < this->l - 1; k++) {
    for (int j = 0; j < this->n[k + 1]; j++) {
      delete[] this->w[k][j];
    }
    delete[] this->w[k];
  }
  delete[] this->w;
  
  for (int k = 0; k < this->l; k++) {
    delete[] this->y[k];
  }
  
  for (int k = 0; k < this->l; k++) {
    delete[] this->d[k];
  }
  
  delete[] this->y;
  delete[] this->d;
  delete[] this->n;
}

void NN::feedforward() {
  for (int k = 1; k < this->l; k++) {
    for (int i = 0; i < this->n[k]; i++) {
      double weight = 0.0;
      for (int j = 0; j < this->n[k - 1]; j++) {
        weight += this->w[k - 1][i][j] * this->y[k - 1][j];
      }
      double res = 1.0 / (1.0 + exp(-LAMBDA * weight));
      this->y[k][i] = res;
    }
  }
}

double NN::backpropagation(double *t) {
  //For each layer
  for (int k = this->l - 1; k >= 0; k--) {
    //If output layer
    if (k == this->l - 1) {
      //For each neuron
      for (int i = 0; i < this->n[k]; i++) {
        double delta = this->y[k][i] * (1 - this->y[k][i]);
        float error = t[i] - this->y[k][i];
        
        this->d[k][i] = error * LAMBDA * delta;
      }
      // Else if hidden layer
    } else if (k != 0) {
      //For each neuron
      for (int i = 0; i < this->n[k]; i++) {
        double errorResult = 0.0;
        //Neurons from upper layer
        for (int j = 0; j < this->n[k + 1]; j++) {
          errorResult += this->d[k + 1][j] * this->w[k][j][i];
        }
        this->d[k][i] = errorResult * LAMBDA * (this->y[k][i] * (1 - this->y[k][i]));
      }
    }
  }
  
  //update weights
  //layers
  for (int k = 0; k < this->l - 1; k++) {
    //upper layer
    for (int i = 0; i < this->n[k + 1]; i++) {
      //lower layer
      for (int j = 0; j < this->n[k]; j++) {
        this->w[k][i][j] = this->w[k][i][j] + ETA * this->d[k + 1][i] * this->y[k][j];
      }
    }
  }
  
  //compute error
  double error = 0.0;
  for (int n = 0; n < this->n[this->l - 1]; n++) {
    error += SQR(t[n] - this->y[this->l - 1][n]);
  }
  error /= 2.0;
  
  return error;
}

void NN::setInput(double *in, bool verbose) {
  memcpy(this->in, in, sizeof(double) * this->n[0]);
  
  if (verbose) {
    printf("input=(");
    for (int i = 0; i < this->n[0]; i++) {
      printf("%0.3f", this->in[i]);
      if (i < this->n[0] - 1) {
        printf(", ");
      }
    }
    printf(")\n");
  }
}

int NN::getOutput(bool verbose) {
  double max = 0.0;
  int max_i = 0;
  if (verbose) printf(" output=");
  for (int i = 0; i < this->n[this->l - 1]; i++) {
    if (verbose) printf("%0.3f ", this->out[i]);
    if (this->out[i] > max) {
      max = this->out[i];
      max_i = i;
    }
  }
  if (verbose) printf(" -> %d\n", max_i);
  if (this->out[0] > this->out[1] && this->out[0] - this->out[1] < 0.1) return 2;
  return max_i;
}

void randomize(double *p, int n) {
  for (int i = 0; i < n; i++) {
    p[i] = random(0, 1);
  }
}
