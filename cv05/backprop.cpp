#include "pch.h"
#include "backprop.h"
#include "utils.h"

#define LAMBDA 1.0
#define ETA 0.1

void randomize(double *p, int n) {
  for (int i = 0; i < n; i++) {
    p[i] = random(0, 1);
  }
}

/**
 * @brief Creates neural network
 * @param n Number of input layer neurons
 * @param h Number of hidden layer neurons
 * @param o Number of output layer neurons
 * @return Ptr to created neural network
 * @warning Do not forget to destroy neural network at the end of the program. --> releaseNN
 */
NN *createNN(int n, int h, int o) {
  NN *nn = new NN;
  
  nn->n = new int[3];
  nn->n[0] = n;
  nn->n[1] = h;
  nn->n[2] = o;
  nn->l = 3;
  
  nn->w = new double **[nn->l - 1];
  
  
  for (int k = 0; k < nn->l - 1; k++) {
    nn->w[k] = new double *[nn->n[k + 1]];
    for (int j = 0; j < nn->n[k + 1]; j++) {
      nn->w[k][j] = new double[nn->n[k]];
      randomize(nn->w[k][j], nn->n[k]);
      // BIAS
      //nn->w[k][j] = new double[nn->n[k] + 1];
      //randomize( nn->w[k][j], nn->n[k] + 1 );
    }
  }
  
  nn->y = new double *[nn->l];
  for (int k = 0; k < nn->l; k++) {
    nn->y[k] = new double[nn->n[k]];
    memset(nn->y[k], 0, sizeof(double) * nn->n[k]);
  }
  
  nn->in = nn->y[0];
  nn->out = nn->y[nn->l - 1];
  
  nn->d = new double *[nn->l];
  for (int k = 0; k < nn->l; k++) {
    nn->d[k] = new double[nn->n[k]];
    memset(nn->d[k], 0, sizeof(double) * nn->n[k]);
  }
  
  return nn;
}


void releaseNN(NN *&nn) {
  for (int k = 0; k < nn->l - 1; k++) {
    for (int j = 0; j < nn->n[k + 1]; j++) {
      delete[] nn->w[k][j];
    }
    delete[] nn->w[k];
  }
  delete[] nn->w;
  
  for (int k = 0; k < nn->l; k++) {
    delete[] nn->y[k];
  }
  delete[] nn->y;
  
  for (int k = 0; k < nn->l; k++) {
    delete[] nn->d[k];
    
  }
  delete[] nn->d;
  
  delete[] nn->n;
  
  delete nn;
  nn = nullptr;
}

void feedforward(NN *nn) {
  for (int k = 1; k < nn->l; k++) {
    for (int i = 0; i < nn->n[k]; i++) {
      double weight = 0.0;
      for (int j = 0; j < nn->n[k - 1]; j++) {
        weight += nn->w[k - 1][i][j] * nn->y[k - 1][j];
      }
      double res = 1.0 / (1.0 + exp(-LAMBDA * weight));
      nn->y[k][i] = res;
    }
  }
}

double backpropagation(NN *nn, double *t) {
  //For each layer
  for (int k = nn->l - 1; k >= 0; k--) {
    //If output layer
    if (k == nn->l - 1) {
      //For each neuron
      for (int i = 0; i < nn->n[k]; i++) {
        double delta = nn->y[k][i] * (1 - nn->y[k][i]);
        float error = t[i] - nn->y[k][i];
        
        nn->d[k][i] = error * LAMBDA * delta;
      }
      // Else if hidden layer
    } else if (k != 0) {
      //For each neuron
      for (int i = 0; i < nn->n[k]; i++) {
        double errorResult = 0.0;
        //Neurons from upper layer
        for (int j = 0; j < nn->n[k + 1]; j++) {
          errorResult += nn->d[k + 1][j] * nn->w[k][j][i];
        }
        nn->d[k][i] = errorResult * LAMBDA * (nn->y[k][i] * (1 - nn->y[k][i]));
      }
    }
  }
  
  //update weights
  //layers
  for (int k = 0; k < nn->l - 1; k++) {
    //upper layer
    for (int i = 0; i < nn->n[k + 1]; i++) {
      //lower layer
      for (int j = 0; j < nn->n[k]; j++) {
        nn->w[k][i][j] = nn->w[k][i][j] + ETA * nn->d[k + 1][i] * nn->y[k][j];
      }
    }
  }
  
  //compute error
  double error = 0.0;
  for (int n = 0; n < nn->n[nn->l - 1]; n++) {
    error += SQR(t[n] - nn->y[nn->l - 1][n]);
  }
  error /= 2.0;
  
  return error;
}

void setInput(NN *nn, double *in, bool verbose) {
  memcpy(nn->in, in, sizeof(double) * nn->n[0]);
  
  if (verbose) {
    printf("input=(");
    for (int i = 0; i < nn->n[0]; i++) {
      printf("%0.3f", nn->in[i]);
      if (i < nn->n[0] - 1) {
        printf(", ");
      }
    }
    printf(")\n");
  }
}

int getOutput(NN *nn, bool verbose) {
  double max = 0.0;
  int max_i = 0;
  if (verbose) printf(" output=");
  for (int i = 0; i < nn->n[nn->l - 1]; i++) {
    if (verbose) printf("%0.3f ", nn->out[i]);
    if (nn->out[i] > max) {
      max = nn->out[i];
      max_i = i;
    }
  }
  if (verbose) printf(" -> %d\n", max_i);
  if (nn->out[0] > nn->out[1] && nn->out[0] - nn->out[1] < 0.1) return 2;
  return max_i;
}
