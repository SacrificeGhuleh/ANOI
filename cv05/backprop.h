#pragma once

struct NN {
  int *n; //! @brief pocty neuronu
  int l; //! @brief pocet vrstev
  double ***w; //! @brief vahy
  
  double *in; //! @brief vstupni vektor
  double *out; //! @brief vystupni vektor
  double **y; //! @brief vystupni vektory vrstev
  
  double **d; //! @brief chyby neuronu
};

NN *createNN(int n, int h, int o);

void releaseNN(NN *&nn);

void feedforward(NN *nn);

double backpropagation(NN *nn, double *t);

void setInput(NN *nn, double *in, bool verbose = false);

int getOutput(NN *nn, bool verbose = false);
