#pragma once

class NN {
public:
  NN(int n, int h, int o);
  
  virtual ~NN();
  
  void feedforward();
  
  double backpropagation(double *t);
  
  void setInput(double *in, bool verbose = false);
  
  int getOutput(bool verbose = false);

//private:
  
  int *n; //! @brief pocty neuronu
  int l; //! @brief pocet vrstev
  double ***w; //! @brief vahy
  
  double *in; //! @brief vstupni vektor
  double *out; //! @brief vystupni vektor
  double **y; //! @brief vystupni vektory vrstev
  
  double **d; //! @brief chyby neuronu
};