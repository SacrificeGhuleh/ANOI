//
// Created by zvone on 11-Apr-20.
//
#include "pch.h"

std::random_device device;
typedef std::mt19937 Engine;
typedef std::uniform_real_distribution<double> Distribution;

auto uniform_generator = std::bind(Distribution(0.0f, 1.0f), Engine(device()));

double random(const double range_min, const double range_max) {
  double ksi;
  #pragma omp critical ( random )
  {
    ksi = static_cast<double>( uniform_generator());
  }
  return ksi * (range_max - range_min) + range_min;
}