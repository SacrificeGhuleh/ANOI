#ifndef PCH_H
#define PCH_H

/**
 * @file pch.h
 * @brief Precompiled headers file.
 * Precompiled header file can improve compilation time. Precompiled headers are
 * compiled once and than reused. This file should be filled with often used
 * library header files. e.g. <iostream>. Precompiled headers file should not be
 * included in header files.
 */

//standard cpp libraries
#include <iostream>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>
#include <random>



//opencv headers
//#include <opencv2/opencv.hpp>   //General cv header
#include <opencv2/core/mat.hpp>   //cv::Mat
#include <opencv2/imgcodecs.hpp>  //cv::imread
#include <opencv2/highgui.hpp>    //cv::imshow, cv::waitKey
#include <opencv2/imgproc.hpp>    //cv::imshow, cv::waitKey

template<class T>
inline T SQR(T val) {
  return val * val;
}


#endif //PCH_H