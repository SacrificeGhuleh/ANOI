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

//opencv headers
//#include <opencv2/opencv.hpp>   //General cv header
#include <opencv2/core/mat.hpp>   //cv::Mat
#include <opencv2/imgcodecs.hpp>  //cv::imread
#include <opencv2/highgui.hpp>    //cv::imshow, cv::waitKey
#include <opencv2/imgproc.hpp>    //cv::imshow, cv::waitKey

const cv::Vec3b colors[] = {
    cv::Vec3b(230, 25, 75),
    cv::Vec3b(60, 180, 75),
    cv::Vec3b(255, 225, 25),
    cv::Vec3b(0, 130, 200),
    cv::Vec3b(245, 130, 48),
    cv::Vec3b(145, 30, 180),
    cv::Vec3b(70, 240, 240),
    cv::Vec3b(240, 50, 230),
    cv::Vec3b(210, 245, 60),
    cv::Vec3b(250, 190, 190),
    cv::Vec3b(0, 128, 128),
    cv::Vec3b(230, 190, 255),
    cv::Vec3b(170, 110, 40),
    cv::Vec3b(255, 250, 200),
    cv::Vec3b(128, 0, 0),
    cv::Vec3b(170, 255, 195),
    cv::Vec3b(128, 128, 0),
    cv::Vec3b(255, 215, 180),
    cv::Vec3b(0, 0, 128),
    cv::Vec3b(128, 128, 128)};

constexpr uint16_t colorsSize = sizeof(colors) / sizeof(cv::Vec3b);

#endif //PCH_H