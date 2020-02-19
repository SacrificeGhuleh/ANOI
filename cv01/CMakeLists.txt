CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
PROJECT(cv01-image-segmentation)

# add libs you need
SET(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
SET(SRC main.cpp)
ADD_EXECUTABLE(${PROJECT_NAME} ${SRC} pch.cpp)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})