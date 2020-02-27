CMAKE_MINIMUM_REQUIRED(VERSION 3.14)
PROJECT(cv03-etalons)

# add libs you need
SET(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)
SET(SRC main.cpp colors.h object.cpp object.h classifier.cpp classifier.h trainedfeatures.cpp trainedfeatures.h objectclass.cpp objectclass.h)
ADD_EXECUTABLE(${PROJECT_NAME} ${SRC} pch.cpp cli.h)
TARGET_LINK_LIBRARIES(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})