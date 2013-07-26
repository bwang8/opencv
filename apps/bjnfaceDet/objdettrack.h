#ifndef OBJDETTRACK_H
#define OBJDETTRACK_H

#include "opencv2/opencv.hpp"

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <math.h> 
#include <string>

class ObjDetTrack{
public:
  std::vector<cv::Rect> casDetect(const cv::Mat& currframe, cv::Mat& dispWindow, bool detectAfterRot=false);
  std::vector<cv::Rect> camTracking(const cv::Mat& currframe, std::vector<cv::Rect>& trackingWindow, cv::Mat& dispWindow);
  void removeOverlapWindows(cv::Size frameSize, std::vector<cv::Rect>& trackingWindow, double overlap_frac_criteria=0.5);
  cv::Mat updateConfidenceMap(std::vector<cv::Rect> detResult, int detOrTrackUpdateFlag, cv::Size2i mapSize);

  //debugging display functions
  void displayFaceBox(std::string winName, cv::Mat& frame, std::vector<cv::Rect> cascadeDetectionResults);
  void displayColorHist(std::string winName, int hsize, cv::Mat& hist);

  ObjDetTrack();
  ObjDetTrack(std::vector<cv::CascadeClassifier> allcas, std::vector<cv::Mat> objHueHist, double shrinkratio, cv::Mat newConfidenceMap);
  std::vector<cv::CascadeClassifier> getAllCas();
  void setAllCas(std::vector<cv::CascadeClassifier> newAllCas);
  std::vector<cv::Mat> getObjHueHist();
  void setObjHueHist(std::vector<cv::Mat> newObjHueHist);
  double getShrinkRatio();
  void setShrinkRatio(double newShrinkRatio);

private:
  std::vector<cv::CascadeClassifier> allcas;
  std::vector<cv::Mat> objHueHist; 
  double shrinkratio;
  cv::Mat confidenceMap;

  //helper for CAMshift tracking
  void histPeakAccent(cv::Mat& hist, int farthestBinFromPeak);
  void thereisnobluepeople(cv::Mat& hist);

  //helper for haar wavelet cascade face/object detection
  std::vector<cv::Rect> runAllCascadeOnFrame(const cv::Mat& frame);
  //extra functionality beyond opencv's rotation ability
  //keep entire image without cropping parts that don't fit in old frame
  cv::Mat rotateFrame(const cv::Mat& frame, cv::Mat& frameAfterRot, double rotDegree);
  std::vector<cv::Rect> revRotOnRects(std::vector<cv::Rect> rotDetResult, cv::Mat revRotM, cv::Size2f orig_size);

  //general helper/functionality that opencv should have
  cv::Point2f transformPt(cv::Mat affM, cv::Point2f pt);
  void resizeRect(cv::Rect& myrect, double widthScale, double heightScale);

};

#endif