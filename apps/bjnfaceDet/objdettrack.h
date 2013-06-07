#ifndef OBJDETTRACK_H
#define OBJDETTRACK_H

#include "opencv2/opencv.hpp"

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <math.h> 
#include <string>

using namespace std;
using namespace cv;

class ObjDetTrack{
public:
  vector<Rect> casDetect(const Mat& currframe, Mat& dispWindow, bool detectAfterRot=false);
  vector<Rect> camTracking(const Mat& currframe, vector<Rect>& trackingWindow, Mat& dispWindow);
  void removeOverlapWindows(Size frameSize, vector<Rect>& trackingWindow, double overlap_frac_criteria=0.5);
  Mat updateConfidenceMap(vector<Rect> detResult, int detOrTrackUpdateFlag, Size2i mapSize);

  //debugging display functions
  void displayFaceBox(string winName, Mat& frame, vector<Rect> cascadeDetectionResults);
  void displayColorHist(string winName, int hsize, Mat& hist);

  ObjDetTrack();
  ObjDetTrack(vector<CascadeClassifier> allcas, vector<Mat> objHueHist, double shrinkratio, Mat newConfidenceMap);
  vector<CascadeClassifier> getAllCas();
  void setAllCas(vector<CascadeClassifier> newAllCas);
  vector<Mat> getObjHueHist();
  void setObjHueHist(vector<Mat> newObjHueHist);
  double getShrinkRatio();
  void setShrinkRatio(double newShrinkRatio);

private:
  vector<CascadeClassifier> allcas;
  vector<Mat> objHueHist; 
  double shrinkratio;
  Mat confidenceMap;

  //helper for CAMshift tracking
  void histPeakAccent(Mat& hist, int farthestBinFromPeak);
  void thereisnobluepeople(Mat& hist);

  //helper for haar wavelet cascade face/object detection
  vector<Rect> runAllCascadeOnFrame(const Mat& frame);
  //extra functionality beyond opencv's rotation ability
  //keep entire image without cropping parts that don't fit in old frame
  Mat rotateFrame(const Mat& frame, Mat& frameAfterRot, double rotDegree);
  vector<Rect> revRotOnRects(vector<Rect> rotDetResult, Mat revRotM, Size2f orig_size);

  //general helper/functionality that opencv should have
  Point2f transformPt(Mat affM, Point2f pt);
  void resizeRect(Rect& myrect, double widthScale, double heightScale);

};

#endif