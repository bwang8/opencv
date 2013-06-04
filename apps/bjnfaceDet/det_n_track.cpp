#include "opencv2/opencv.hpp"

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <math.h> 
#include <string>

using namespace cv;
using namespace std;

string fface_cas_fn = "../../data/haarcascades/haarcascade_frontalface_alt.xml";
string pface_cas_fn = "../../data/haarcascades/haarcascade_profileface.xml";

string fface_lbp_cas_fn = "../../data/lbpcascades/lbpcascade_frontalface.xml";
string pface_lbp_cas_fn = "../../data/lbpcascades/lbpcascade_profileface.xml";

string window_name = "Capture - Face detection";
//RNG rng(12345);

void displayColorHist(string windowNum, int hsize, Mat& hist){
  string windowName = "color histogram "+windowNum;
  namedWindow(windowName);

  Mat histimg = Mat::zeros(200, 320, CV_8UC3);
  int binW = histimg.cols / hsize;
  Mat buf(1, hsize, CV_8UC3);
  for( int i = 0; i < hsize; i++ )
      buf.at<Vec3b>(i) = Vec3b(saturate_cast<uchar>(i*180./hsize), 255, 255);
  cvtColor(buf, buf, COLOR_HSV2BGR);

  for( int i = 0; i < hsize; i++ )
  {
      int val = saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
      rectangle( histimg, Point(i*binW,histimg.rows),
                 Point((i+1)*binW,histimg.rows - val),
                 Scalar(buf.at<Vec3b>(i)), -1, 8 );
  }

  imshow(windowName, histimg);
}

void histPeakAccent(Mat& hist, int farthestBinFromPeak){
  float max = 0;
  int max_ind = 0;
  int hsize = hist.size().height;
  for(int i=0; i<hsize; i++){
    if(max < hist.at<float>(i)){
      max = hist.at<float>(i);
      max_ind = i;
    }
  }

  for(int i=max_ind+1; i<hsize && i<=max_ind+farthestBinFromPeak; i++){
    hist.at<float>(i) = hist.at<float>(i)*exp(-(i-max_ind));
  }
  for(int i=max_ind-1; i>=0 && i>=max_ind-farthestBinFromPeak; i--){
    hist.at<float>(i) = hist.at<float>(i)*exp((i-max_ind));
  }
  for(int i=0; i<hsize; i++){
    if(i < max_ind-farthestBinFromPeak || max_ind+farthestBinFromPeak < i){
      hist.at<float>(i) = 0;
    }
  }
}

void thereisnobluepeople(Mat& hist){
  int hsize = hist.size().height;
  for(int i=(hsize*2/5); i<(int)(hsize*4/5); i++){
    hist.at<float>(i) = hist.at<float>(i)*0.3;
  }
}

void camTracking(const Mat& currframe, vector<Rect>& trackingWindow, vector<Mat>& objHueHist, Mat& dispWindow){
  assert(trackingWindow.size() > 0);

  //convert to hsv and extract hue
  Mat hsv, hue;
  cvtColor(currframe, hsv, COLOR_BGR2HSV);
  int chs[] = {0, 0};
  hue.create(hsv.size(), hsv.depth());
  mixChannels(&hsv, 1, &hue, 1, chs, 1);

  //create mask for pixels too black, white, or gray
  Mat mask;
  int vmin = 10, vmax = 256, smin = 30;
  inRange(hsv, Scalar(0, smin, MIN(vmin, vmax)), Scalar(180, 256, MAX(vmin, vmax)), mask);

  const int hsize = 32;
  float hranges[] = {0, 180};
  const float* phranges = hranges;
  const int ch = 0;
  //if new objects detected or old object lost or refresh requested, then recreate color histograms
  if(objHueHist.size() != trackingWindow.size()){
    objHueHist.clear();

    //create color histogram for a new object
    for(int i=0; i<trackingWindow.size(); i++){
      Mat roi(hue, trackingWindow[i]);

      //create a mask that pass through only the oval/ellipse of the face detection window
      Mat maskellipse = Mat::zeros(mask.size(), CV_8UC1);
      Rect myrect = trackingWindow[i];
      RotatedRect myrotrect = RotatedRect(Point2f(myrect.x+myrect.width/2, myrect.y+myrect.height/2),
        Size2f(myrect.width, myrect.height), 0);
      ellipse( maskellipse, myrotrect, Scalar(255), -1, 8);
      maskellipse &= mask;
      Mat maskroi(maskellipse, trackingWindow[i]);

      objHueHist.push_back(Mat());
      calcHist(&roi, 1, &ch, maskroi, objHueHist[i], 1, &hsize, &phranges);
      
      //DEBUG
      //display color histogram before suppressing non-peak bins
      normalize(objHueHist[i], objHueHist[i], 0, 255, NORM_MINMAX);
      displayColorHist("1", hsize, objHueHist[i]);

      //anti-blue people suppression
      thereisnobluepeople(objHueHist[i]);
      int farthestBinFromPeak = 3;
      histPeakAccent(objHueHist[i], farthestBinFromPeak);
      normalize(objHueHist[i], objHueHist[i], 0, 255, NORM_MINMAX);
      
      //DEBUG
      //display color histogram after suppressing non-peak bins
      displayColorHist("2", hsize, objHueHist[i]);
    }
  }
 
  //backprojection and camshift
  for(int i=0; i<trackingWindow.size(); i++){
    Mat backproj;

    calcBackProject(&hue, 1, &ch, objHueHist[i], backproj, &phranges);
    backproj &= mask;

    RotatedRect trackBox = CamShift(backproj, trackingWindow[i], TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));

    //draw the tracking boxes onto the new frame
    ellipse( dispWindow, trackBox, Scalar(0,0,255), 3, LINE_AA );
    rectangle( dispWindow, trackingWindow[i], Scalar(0,255,255), 3, LINE_AA);
  } 

}


void resizeRect(Rect& myrect, double widthScale, double heightScale){
  myrect.x = (int) (myrect.x * widthScale);
  myrect.y = (int) (myrect.y * heightScale);

  myrect.width = (int) (myrect.width * widthScale);
  myrect.height = (int) (myrect.height * heightScale);
}

//remove detection rectangles/windows that have too much overlap with each other
//currently algorithm is fairly stupid (O(n^2)), 
//but opencv may have optimization that make matrix operations fast
void removeOverlapWindows(Size frameSize, vector<Rect>& detResult, double overlap_frac_criteria=0.5){
  Mat detectBitFlip = Mat::zeros(frameSize, CV_8UC1);

  for(vector<Rect>::iterator it = detResult.begin(); it != detResult.end(); ){
    Mat detRect(detectBitFlip, *it);

    //calculate the portion of the detection rectangle being already detected
    double num_ones = detRect.dot(Mat::ones(detRect.size(),CV_8UC1));
    double overlap_frac = num_ones / (*it).area();

    if(overlap_frac > overlap_frac_criteria){
      detResult.erase(it);
    } else{
      detRect = 1; //set all pixels in detection rectangle to 1
      it++;
    }
  }
}

Point2f transformPt(Mat affM, Point2f pt){
  return Point2f(
    affM.at<float>(1,1)*pt.x+affM.at<float>(1,2)*pt.y+affM.at<float>(1,3),
    affM.at<float>(2,1)*pt.x+affM.at<float>(2,2)*pt.y+affM.at<float>(2,3));
}

void displayFaceBox(string winName, Mat& frame, vector<Rect> cascadeDetectionResults){
  namedWindow(winName);
  for(int myi=0; myi<cascadeDetectionResults.size(); myi++)
    rectangle(frame, cascadeDetectionResults[0], Scalar(255,0,255), 3, LINE_AA);
  imshow(winName, frame);
}

vector<Rect> casDetect(Mat currframe, vector<CascadeClassifier> all_cas, Mat& dispWindow, bool detectAfterRot=false){
  //lower the resolution so to speed up detection
  double shrinkratio = 0.7;
  Mat dsframe; //down sampled frame
  resize( currframe, dsframe, Size(0,0), shrinkratio, shrinkratio );

  Mat frame_gray;
  cvtColor( dsframe, frame_gray, COLOR_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //put all detection results hereS
  vector<Rect> detResult;
  
  //simple straight (no rotation) cascade face/object detection
  for(int i=0; i<all_cas.size(); i++){
    vector<Rect> casResult;
    all_cas[i].detectMultiScale(frame_gray, casResult, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
    detResult.insert(detResult.end(), casResult.begin(), casResult.end() );

    displayFaceBox("straight", frame_gray, casResult);
  }
 
  //implements small angle rotations here. Could be really slow
  if(detectAfterRot){
    vector<double> rotAngles;
    //rotAngles.push_back(-25);
    rotAngles.push_back(25);

    for(int ang_ind=0; ang_ind<rotAngles.size(); ang_ind++){
      double PI = 3.14159265;
      double w = frame_gray.size().width;
      double h = frame_gray.size().height;
      double mysin = sin(rotAngles[ang_ind]*PI/180);
      double mycos = cos(rotAngles[ang_ind]*PI/180);
      double nw = h * mysin + w * mycos;
      double nh = h * mycos + w * mysin;
      Point2f src_pts[] = {Point2f(0,0), Point2f(w,0), Point2f(0,h)};
      Point2f tar_pts[] = {Point2f(0,w * mysin), Point2f(w * mycos), Point2f(h * mysin, nh)};
      Mat rotM = getAffineTransform(src_pts, tar_pts);
      Mat revRotM;
      invertAffineTransform(rotM, revRotM);

      Mat frameAfterRot;
      warpAffine(frame_gray, frameAfterRot, rotM, Size2f(nw, nh));

      vector<Rect> rotDetResult;
      for(int i=0; i<all_cas.size(); i++){
        vector<Rect> casResultRot;
        all_cas[i].detectMultiScale(frameAfterRot, casResultRot, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30));
        rotDetResult.insert(rotDetResult.end(), casResultRot.begin(), casResultRot.end());
      }
      printf("detected ==%d== sideways\n", (int) rotDetResult.size() );

      displayFaceBox("frameAfterRot", frameAfterRot, rotDetResult);

      vector<Rect> casResultOrigCoord;
      for(int j=0; j<casResultRot.size(); j++){
        Rect cr = casResultRot[j];
        Point2f crr_initializer[] = 
          {transformPt(revRotM, Point2f(cr.x,cr.y)),
           transformPt(revRotM, Point2f(cr.x+cr.width,cr.y)),
           transformPt(revRotM, Point2f(cr.x,cr.y+cr.height)),
           transformPt(revRotM, Point2f(cr.x+cr.width,cr.y+cr.height))};
        RotatedRect crr = RotatedRect();
        crr.points(crr_initializer);
        casResultOrigCoord.push_back(crr.boundingRect());
      }
      displayFaceBox("frame_gray", frame_gray, casResultOrigCoord);
      detResult.insert(detResult.end(), casResultOrigCoord.begin(), casResultOrigCoord.end());
    } 

  }

  //eliminate duplicates/overlapping detection
  //they are defined as detection that are overlapping >50% area with other detection rectangles
  printf("=pre overlap elimination: detected %d faces/objects\n", (int) detResult.size());
  removeOverlapWindows(dsframe.size(), detResult, 0.5);
  printf("post overlap elimination: detected %d faces/objects\n", (int) detResult.size());

  //reverse downsampling of image, by resizing detection rectangles/windows
  for(int i=0; i<detResult.size(); i++){
    resizeRect(detResult[i], 1/shrinkratio, 1/shrinkratio);
  }

  //draw detected objects/faces onto dispWindow
  for(int i=0; i<detResult.size(); i++){
    rectangle(dispWindow, detResult[i], Scalar(0,255,0), 3, LINE_AA);
  }

  return detResult;
}

int main(){
  //initialize capturing
  VideoCapture cap;
  int camNum = 0; //webcam
  cap.open(camNum);
  if( !cap.isOpened() ){
    cout << "***Could not initialize capturing...***\n";
    cout << "Current parameter's value: \n";
    return -1;
  }
 
  //load pretrained cascade classifiers in xml format
  CascadeClassifier fface_cas;
  CascadeClassifier pface_cas;
  if( !fface_cas.load( fface_cas_fn ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !pface_cas.load( pface_cas_fn ) ){ printf("--(!)Error loading\n"); return -1; };

  //place all cascade classifiers together
  vector<CascadeClassifier> all_cas;
  all_cas.push_back(fface_cas);
  all_cas.push_back(pface_cas);

  //create display window
  namedWindow("haar cascade and camshift", 0);

  /*
    detect-cycle 
    ?? 1) every time tracking window goes to the edge of the picture 
    * 2) when a face/object experiences a huge change or being unstable in size
    3) every 100 cycles
    4) every time 2 face/object overlap in tracking (maybe camshift will handle it fine? no it does not)
    ?? 5) every time a face/object significantly change its color
    6) when there is no face detected ( * and there was detection in recent cycles)
    * 7) when validation failed inside face tracking phase
  */

  //start detection and tracking loop
  Mat currframe;
  vector<Rect> detResult;
  vector<Mat> objHueHist;
  int detOrTrackFlag = 0; //0 for detection, 1 for tracking
  int num_tracked_objects = 0;

  for(int cycle = 0; ;cycle++){
    cap>>currframe;
    if(currframe.empty()){
      break;
    }

    //display
    Mat dispWindow;
    currframe.copyTo(dispWindow);

    switch(detOrTrackFlag){
    //DETECTION PHASE
    case 0:
      detResult = casDetect(currframe, all_cas, dispWindow, true);
      num_tracked_objects = detResult.size();

      if(detResult.size() > 0){
        detOrTrackFlag = 1;
        objHueHist.clear(); //clear to regenerate color histogram
      }

      break;
    //TRACKING PHASE
    case 1:
      //make camTracking return flags that warn about signs of false detection/tracking
      camTracking(currframe, detResult, objHueHist, dispWindow);

      //redo detection if too many cycles passed without a detection phase
      if(cycle%30 == 0){
        detOrTrackFlag = 0;
      }

      //redo detection if large overlap between tracking window
      if(cycle%5 == 0){
        removeOverlapWindows(currframe.size(), detResult, 0.3);
        if(detResult.size() < num_tracked_objects){
          detOrTrackFlag = 0;
          printf("overlap during tracking, redo detection.\n");
        }
      }

      //redo detection if horizontal:vertical window ratio is too high
      //expect face to be upright

      break;
    }

    //display
    imshow("haar cascade and camshift", dispWindow);
  }

}

