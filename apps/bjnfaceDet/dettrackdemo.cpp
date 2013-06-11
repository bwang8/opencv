#include "objdettrack.h"

string fface_cas_fn = "../../data/haarcascades/haarcascade_frontalface_alt.xml";
string pface_cas_fn = "../../data/haarcascades/haarcascade_profileface.xml";

string fface_lbp_cas_fn = "../../data/lbpcascades/lbpcascade_frontalface.xml";
string pface_lbp_cas_fn = "../../data/lbpcascades/lbpcascade_profileface.xml";

string window_name = "Capture - Face detection";
//RNG rng(12345);

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
  vector<CascadeClassifier> allCas;
  allCas.push_back(fface_cas);
  allCas.push_back(pface_cas);

printf("Numtrheads %d\n", getNumThreads());
printf("setting numthreads to 1\n");
setNumThreads(1);
printf("Numtrheads %d\n", getNumThreads());


  //create display window
  namedWindow("haar cascade and camshift", 0);
  namedWindow("confidence map");

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
  int detOrTrackFlag = 0; //0 for detection, 1 for tracking
  int num_tracked_objects = 0;

  ObjDetTrack faceDT = ObjDetTrack(allCas, vector<Mat>(), 0.7, Mat());

  vector<Rect> startingDetResult;

  for(int cycle = 0; ;cycle++){
    cap>>currframe;
    if(currframe.empty()){
      break;
    }

    //display
    Mat dispWindow;
    currframe.copyTo(dispWindow);
    Mat confMap;

    switch(detOrTrackFlag){
    //DETECTION PHASE
    case 0:
      detResult = faceDT.casDetect(currframe, dispWindow, true);

      num_tracked_objects = detResult.size();

      if(detResult.size() > 0){
        detOrTrackFlag = 1;
        faceDT.setObjHueHist(vector<Mat>()); //clear to regenerate color histogram
        startingDetResult = detResult; //make a backup copy
      }

      confMap = faceDT.updateConfidenceMap(detResult, 0, currframe.size());
      imshow("confidence map", confMap);

      if(detOrTrackFlag == 0) break;
      //else continue to tracking phase without getting a new frame from webcam
      //that way with quick movement, you would not use outdated detection window on new capture
    //TRACKING PHASE
    case 1:
      //make camTracking return flags that warn about signs of false detection/tracking
      faceDT.camTracking(currframe, detResult, dispWindow);

      //redo detection if too many cycles passed without a detection phase
      if(cycle%60 == 0){
        detOrTrackFlag = 0;
      }

      //redo detection if large overlap between tracking window
      if(cycle%5 == 0){
        faceDT.removeOverlapWindows(currframe.size(), detResult, 0.3);
        if(detResult.size() < num_tracked_objects){
          detOrTrackFlag = 0;
          printf("overlap during tracking, redo detection.\n");
        }
      }

      //redo detection if horizontal:vertical window ratio is too high
      //expect face to be upright

      //if tracking box for faces grew too much since the detection phase, redo detection
      for(int i=0; i<startingDetResult.size(); i++){
        if(detResult[i].area() > 4*startingDetResult[i].area()){
          detOrTrackFlag = 0;
        }
      }

      confMap = faceDT.updateConfidenceMap(detResult, 1, currframe.size());
      imshow("confidence map", confMap);
      break;
    }

    char c = (char)waitKey(10);
    switch(c){
    case 'd':
      printf("======current shrink ratio : %f\n", faceDT.getShrinkRatio());
      faceDT.setShrinkRatio(faceDT.getShrinkRatio()-0.05);
      break;
    case 'u':
      printf("======current shrink ratio : %f\n", faceDT.getShrinkRatio());
      faceDT.setShrinkRatio(faceDT.getShrinkRatio()+0.05);
      break;
    default:
      break;
    }

    //super-impose heat map onto display
    Mat tempMat;
    normalize(confMap, tempMat, 0, 0.5, NORM_MINMAX);
    Mat tempMap;
    cvtColor(tempMat, tempMap, COLOR_GRAY2RGB);
    Mat temp;
    tempMap.convertTo(temp, CV_8U, 256);
    dispWindow += temp;

    // for(int i=0; i<confMap.rows; i++){
    //   for(int j=0; j<confMap.cols; j++){
    //     dispWindow
    //   }
    // }

    //display
    imshow("haar cascade and camshift", dispWindow);
  }

}
