/**
 * @file objectDetection.cpp
 * @author A. Huaman ( based in the classic facedetect.cpp in samples/c )
 * @brief A simplified version of facedetect.cpp, show how to load a cascade classifier and how to find objects (Face + eyes) in a video stream
 */
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/utility.hpp"

#include "opencv2/highgui/highgui_c.h"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
string fface_cas_fn = "haarcascade_frontalface_alt.xml";
string pface_cas_fn = "haarcascade_profileface.xml";

string fface_lbp_cas_fn = "lbpcascade_frontalface.xml";
string pface_lbp_cas_fn = "lbpcascade_profileface.xml";
//string eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";

CascadeClassifier frontal_face_cascade;
CascadeClassifier profile_face_cascade;
//CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

/**
 * @function main
 */
int main( void )
{
  CvCapture* capture;
  Mat frame;

  //-- 1. Load the cascades
  if( !frontal_face_cascade.load( fface_cas_fn ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !profile_face_cascade.load( pface_cas_fn ) ){ printf("--(!)Error loading\n"); return -1; };
  //if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

  //-- 2. Read the video stream
  capture = cvCaptureFromCAM( -1 );
  if( capture )
  {
    for(;;)
    {
      double t = (double)getTickCount();
      frame = cv::cvarrToMat(cvQueryFrame( capture ));

      //-- 3. Apply the classifier to the frame
      if( !frame.empty() )
      { detectAndDisplay( frame ); }
      else
      { printf(" --(!) No captured frame -- Break!"); break; }

      t = ((double)getTickCount() - t)/getTickFrequency();
      printf("time per detection cycle: %f\n", t);

      int c = waitKey(10);
      if( (char)c == 'x' ) { break; }

    }
  }
  return 0;
}

/**
 * @function detectAndDisplay
 */
void detectAndDisplay( Mat frame )
{
  //lower the resolution so to speed up detection
  Mat dsframe; //down sampled frame
  resize( frame, dsframe, Size(0,0), 0.5, 0.5);

  std::vector<Rect> faces;
  std::vector<Rect> pfaces;
  Mat frame_gray;

  cvtColor( dsframe, frame_gray, COLOR_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  //Size mysize = frame_gray.size();
  //printf("width==%d height==%d\n", mysize.width, mysize.height);

  //-- Detect faces
  // both frontal and profile face detection are considered
  frontal_face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
  profile_face_cascade.detectMultiScale( frame_gray, pfaces, 1.1, 2, 0|CASCADE_SCALE_IMAGE, Size(30, 30) );
  faces.insert(faces.end(),pfaces.begin(),pfaces.end());

  for( size_t i = 0; i < faces.size(); i++ )
  {
    Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
    ellipse( dsframe, center, Size( faces[i].width/2, faces[i].height/2), 0, 0, 360, Scalar( 255, 0, 255 ), 2, 8, 0 );
  }
  //-- Show what you got
  imshow( window_name, dsframe );
}
