#include "objdettrack.h"

Mat ObjDetTrack::updateConfidenceMap(vector<Rect> detResult, int detOrTrackUpdateFlag, Size2i mapSize){
  if( confidenceMap.empty() || detOrTrackUpdateFlag == 0){
    printf("initializing new confidence map\n");
    confidenceMap = Mat::ones(mapSize, CV_32FC1);
    normalize(confidenceMap, confidenceMap, mapSize.area(), 0, NORM_L1);
  }

  for(int i=0; i<detResult.size(); i++){
    Mat confroi;
    if(detOrTrackUpdateFlag == 0){
      //expand window in y direction (heighten/elongate) if detection 
      detResult[i].y = max(detResult[i].y-0.3*detResult[i].height, 0.0);
      detResult[i].height = min(1.6*detResult[i].height, (double)mapSize.height-detResult[i].y);

      confroi = confidenceMap(detResult[i]);
      confroi = 5; 
      //constant because this is detection phase, confidence map resets at beginning of phase
    } else{
      //expand window in x direction (widen) if tracking
      detResult[i].x = max(detResult[i].x-0.4*detResult[i].width, 0.0);
      detResult[i].width = min(1.8*detResult[i].width, (double)mapSize.width-detResult[i].x);

      confroi = confidenceMap(detResult[i]);
      confroi += 0.1;
    }
  }

  normalize(confidenceMap, confidenceMap, mapSize.area(), 0, NORM_L1);
  return confidenceMap;
}

void reducePixelRepresentation(Mat& frame, int numLevels){
  //not really quantization (as in picking most frequently occurring color and do NN to cluster every color to these)

  for(int i=0; i<frame.rows; i++){
    for(int j=0; j<frame.cols; j++){
      frame.at<uchar>(i,j) = floor(frame.at<uchar>(i,j)/numLevels) * numLevels;
    }
  }
}

vector<Rect> ObjDetTrack::casDetect(const Mat& currframe, Mat& dispWindow, bool detectAfterRot){
  //lower the resolution so to speed up detection
  //printf("=====shrinkratio = %f\n", shrinkratio);
  Mat dsframe; //down sampled frame
  resize( currframe, dsframe, Size(0,0), shrinkratio, shrinkratio );

  //convert to gray and equalize
  Mat frame_gray;
  cvtColor( dsframe, frame_gray, COLOR_BGR2GRAY );
  equalizeHist( frame_gray, frame_gray );

  reducePixelRepresentation(frame_gray, (256/16));

  //put all detection results here
  vector<Rect> detResult;
  
  //simple straight (no rotation) cascade face/object detection
  vector<Rect> straightDetResult = runAllCascadeOnFrame(frame_gray);
  printf("detected ==%d== straight\n", (int)straightDetResult.size());
  detResult.insert(detResult.end(), straightDetResult.begin(), straightDetResult.end());

  //implements detection after small angle rotations here. Could be really slow
  //only do this if no straight detResults
  if(detectAfterRot){// && detResult.size() == 0){
    vector<double> rotAngles;
    rotAngles.push_back(-30);
    rotAngles.push_back(30);

    for(int ang_ind=0; ang_ind<rotAngles.size(); ang_ind++){
      Mat frameAfterRot;
      Mat revRotM = rotateFrame(frame_gray, frameAfterRot, rotAngles[ang_ind]);
      vector<Rect> rotDetResult = runAllCascadeOnFrame(frameAfterRot);

      ostringstream strs;
      strs << rotAngles[ang_ind];
      string anglestr = strs.str();
      printf("detected ==%d== sideways angle %f\n", (int) rotDetResult.size(), rotAngles[ang_ind]);
      //displayFaceBox("detection after rotation "+anglestr, frameAfterRot, rotDetResult);

      vector<Rect> revRotDetResult = revRotOnRects(rotDetResult, revRotM, frame_gray.size());
      
      displayFaceBox("detection after rotating back "+anglestr, frame_gray, revRotDetResult);
      
      // if(revRotDetResult.size() != 0){
      //   printf("rotated back box 0: %d %d %d %d\n", revRotDetResult[0].x, revRotDetResult[0].y, 
      //     revRotDetResult[0].width, revRotDetResult[0].height);
      // }

      detResult.insert(detResult.end(), revRotDetResult.begin(), revRotDetResult.end());
    }
  }

  //eliminate duplicates/overlapping detection
  //they are defined as detection that are overlapping >50% area with other detection rectangles
  printf("=pre overlap elimination: detected %d faces/objects\n", (int) detResult.size());
  removeOverlapWindows(dsframe.size(), detResult);
  printf("post overlap elimination: detected %d faces/objects\n", (int) detResult.size());

  //reverse downsampling of image, by resizing detection rectangles/windows
  for(int i=0; i<detResult.size(); i++){
    resizeRect(detResult[i], 1/shrinkratio, 1/shrinkratio);
  }

  //draw detected objects/faces onto dispWindow
  for(int i=0; i<detResult.size(); i++){
    rectangle(dispWindow, detResult[i], Scalar(0,255,0), 3, 16);
  }

  return detResult;
}

vector<Rect> ObjDetTrack::camTracking(const Mat& currframe, vector<Rect>& trackingWindow, Mat& dispWindow){
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
      //the point was for the mask to only allow skin color into the histogram,
      //and block out background colors in the corners of the rectangle
      //but there is no point anymore, because I am tuning the color histogram purely for skin hues
      // Mat maskellipse = Mat::zeros(mask.size(), CV_8UC1);
      // Rect myrect = trackingWindow[i];
      // RotatedRect myrotrect = RotatedRect(Point2f(myrect.x+myrect.width/2, myrect.y+myrect.height/2),
      //   Size2f(myrect.width, myrect.height), 0);
      // ellipse( maskellipse, myrotrect, Scalar(255), -1, 8);
      // maskellipse &= mask;
      Mat maskroi(mask, trackingWindow[i]);

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
 
  vector<Rect> boundWindow;
  //backprojection and camshift
  for(int i=0; i<trackingWindow.size(); i++){
    Mat backproj;

    calcBackProject(&hue, 1, &ch, objHueHist[i], backproj, &phranges);
    backproj &= mask;

    //meanShift(backproj, trackingWindow[i], TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
    
    RotatedRect trackBox = CamShift(backproj, trackingWindow[i], TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
    boundWindow.push_back(trackBox.boundingRect());

    //draw the tracking boxes onto the new frame
    ellipse( dispWindow, trackBox, Scalar(0,0,255), 3, 16 );
    rectangle( dispWindow, trackingWindow[i], Scalar(0,255,255), 3, 16);
  } 

  return boundWindow;
}

//remove detection rectangles/windows that have too much overlap with each other
//currently algorithm is fairly stupid (O(n^2)), 
//but opencv may have optimization that make matrix operations fast
//TODO: reimplement using Rect's intersect function
void ObjDetTrack::removeOverlapWindows(Size frameSize, vector<Rect>& trackingWindow, double overlap_frac_criteria){
  Mat detectBitFlip = Mat::zeros(frameSize, CV_8UC1);

  for(vector<Rect>::iterator it = trackingWindow.begin(); it != trackingWindow.end(); ){
    Mat detRect(detectBitFlip, *it);

    //calculate the portion of the detection rectangle being already detected
    double num_ones = detRect.dot(Mat::ones(detRect.size(),CV_8UC1));
    double overlap_frac = num_ones / (*it).area();

    if(overlap_frac > overlap_frac_criteria){
      trackingWindow.erase(it);
    } else{
      detRect = 1; //set all pixels in detection rectangle to 1
      it++;
    }
  }
}

void ObjDetTrack::displayColorHist(string winName, int hsize, Mat& hist){
  string windowName = "color histogram "+winName;
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

void ObjDetTrack::histPeakAccent(Mat& hist, int farthestBinFromPeak){
  float max = 0;
  int max_ind = 0;
  int hsize = hist.size().height;

  //find peak hue
  for(int i=0; i<hsize; i++){
    if(max < hist.at<float>(i)){
      max = hist.at<float>(i);
      max_ind = i;
    }
  }

  if(farthestBinFromPeak <= 0){
    farthestBinFromPeak = 1;
  }

  //hue range wraps around
  for(int i=0; i<hsize; i++){
    int dist2peak = min(abs(i-max_ind),max_ind+(hsize-i));

    //exponential decay hue contribution by distance from peak hue
    if(dist2peak < farthestBinFromPeak){
      hist.at<float>(i) = hist.at<float>(i)*exp(-dist2peak);
    }
    //set hue contribution to 0 for hues too far from peak hue
    else{
      hist.at<float>(i) = 0;
    }
  }
}

void ObjDetTrack::thereisnobluepeople(Mat& hist){
  int hsize = hist.size().height;

  //take advantage of the fact that no one's skin hue is blue
  //reduce hue contribution from blue range of the hue
  for(int i=(hsize*2/5); i<(int)(hsize*4/5); i++){
    hist.at<float>(i) = hist.at<float>(i)*0.3;
  }
}

void ObjDetTrack::resizeRect(Rect& myrect, double widthScale, double heightScale){
  myrect.x = (int) (myrect.x * widthScale);
  myrect.y = (int) (myrect.y * heightScale);

  myrect.width = (int) (myrect.width * widthScale);
  myrect.height = (int) (myrect.height * heightScale);
}

void ObjDetTrack::displayFaceBox(string winName, Mat& frame, vector<Rect> cascadeDetectionResults){
  namedWindow(winName);
  for(int myi=0; myi<cascadeDetectionResults.size(); myi++)
    rectangle(frame, cascadeDetectionResults[myi], Scalar(255), 3, 16);

  if(cascadeDetectionResults.size()!=0)
    imshow(winName, frame);
}

vector<Rect> ObjDetTrack::runAllCascadeOnFrame(const Mat& frame){
  vector<Rect> allCasResult;

  for(int i=0; i<allcas.size(); i++){
    vector<Rect> casResult;
    allcas[i].detectMultiScale(frame, casResult, 1.1, 4, 0|CASCADE_SCALE_IMAGE, Size(30*shrinkratio, 30*shrinkratio) );
    allCasResult.insert(allCasResult.end(), casResult.begin(), casResult.end());
  }

  return allCasResult;
}

Mat ObjDetTrack::rotateFrame(const Mat& frame, Mat& frameAfterRot, double rotDegree){
  Mat rotM = getRotationMatrix2D(1/2*Point2f(frame.cols,frame.rows), rotDegree, 1); 
  vector<Point2f> afterRotCorners;
  afterRotCorners.push_back(transformPt(rotM,Point2f(0,0)));
  afterRotCorners.push_back(transformPt(rotM,Point2f(frame.cols,0)));
  afterRotCorners.push_back(transformPt(rotM,Point2f(0,frame.rows)));
  afterRotCorners.push_back(transformPt(rotM,Point2f(frame.cols,frame.rows)));

  Rect bRect = minAreaRect(afterRotCorners).boundingRect();
  rotM.at<double>(0,2) = -bRect.x;
  rotM.at<double>(1,2) = -bRect.y;
  warpAffine(frame, frameAfterRot, rotM, bRect.size());

  Mat revRotM;
  invertAffineTransform(rotM, revRotM);

  return revRotM;
}

vector<Rect> ObjDetTrack::revRotOnRects(vector<Rect> rotDetResult, Mat revRotM, Size2f orig_size){
  vector<Rect> casResultOrigCoord;

  //Probably use minAreaRect next time I refactor this
  //minAreaRect finds me the bounding RotatedRect which I can use to find 
  //the bounding Rect around the RotatedRect

  for(int j=0; j<rotDetResult.size(); j++){
    Rect cr = rotDetResult[j];
    Point2f crinit[] = 
      {transformPt(revRotM, Point2f(cr.x,cr.y)),
       transformPt(revRotM, Point2f(cr.x+cr.width,cr.y)),
       transformPt(revRotM, Point2f(cr.x,cr.y+cr.height)),
       transformPt(revRotM, Point2f(cr.x+cr.width,cr.y+cr.height))};

    int min_x = orig_size.width;
    int max_x = 0;
    int min_y = orig_size.height;
    int max_y = 0;
    for(int i=0; i<4; i++){
      min_x = min((int)crinit[i].x, min_x);
      max_x = max((int)crinit[i].x, max_x);
      min_y = min((int)crinit[i].y, min_y);
      max_y = max((int)crinit[i].y, max_y);
    }
    min_x = max(min_x, 0);
    max_x = min(max_x, (int)orig_size.width);
    min_y = max(min_y, 0);
    max_y = min(max_y, (int)orig_size.height);
    casResultOrigCoord.push_back( Rect(Point2f(min_x,min_y),Size2f(max_x-min_x,max_y-min_y)) );
  }

  return casResultOrigCoord; 
}

Point2f ObjDetTrack::transformPt(Mat affM, Point2f pt){
  return Point2f(
    affM.at<double>(0,0)*pt.x+affM.at<double>(0,1)*pt.y+affM.at<double>(0,2),
    affM.at<double>(1,0)*pt.x+affM.at<double>(1,1)*pt.y+affM.at<double>(1,2));
}

ObjDetTrack::ObjDetTrack(){
  shrinkratio = 1;
}

ObjDetTrack::ObjDetTrack(vector<CascadeClassifier> newAllCas, 
  vector<Mat> newObjHueHist, double newShrinkRatio, Mat newConfidenceMap){
  allcas = newAllCas;
  objHueHist = newObjHueHist;
  shrinkratio = newShrinkRatio;
  confidenceMap = newConfidenceMap;
}

vector<CascadeClassifier> ObjDetTrack::getAllCas(){
  return allcas;
}

void ObjDetTrack::setAllCas(vector<CascadeClassifier> newAllCas){
  allcas = newAllCas;
}

vector<Mat> ObjDetTrack::getObjHueHist(){
  return objHueHist;
}

void ObjDetTrack::setObjHueHist(vector<Mat> newObjHueHist){
  objHueHist = newObjHueHist;
}

double ObjDetTrack::getShrinkRatio(){
  return shrinkratio;
}

void ObjDetTrack::setShrinkRatio(double newShrinkRatio){
  if(newShrinkRatio>0){
    shrinkratio = newShrinkRatio;
  }
}

//local debugging function, prints out a rotation matrix
void printRotM(Mat revRotM, string title){
    printf("===%s\n",title.c_str());
    printf("affM.size %d %d\n",revRotM.rows,revRotM.cols);
    printf("affM: %f %f %f\n",revRotM.at<double>(0,0),revRotM.at<double>(0,1),revRotM.at<double>(0,2));
    printf("affM: %f %f %f\n",revRotM.at<double>(1,0),revRotM.at<double>(1,1),revRotM.at<double>(1,2));
}



