#include "objdettrack.h"

cv::Mat ObjDetTrack::updateConfidenceMap(std::vector<cv::Rect> detResult, int detOrTrackUpdateFlag, cv::Size2i mapSize){
  if( confidenceMap.empty() || detOrTrackUpdateFlag == 0){
    printf("initializing new confidence map\n");
    confidenceMap = cv::Mat::ones(mapSize, CV_32FC1);
    normalize(confidenceMap, confidenceMap, mapSize.area(), 0, cv::NORM_L1);
  }

  for(uint32_t i=0; i<detResult.size(); i++){
    cv::Mat confroi;
    if(detOrTrackUpdateFlag == 0){
      //expand window in y direction (heighten/elongate) if detection 
      detResult[i].y = std::max(detResult[i].y-0.3*detResult[i].height, 0.0);
      detResult[i].height = std::min(1.6*detResult[i].height, (double)mapSize.height-detResult[i].y);

      confroi = confidenceMap(detResult[i]);
      confroi = 5; 
      //constant because this is detection phase, confidence map resets at beginning of phase
    } else{
      //expand window in x direction (widen) if tracking
      detResult[i].x = std::max(detResult[i].x-0.4*detResult[i].width, 0.0);
      detResult[i].width = std::min(1.8*detResult[i].width, (double)mapSize.width-detResult[i].x);

      confroi = confidenceMap(detResult[i]);
      confroi += 0.1;
    }
  }

  normalize(confidenceMap, confidenceMap, mapSize.area(), 0, cv::NORM_L1);
  return confidenceMap;
}

void reducePixelRepresentation(cv::Mat& frame, int numLevels){
  //not really quantization (as in picking most frequently occurring color and do NN to cluster every color to these)

  for(int i=0; i<frame.rows; i++){
    for(int j=0; j<frame.cols; j++){
      frame.at<uchar>(i,j) = floor(frame.at<uchar>(i,j)/numLevels) * numLevels;
    }
  }
}

std::vector<cv::Rect> ObjDetTrack::casDetect(const cv::Mat& currframe, cv::Mat& dispWindow, bool detectAfterRot){
  //lower the resolution so to speed up detection
  //printf("=====shrinkratio = %f\n", shrinkratio);
  cv::Mat dsframe; //down sampled frame
  resize( currframe, dsframe, cv::Size(0,0), shrinkratio, shrinkratio );

  //convert to gray and equalize
  cv::Mat frame_gray;
  cvtColor( dsframe, frame_gray, cv::COLOR_BGR2GRAY );
imwrite("/home/bwang/dev/opencv/apps/bjnfaceDet/Graycap.jpg",frame_gray);
  equalizeHist( frame_gray, frame_gray );
imwrite("/home/bwang/dev/opencv/apps/bjnfaceDet/eqGraycap.jpg",frame_gray);

  reducePixelRepresentation(frame_gray, (256/16));

  //put all detection results here
  std::vector<cv::Rect> detResult;
  
  //simple straight (no rotation) cascade face/object detection
  std::vector<cv::Rect> straightDetResult = runAllCascadeOnFrame(frame_gray);
  printf("detected ==%d== straight\n", (int)straightDetResult.size());
  detResult.insert(detResult.end(), straightDetResult.begin(), straightDetResult.end());

  //implements detection after small angle rotations here. Could be really slow
  //only do this if no straight detResults
  if(detectAfterRot){// && detResult.size() == 0){
    std::vector<double> rotAngles;
    rotAngles.push_back(-30);
    rotAngles.push_back(30);

    for(uint32_t ang_ind=0; ang_ind<rotAngles.size(); ang_ind++){
      cv::Mat frameAfterRot;
      cv::Mat revRotM = rotateFrame(frame_gray, frameAfterRot, rotAngles[ang_ind]);
      std::vector<cv::Rect> rotDetResult = runAllCascadeOnFrame(frameAfterRot);

      std::ostringstream strs;
      strs << rotAngles[ang_ind];
      std::string anglestr = strs.str();
      printf("detected ==%d== sideways angle %f\n", (int) rotDetResult.size(), rotAngles[ang_ind]);
      //displayFaceBox("detection after rotation "+anglestr, frameAfterRot, rotDetResult);

      std::vector<cv::Rect> revRotDetResult = revRotOnRects(rotDetResult, revRotM, frame_gray.size());
      
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
  for(uint32_t i=0; i<detResult.size(); i++){
    resizeRect(detResult[i], 1/shrinkratio, 1/shrinkratio);
  }

  //draw detected objects/faces onto dispWindow
  for(uint32_t i=0; i<detResult.size(); i++){
    rectangle(dispWindow, detResult[i], cv::Scalar(0,255,0), 3, 16);
  }

  return detResult;
}

std::vector<cv::Rect> ObjDetTrack::camTracking(const cv::Mat& currframe, std::vector<cv::Rect>& trackingWindow, cv::Mat& dispWindow){
  assert(trackingWindow.size() > 0);

  //convert to hsv and extract hue
  cv::Mat hsv, hue;
  cvtColor(currframe, hsv, cv::COLOR_BGR2HSV);
  int chs[] = {0, 0};
  hue.create(hsv.size(), hsv.depth());
  mixChannels(&hsv, 1, &hue, 1, chs, 1);

  //create mask for pixels too black, white, or gray
  cv::Mat mask;
  int vmin = 10, vmax = 256, smin = 30;
  inRange(hsv, cv::Scalar(0, smin, MIN(vmin, vmax)), cv::Scalar(180, 256, MAX(vmin, vmax)), mask);

  const int hsize = 32;
  float hranges[] = {0, 180};
  const float* phranges = hranges;
  const int ch = 0;
  //if new objects detected or old object lost or refresh requested, then recreate color histograms
  if(objHueHist.size() != trackingWindow.size()){
    objHueHist.clear();

    //create color histogram for a new object
    for(uint32_t i=0; i<trackingWindow.size(); i++){
      cv::Mat roi(hue, trackingWindow[i]);

      //create a mask that pass through only the oval/ellipse of the face detection window
      //the point was for the mask to only allow skin color into the histogram,
      //and block out background colors in the corners of the rectangle
      //but there is no point anymore, because I am tuning the color histogram purely for skin hues
      // cv::Mat maskellipse = cv::Mat::zeros(mask.size(), CV_8UC1);
      // cv::Rect myrect = trackingWindow[i];
      // cv::RotatedRect myrotrect = cv::RotatedRect(cv::Point2f(myrect.x+myrect.width/2, myrect.y+myrect.height/2),
      //   cv::Size2f(myrect.width, myrect.height), 0);
      // ellipse( maskellipse, myrotrect, cv::Scalar(255), -1, 8);
      // maskellipse &= mask;
      cv::Mat maskroi(mask, trackingWindow[i]);

      objHueHist.push_back(cv::Mat());
      calcHist(&roi, 1, &ch, maskroi, objHueHist[i], 1, &hsize, &phranges);
      
      //DEBUG
      //display color histogram before suppressing non-peak bins
      normalize(objHueHist[i], objHueHist[i], 0, 255, cv::NORM_MINMAX);
      displayColorHist("1", hsize, objHueHist[i]);

      //anti-blue people suppression
      thereisnobluepeople(objHueHist[i]);
      int farthestBinFromPeak = 3;
      histPeakAccent(objHueHist[i], farthestBinFromPeak);
      normalize(objHueHist[i], objHueHist[i], 0, 255, cv::NORM_MINMAX);
      
      //DEBUG
      //display color histogram after suppressing non-peak bins
      displayColorHist("2", hsize, objHueHist[i]);
    }
  }
 
  std::vector<cv::Rect> boundWindow;
  //backprojection and camshift
  for(uint32_t i=0; i<trackingWindow.size(); i++){
    cv::Mat backproj;

    calcBackProject(&hue, 1, &ch, objHueHist[i], backproj, &phranges);
    backproj &= mask;

    //meanShift(backproj, trackingWindow[i], TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));
    
    cv::RotatedRect trackBox = CamShift(backproj, trackingWindow[i], cv::TermCriteria( cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 10, 1 ));
    boundWindow.push_back(trackBox.boundingRect());

    //draw the tracking boxes onto the new frame
    ellipse( dispWindow, trackBox, cv::Scalar(0,0,255), 3, 16 );
    rectangle( dispWindow, trackingWindow[i], cv::Scalar(0,255,255), 3, 16);
  } 

  return boundWindow;
}

//remove detection rectangles/windows that have too much overlap with each other
//currently algorithm is fairly stupid (O(n^2)), 
//but opencv may have optimization that make matrix operations fast
//TODO: reimplement using Rect's intersect function
void ObjDetTrack::removeOverlapWindows(cv::Size frameSize, std::vector<cv::Rect>& trackingWindow, double overlap_frac_criteria){
  cv::Mat detectBitFlip = cv::Mat::zeros(frameSize, CV_8UC1);

  for(std::vector<cv::Rect>::iterator it = trackingWindow.begin(); it != trackingWindow.end(); ){
    cv::Mat detRect(detectBitFlip, *it);

    //calculate the portion of the detection rectangle being already detected
    double num_ones = detRect.dot(cv::Mat::ones(detRect.size(),CV_8UC1));
    double overlap_frac = num_ones / (*it).area();

    if(overlap_frac > overlap_frac_criteria){
      trackingWindow.erase(it);
    } else{
      detRect = 1; //set all pixels in detection rectangle to 1
      it++;
    }
  }
}

void ObjDetTrack::displayColorHist(std::string winName, int hsize, cv::Mat& hist){
  std::string windowName = "color histogram "+winName;
  cv::namedWindow(windowName);

  cv::Mat histimg = cv::Mat::zeros(200, 320, CV_8UC3);
  int binW = histimg.cols / hsize;
  cv::Mat buf(1, hsize, CV_8UC3);
  for( int i = 0; i < hsize; i++ )
    buf.at<cv::Vec3b>(i) = cv::Vec3b(cv::saturate_cast<uchar>(i*180./hsize), 255, 255);
  cvtColor(buf, buf, cv::COLOR_HSV2BGR);

  for( int i = 0; i < hsize; i++ )
  {
    int val = cv::saturate_cast<int>(hist.at<float>(i)*histimg.rows/255);
    rectangle( histimg, cv::Point(i*binW,histimg.rows),
               cv::Point((i+1)*binW,histimg.rows - val),
               cv::Scalar(buf.at<cv::Vec3b>(i)), -1, 8 );
  }

  imshow(windowName, histimg);
}

void ObjDetTrack::histPeakAccent(cv::Mat& hist, int farthestBinFromPeak){
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
    int dist2peak = std::min( std::min(abs(i-max_ind),max_ind+(hsize-i)) , (hsize-max_ind)+i );

    //exponential decay hue contribution by distance from peak hue
    if(dist2peak < farthestBinFromPeak){
      hist.at<float>(i) = hist.at<float>(i)*exp(-0.5*dist2peak);
    }
    //set hue contribution to 0 for hues too far from peak hue
    else{
      hist.at<float>(i) = 0;
    }
  }
}

void ObjDetTrack::thereisnobluepeople(cv::Mat& hist){
  int hsize = hist.size().height;

  //take advantage of the fact that no one's skin hue is blue
  //reduce hue contribution from blue range of the hue
  for(int i=(hsize*2/5); i<(int)(hsize*4/5); i++){
    hist.at<float>(i) = hist.at<float>(i)*0.3;
  }
}

void ObjDetTrack::resizeRect(cv::Rect& myrect, double widthScale, double heightScale){
  myrect.x = (int) (myrect.x * widthScale);
  myrect.y = (int) (myrect.y * heightScale);

  myrect.width = (int) (myrect.width * widthScale);
  myrect.height = (int) (myrect.height * heightScale);
}

void ObjDetTrack::displayFaceBox(std::string winName, cv::Mat& frame, std::vector<cv::Rect> cascadeDetectionResults){
  cv::namedWindow(winName);
  for(uint32_t myi=0; myi<cascadeDetectionResults.size(); myi++)
    rectangle(frame, cascadeDetectionResults[myi], cv::Scalar(255), 3, 16);

  if(cascadeDetectionResults.size()!=0)
    imshow(winName, frame);
}

std::vector<cv::Rect> ObjDetTrack::runAllCascadeOnFrame(const cv::Mat& frame){
  std::vector<cv::Rect> allCasResult;

  for(uint32_t i=0; i<allcas.size(); i++){
    std::vector<cv::Rect> casResult;
    allcas[i].detectMultiScale(frame, casResult, 1.1, 4, 0|cv::CASCADE_SCALE_IMAGE, cv::Size(30*shrinkratio, 30*shrinkratio) );
    allCasResult.insert(allCasResult.end(), casResult.begin(), casResult.end());
  }

  return allCasResult;
}

cv::Mat ObjDetTrack::rotateFrame(const cv::Mat& frame, cv::Mat& frameAfterRot, double rotDegree){
  cv::Mat rotM = getRotationMatrix2D(1/2*cv::Point2f(frame.cols,frame.rows), rotDegree, 1); 
  std::vector<cv::Point2f> afterRotCorners;
  afterRotCorners.push_back(transformPt(rotM,cv::Point2f(0,0)));
  afterRotCorners.push_back(transformPt(rotM,cv::Point2f(frame.cols,0)));
  afterRotCorners.push_back(transformPt(rotM,cv::Point2f(0,frame.rows)));
  afterRotCorners.push_back(transformPt(rotM,cv::Point2f(frame.cols,frame.rows)));

  cv::Rect bRect = minAreaRect(afterRotCorners).boundingRect();
  rotM.at<double>(0,2) = -bRect.x;
  rotM.at<double>(1,2) = -bRect.y;
  warpAffine(frame, frameAfterRot, rotM, bRect.size());

  cv::Mat revRotM;
  invertAffineTransform(rotM, revRotM);

  return revRotM;
}

std::vector<cv::Rect> ObjDetTrack::revRotOnRects(std::vector<cv::Rect> rotDetResult, cv::Mat revRotM, cv::Size2f orig_size){
  std::vector<cv::Rect> casResultOrigCoord;

  //Probably use minAreaRect next time I refactor this
  //minAreaRect finds me the bounding RotatedRect which I can use to find 
  //the bounding Rect around the RotatedRect

  for(uint32_t j=0; j<rotDetResult.size(); j++){
    cv::Rect cr = rotDetResult[j];
    cv::Point2f crinit[] = 
      {transformPt(revRotM, cv::Point2f(cr.x,cr.y)),
       transformPt(revRotM, cv::Point2f(cr.x+cr.width,cr.y)),
       transformPt(revRotM, cv::Point2f(cr.x,cr.y+cr.height)),
       transformPt(revRotM, cv::Point2f(cr.x+cr.width,cr.y+cr.height))};

    int min_x = orig_size.width;
    int max_x = 0;
    int min_y = orig_size.height;
    int max_y = 0;
    for(int i=0; i<4; i++){
      min_x = std::min((int)crinit[i].x, min_x);
      max_x = std::max((int)crinit[i].x, max_x);
      min_y = std::min((int)crinit[i].y, min_y);
      max_y = std::max((int)crinit[i].y, max_y);
    }
    min_x = std::max(min_x, 0);
    max_x = std::min(max_x, (int)orig_size.width);
    min_y = std::max(min_y, 0);
    max_y = std::min(max_y, (int)orig_size.height);
    casResultOrigCoord.push_back( cv::Rect(cv::Point2f(min_x,min_y),cv::Size2f(max_x-min_x,max_y-min_y)) );
  }

  return casResultOrigCoord; 
}

cv::Point2f ObjDetTrack::transformPt(cv::Mat affM, cv::Point2f pt){
  return cv::Point2f(
    affM.at<double>(0,0)*pt.x+affM.at<double>(0,1)*pt.y+affM.at<double>(0,2),
    affM.at<double>(1,0)*pt.x+affM.at<double>(1,1)*pt.y+affM.at<double>(1,2));
}

ObjDetTrack::ObjDetTrack(){
  shrinkratio = 1;
}

ObjDetTrack::ObjDetTrack(std::vector<cv::CascadeClassifier> newAllCas, 
  std::vector<cv::Mat> newObjHueHist, double newShrinkRatio, cv::Mat newConfidenceMap){
  allcas = newAllCas;
  objHueHist = newObjHueHist;
  shrinkratio = newShrinkRatio;
  confidenceMap = newConfidenceMap;
}

std::vector<cv::CascadeClassifier> ObjDetTrack::getAllCas(){
  return allcas;
}

void ObjDetTrack::setAllCas(std::vector<cv::CascadeClassifier> newAllCas){
  allcas = newAllCas;
}

std::vector<cv::Mat> ObjDetTrack::getObjHueHist(){
  return objHueHist;
}

void ObjDetTrack::setObjHueHist(std::vector<cv::Mat> newObjHueHist){
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
void printRotM(cv::Mat revRotM, std::string title){
    printf("===%s\n",title.c_str());
    printf("affM.size %d %d\n",revRotM.rows,revRotM.cols);
    printf("affM: %f %f %f\n",revRotM.at<double>(0,0),revRotM.at<double>(0,1),revRotM.at<double>(0,2));
    printf("affM: %f %f %f\n",revRotM.at<double>(1,0),revRotM.at<double>(1,1),revRotM.at<double>(1,2));
}



