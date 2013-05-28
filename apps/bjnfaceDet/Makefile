all: face objExample

face: faceDet

faceDet: faceDetection.cpp
	g++ -o faceDet -g faceDetection.cpp -I/usr/local/include/opencv -I/usr/local/include  -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann

objExample: objDet

objDet: objectDetection.cpp
	g++ -o objDet -g objectDetection.cpp -I/usr/local/include/opencv -I/usr/local/include  -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann

clean:
	rm faceDet objDet
