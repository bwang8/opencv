CC = g++
CFLAGS = `pkg-config --cflags opencv`
LDFLAGS = `pkg-config --libs opencv`

all: dettrackdemo faceDet objDet

dettrackdemo: objdettrack.o dettrackdemo.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ -g $+

objdettrack.o: objdettrack.cpp
	$(CC) $(CFLAGS) -o $@ -c -g $+

## Non important example code test/play to learn opencv
faceDet: faceDetection.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o faceDet -g faceDetection.cpp 

objDet: objectDetection.cpp
	$(CC) $(CFLAGS) $(LDFLAGS) -o objDet -g objectDetection.cpp 
##

clean:
	rm *.o faceDet objDet dettrackdemo
	rm -r *.dSYM/ 
