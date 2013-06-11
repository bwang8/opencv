CC = g++
CFLAGS = `pkg-config --cflags opencv`
LDFLAGS = `pkg-config --libs opencv`

all: dettrackdemo

dettrackdemo: objdettrack.o dettrackdemo.cpp
	$(CC) $(CFLAGS) -o $@ -g $+ $(LDFLAGS)

objdettrack.o: objdettrack.cpp
	$(CC) $(CFLAGS) -o $@ -c -g $+

## Non important example code test/play to learn opencv
#faceDet: faceDetection.cpp
#	$(CC) $(CFLAGS) -o faceDet -g faceDetection.cpp $(LDFLAGS)

#objDet: objectDetection.cpp
#	$(CC) $(CFLAGS) -o objDet -g objectDetection.cpp $(LDFLAGS)
##

clean:
	rm *.o dettrackdemo
	rm -r *.dSYM/ 
