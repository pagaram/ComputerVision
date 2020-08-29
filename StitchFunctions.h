#include <iostream>
#include<opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class StitchFunctions
{
private:	
	void createMask(Mat& stit, Mat& Mk, Mat& th);
	void cropPanorama(Mat& stit, Mat& Mk, Mat& th);
	int getMaxAreaContourId(vector <vector<Point>> contours);
public:
	Mat StitchImages(vector<Mat> images);
}; 
