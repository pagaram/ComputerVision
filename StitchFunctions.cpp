#include <iostream>
#include<opencv2/opencv.hpp>
#include "StitchFunctions.h"
using namespace std;
using namespace cv;

Mat StitchFunctions::StitchImages(vector<Mat> images)
{
	Mat pano;

	//initial opencv stitcher
	Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
	Stitcher::Status status = stitcher->stitch(images, pano);

	//creating mask to crop image
	StitchFunctions stitchFunc;
	Mat thresh = Mat::zeros(pano.rows, pano.cols, CV_64FC1);
	Mat mask = Mat::zeros(pano.rows, pano.cols, CV_64FC1);
	stitchFunc.createMask(pano, mask, thresh);
	
	//image crop
	stitchFunc.cropPanorama(pano, mask, thresh);

	return pano;
}

int StitchFunctions::getMaxAreaContourId(vector <vector<Point>> contours)
{
	double maxArea = 0;
	int maxAreaContourId = -1;
	for (int j = 0; j < contours.size(); j++) {
		double newArea = contourArea(contours.at(j));
		if (newArea > maxArea) {
			maxArea = newArea;
			maxAreaContourId = j;
		} 
	}
	return maxAreaContourId;
}

void StitchFunctions::createMask(Mat& stit, Mat& Mk, Mat& th)
{
	//making border
	Mat dst;
	Scalar value(0, 0, 0);
	dst = Mat::zeros(stit.rows, stit.cols, CV_64FC1);
	copyMakeBorder(stit, dst, 10, 10, 10, 10, BORDER_CONSTANT, value);
	stit = dst;

	//grayscaling image
	Mat gray = Mat::zeros(stit.rows, stit.cols, CV_64FC1);
	cvtColor(stit, gray, COLOR_BGR2GRAY);

	//binarizing image
	th = Mat::zeros(stit.rows, stit.cols, CV_64FC1);
	threshold(gray, th, 0, 255, THRESH_BINARY);

	//finding contours
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	findContours(th, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	int contourID = getMaxAreaContourId(contours);
	vector<Point> contour_poly;
	approxPolyDP(contours[contourID], contour_poly, 3, true);
	Rect boundRect = boundingRect(Mat(contour_poly));
	
	rectangle(Mk, boundRect, Scalar(255, 0, 0, 0), -1, 8); //creating mask
	
	//adjusting mask size
	Mat MkTemp = Mat::zeros(stit.rows, stit.cols, CV_64FC1);

	for (int i = 0; i < Mk.rows; i++)
	{
		for (int j = 0; j < Mk.cols; j++)
		{
			MkTemp.at<double>(i, j) = Mk.at<double>(i, j);
		}
	}
	
	int xedge = MkTemp.cols - boundRect.x;
	int yedge = MkTemp.rows - boundRect.y;

	int xstart = boundRect.x - 1;
	int ystart = boundRect.y - 1;
	
	for (int i = ystart; i < yedge; i++)
	{
		for (int j = xstart; j < xedge; j++)
		{
			MkTemp.at<double>(i, j) = 255;
		}
	}
	
	Mk = Mat::zeros(MkTemp.rows, MkTemp.cols, CV_64FC1);
	Mk = MkTemp;	
}

void StitchFunctions::cropPanorama(Mat& stit, Mat& Mk, Mat& th)
{
	Mat minRect = Mk;
	Mat eroded = Mat::zeros(Mk.rows, Mk.cols, CV_64FC1);		
	th.convertTo(th, CV_64F);
	
	//erode mask 20 times
	for (int i = 0; i < 20; i++)
	{
		erode(minRect, eroded, Mat());
		minRect = eroded;
	}

	//finding contours of mask to determine bounding
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	minRect.convertTo(minRect,CV_8UC1);
	findContours(minRect, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
	int contourID = getMaxAreaContourId(contours);
	vector<Point> contour_poly;
	approxPolyDP(contours[contourID], contour_poly, 3, true);
	Rect boundRect = boundingRect(Mat(contour_poly));
	
	//cropping stitched image
	Mat cropped = stit(boundRect);
	stit = cropped;	
}