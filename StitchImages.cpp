#include <iostream>
#include<opencv2/opencv.hpp>
#include "StitchFunctions.h"
using namespace std;
using namespace cv;

int main()
{
    //images to be stitched
    Mat img1 = imread("C:\\Users\\Prady\\PycharmProjects\\opencv\\IMG_3595.jpg");
    Mat img2 = imread("C:\\Users\\Prady\\PycharmProjects\\opencv\\IMG_3596.jpg");
    Mat img3 = imread("C:\\Users\\Prady\\PycharmProjects\\opencv\\IMG_3597.jpg");
    Mat img4 = imread("C:\\Users\\Prady\\PycharmProjects\\opencv\\IMG_3598.jpg");

    vector<Mat> images;
    images.push_back(img1);
    images.push_back(img2);
    images.push_back(img3);
    images.push_back(img4);

    //stitch images
    StitchFunctions stit;
    Mat pano = stit.StitchImages(images);
    
    //display final stitched image
    namedWindow("image", WINDOW_NORMAL);
    imshow("image", pano);
    waitKey(0);

    //saving image
    bool saved = imwrite("./stitched.png", pano);

    return 0;
}


