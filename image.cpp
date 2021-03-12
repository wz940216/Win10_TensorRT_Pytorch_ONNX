#include <opencv2/opencv.hpp>
#include "image.hpp"

static const float kMean[3] = { 0.485f, 0.456f, 0.406f };
static const float kStdDev[3] = { 0.229f, 0.224f, 0.225f };
static const int map_[7][3] = { {0,0,0} ,
				{128,0,0},
				{0,128,0},
				{0,0,128},
				{128,128,0},
				{128,0,128},
				{0,128,0}};


float* normal(cv::Mat img) {
	//cv::Mat image(img.rows, img.cols, CV_32FC3);
	float * data;
	data = (float*)calloc(img.rows*img.cols * 3, sizeof(float));

	for (int c = 0; c < 3; ++c)
	{
        
		for (int i = 0; i < img.rows; ++i)
		{ //获取第i行首像素指针 
			cv::Vec3b *p1 = img.ptr<cv::Vec3b>(i);
			//cv::Vec3b *p2 = image.ptr<cv::Vec3b>(i);
			for (int j = 0; j < img.cols; ++j)
			{
				data[c * img.cols * img.rows + i * img.cols + j] = (p1[j][c] / 255.0f - kMean[c]) / kStdDev[c];
       
			}
		}
        
	}
	return data;
}