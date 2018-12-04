#include "opencv\cv.hpp"
#include <iostream>
#include "MeanShift.h"

using namespace cv;
using namespace std;

// take an image in BGR colour space and apply the MeanShift segmentation agorithm
// return the segmented image in BGR colur space
Mat mean_shift_segmentation_application(Mat input_image) {
	
	Mat output_image = input_image;

	// Convert color from RGB to Lab
	cvtColor(output_image, output_image, CV_HSV2BGR);
	cvtColor(output_image, output_image, CV_BGR2Lab);

	// Initilize Mean Shift with spatial bandwith and color bandwith
	MeanShift MSProc(8, 16);

	// Filtering Process
	MSProc.MSFiltering(output_image);

	// Segmentation Process include Filtering Process (Region Growing)
	MSProc.MSSegmentation(output_image);
	
	// Print the bandwith
	cout<<"the Spatial Bandwith is "<<MSProc.hs<<endl;
	cout<<"the Color Bandwith is "<<MSProc.hr<<endl;

	// Convert color from Lab to RGB
	cvtColor(output_image, output_image, CV_Lab2BGR);
	cvtColor(output_image, output_image, CV_BGR2HSV);

	return output_image;
}

// take an image and apply some filters in order to have better processing results
// both input and output images are in BGR colour space
Mat image_pre_filtering(Mat input_image) {
	Mat output_image;

	// reduce dimension
	Size size(500, 500);
	resize(input_image, output_image, size);//resize image
	namedWindow("cropped_img", CV_WINDOW_AUTOSIZE);
	imshow("cropped_img", output_image);

	// noise reduction
	Mat noise_reducted;
//	blur(output_image, output_image, Size(3,3));
//	medianBlur(output_image, output_image, 3);

	int sigma_x = 0;
	int sigma_y = 0;
	GaussianBlur(output_image, noise_reducted, Size(3, 3), sigma_x, sigma_y);
	namedWindow("noise_reducted", CV_WINDOW_AUTOSIZE);
	imshow("noise_reducted", noise_reducted);

	// segmentation - application of meanshift algorithm
	Mat segmentation_applied;

	segmentation_applied = mean_shift_segmentation_application(noise_reducted);
	namedWindow("mean shift segmentation");
	imshow("mean shift segmentation", segmentation_applied);

	output_image = segmentation_applied;
	return output_image;
}

// take an image and select only objects of the colour in the range we are interested in
// the input image must be in GBR colour space
// the output image will be returned as GBR colour space
Mat red_color_filtering(Mat input_image) {

	Mat output_image;

	// convert from RGB to HSV
	cvtColor(input_image, output_image, COLOR_BGR2HSV);
//	cout << "numero canali immagine HSV: " << output_image.channels();

	// range of red colors
	// mask1
	Mat mask1;
	Scalar lower_red_hue_range1 = Scalar(0, 100, 100);
	Scalar upper_red_hue_range1 = Scalar(10, 255, 255);
	inRange(output_image, lower_red_hue_range1, upper_red_hue_range1, mask1);
//	namedWindow("mask1", CV_WINDOW_AUTOSIZE);
//	imshow("mask1", mask1);

	// mask2
	Mat mask2;
	Scalar lower_red_hue_range2 = Scalar(170, 100, 100);
	Scalar upper_red_hue_range2 = Scalar(180, 255, 255);
	inRange(output_image, lower_red_hue_range2, upper_red_hue_range2, mask2);
//	namedWindow("mask2", CV_WINDOW_AUTOSIZE);
//	imshow("mask2", mask1);

	// combined mask - because redvalues in HSV space are in 2 different ranges
	Mat combined_mask;
	addWeighted(mask1, 1.0, mask2, 1.0, 0.0, combined_mask);
//	namedWindow("combined_mask", CV_WINDOW_AUTOSIZE);
//	imshow("combined_mask", combined_mask);

	output_image = combined_mask;
	return output_image;
}

// this function take as input an image and recognise circles inside it
Mat image_circle_recognition(Mat input_image) {
	Mat output_image;

	// chiusura - apertura


	// label objects


	// correlation with sign of a circle


	return output_image;
}

int main(int argc, char* argv[]) {
	cout << "inizio programma" << endl;

	// declaration image container
	Mat img, filtering_result, color_result, shape_result;

	// read image
	img = imread("C:/Users/lucac_000/source/repos/RedBallRecognising/ImagesDataset/redball2.jpg", CV_LOAD_IMAGE_COLOR);

	// check reading result
	if (!img.data) {
		cout << "errore lettura immagine mandril.jpg" << endl;
		getchar();
		return -1;
	}

	// PRE-FILTERING
	filtering_result = image_pre_filtering(img);

	// RECOGNITION OF RED OBJECTS
	color_result = red_color_filtering(filtering_result);

	// RECOGNITION OF CIRCLE SHAPED OBJECTS


	// creation of windows to show the images
	namedWindow("image", CV_WINDOW_AUTOSIZE);
	namedWindow("filtering_result", CV_WINDOW_AUTOSIZE);
	namedWindow("color_result", CV_WINDOW_AUTOSIZE);

	// show images in the windows
	imshow("image", img);
	imshow("filtering_result", filtering_result);
	imshow("color_result", color_result);
	waitKey(0);
	destroyAllWindows();

	cout << "fine programma" << endl;
	getchar();
}