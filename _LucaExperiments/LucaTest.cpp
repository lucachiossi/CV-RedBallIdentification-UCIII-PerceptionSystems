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
	cout << "the Spatial Bandwith is " << MSProc.hs << endl;
	cout << "the Color Bandwith is " << MSProc.hr << endl;

	// Convert color from Lab to RGB
	cvtColor(output_image, output_image, CV_Lab2BGR);
	cvtColor(output_image, output_image, CV_BGR2HSV);

	return output_image;
}

// this function takes an image as inputs and apply some morphological transformations
// in this specific case we work with dilation and erosion
Mat mophological_transformation_application(Mat input_image) {
	Mat output_image = input_image;
	Mat kernel;

	// Create kernel of size 3x3. Try different sizes
	kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	// kernel = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
	cout << kernel << endl;

	// Morphologic transformations
	// in order to completely delete noises we should apply many times erosion and dilation
	
	// deleting noise outside
	int noise_outside_reduction_iteration = 3;
	for (int i = 0; i < noise_outside_reduction_iteration; i++) {
		erode(output_image, output_image, kernel);
	}
	for (int i = 0; i < noise_outside_reduction_iteration; i++) {
		dilate(output_image, output_image, kernel);
	}

	// deleting noise inside
	int noise_inside_reduction_iteration = 5;
	for (int i = 0; i < noise_inside_reduction_iteration; i++) {
		dilate(output_image, output_image, kernel);
	}
	for (int i = 0; i < noise_inside_reduction_iteration; i++) {
		erode(output_image, output_image, kernel);
	}

	return output_image;
}

Mat object_labelling(Mat input_image) {
	Mat output_image = input_image;



	return output_image;
}

// take an image and apply some filters in order to have better processing results
// both input and output images are in BGR colour space
Mat image_pre_filtering(Mat input_image) {
	Mat output_image = input_image;

	// reduce dimension
	cout << "-> resizing image" << endl;
	Size size(1000, 750);
	resize(output_image, output_image, size);//resize image
	namedWindow("cropped_img", CV_WINDOW_AUTOSIZE);
	imshow("cropped_img", output_image);

	// noise reduction
//	blur(output_image, output_image, Size(3,3));
//	medianBlur(output_image, output_image, 3);

	int sigma_x = 0;
	int sigma_y = 0;
	GaussianBlur(output_image, output_image, Size(3, 3), sigma_x, sigma_y);
	namedWindow("noise_reducted", CV_WINDOW_AUTOSIZE);
	imshow("noise_reducted", output_image);

	// segmentation - application of meanshift algorithm
	/*cout << "-> mean shift segmentation" << endl;

	output_image = mean_shift_segmentation_application(output_image);
	namedWindow("mean shift segmentation");
	imshow("mean shift segmentation", output_image);*/

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
	cout << "-> colour range operation" << endl;

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
	namedWindow("combined_mask", CV_WINDOW_AUTOSIZE);
	imshow("combined_mask", combined_mask);

	output_image = combined_mask;
	return output_image;
}

// this function take as input an image and recognise circles inside it
Mat image_circle_recognition(Mat input_image) {
	Mat output_image;

	// application of morphological transformation
	// chiusura - apertura
	cout << "-> application of erosion and dilation" << endl;

	output_image = mophological_transformation_application(input_image);
	namedWindow("morphological transformation");
	imshow("morphological transformation", output_image);

	// label objects
	cout << "-> labelling objects" << endl;

	output_image = object_labelling(output_image);
	namedWindow("object labelling");
	imshow("object labelling", output_image);

	// correlation with sign of a circle


	return output_image;
}

int main(int argc, char* argv[]) {
	cout << "inizio programma" << endl;

	// declaration image container
	Mat img, filtering_result, color_result, shape_result;

	// read image
	img = imread("C:/Users/lucac_000/source/repos/RedBallRecognising/ImagesDataset/my_dataset3.jpg", CV_LOAD_IMAGE_COLOR);

	// check reading result
	if (!img.data) {
		cout << "errore lettura immagine mandril.jpg" << endl;
		getchar();
		return -1;
	}

	// PRE-FILTERING
	cout << "pre filtering..." << endl;
	filtering_result = image_pre_filtering(img);

	// RECOGNITION OF RED OBJECTS
	cout << "red objects detection..." << endl;
	color_result = red_color_filtering(filtering_result);

	// RECOGNITION OF CIRCLE SHAPED OBJECTS
	cout << "circle object recognition..." << endl;
	shape_result = image_circle_recognition(color_result);

	// creation of windows to show the images
	namedWindow("image", CV_WINDOW_AUTOSIZE);
	namedWindow("filtering_result", CV_WINDOW_AUTOSIZE);
	namedWindow("color_result", CV_WINDOW_AUTOSIZE);
	namedWindow("shape_result", CV_WINDOW_AUTOSIZE);

	// show images in the windows
	imshow("image", img);
	imshow("filtering_result", filtering_result);
	imshow("color_result", color_result);
	imshow("shape_result", color_result);
	waitKey(0);
	destroyAllWindows();

	cout << "fine programma" << endl;
	getchar();
}