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
	cout << "kernel fro morphological transformations: " << endl << kernel << endl;

	// Morphologic transformations
	// in order to completely delete noises we should apply many times erosion and dilation
	// doing this we are supposing to work in a certain environment where the red ball will
	// not be farer than a certain distance
	
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

bool circle_detection(vector<Point> contour, Point centre, Mat stats) {
	bool detected = false;

	cout << "centre received: " << centre << endl;
	cout << "stats received: " << stats << endl;
	cout << "contour received: " << contour << endl;

	Mat circle_received(1000, 750, CV_8UC3, Scalar(0, 0, 0));
	Scalar color(rand() & 255, rand() & 255, rand() & 255);
	vector<vector<Point>> contours;
	contours.push_back(contour);
	drawContours(circle_received, contours, 0, color);
	circle(circle_received, centre, 3, color, -1, 8);

	namedWindow("circle received", CV_WINDOW_AUTOSIZE);
	imshow("circle received", circle_received);

	// DETECTING CIRCUMFERENCE
	// Signature
	vector<double> circle_reference;
	vector<double> object_signature;
	for (int i = 0; i < contour.size(); i++) {
		double distance = sqrt(pow((centre.x - contour[i].x), 2) + pow((centre.y - contour[i].y), 2));
		circle_reference.push_back(10);
		object_signature.push_back(distance);
		cout << "signature: " << object_signature[i] << endl;
	}

	// Correlation with Circle's signature
	double circumference_threshold = 10;
	double correlation = 0;

	if (!(correlation > circumference_threshold)) {
		return false;
	}

	// Radius too short -> probably it's a noise object


	// DETECTING CIRCLE
	// Area

	return true;
}

Mat object_detecting_labelling(Mat input_image) {
	Mat output_image = input_image;

	// Labelling objects
	cout << "-> labelling objects" << endl;

	int number_of_labels;
	Mat labels, stats, centroids;
	int connectivity = 8;

	number_of_labels = connectedComponentsWithStats(output_image, labels, stats, centroids, connectivity);

	cout << "number_of_labels: " << number_of_labels << endl;
	cout << "stats:" << endl << stats << endl;
	cout << "centroids:" << endl << centroids << endl;

	// Find all closed contours in the binary image.
	cout << "-> finding contours" << endl;

	vector<vector<Point>> contours;
	findContours(output_image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	cout << "Number of countours"  << contours.size() << endl;

	// Find all circles in the image
	cout << "-> finding circles" << endl;

	for (int i = 1; i < number_of_labels; i++) {
		// check shape
		int x_coordinate = (int)centroids.at<double>(i, 0);
		int y_coordinate = (int)centroids.at<double>(i, 1);
		Point centre = Point(x_coordinate, y_coordinate);
		vector<Point> contour = contours[i-1];
		if (labels.at<int>(centre) != labels.at<int>(contour[0])) {
			cout << "ERRORE CENTRO E CONTORNO NON CORRISPONDONO" << endl;
			exit(-1);
		}
		if (circle_detection(contour, centre, stats.row(i))) {
			// bound object

		}
	}

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
	cout << "-> reducing noise" << endl;
//	blur(output_image, output_image, Size(3,3));
//	medianBlur(output_image, output_image, 3);

	int sigma_x = 10;
	int sigma_y = 10;
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

	// combined mask - because red values in HSV space are in 2 different ranges
	Mat combined_mask;
	addWeighted(mask1, 1.0, mask2, 1.0, 0.0, combined_mask);
//	namedWindow("combined_mask", CV_WINDOW_AUTOSIZE);
//	imshow("combined_mask", combined_mask);

	output_image = combined_mask;
	return output_image;
}

// this function take as input an image and recognise circles inside it
Mat image_circle_recognition(Mat input_image) {
	Mat output_image = input_image;

	// application of morphological transformation
	// chiusura - apertura
	cout << "-> application of erosion and dilation" << endl;

	output_image = mophological_transformation_application(output_image.clone());
	namedWindow("morphological transformation");
	imshow("morphological transformation", output_image);

	// objects detectios and labelling
	cout << "-> labelling objects" << endl;

	output_image = object_detecting_labelling(output_image.clone());
	namedWindow("circle recognized");
	imshow("circle recognized", output_image);

	return output_image;
}

int main(int argc, char* argv[]) {
	cout << "inizio programma" << endl;

	// declaration image containers
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
	filtering_result = image_pre_filtering(img.clone());

	// RECOGNITION OF RED OBJECTS
	cout << "red objects detection..." << endl;
	color_result = red_color_filtering(filtering_result.clone());

	// RECOGNITION OF CIRCLE SHAPED OBJECTS
	cout << "circle object recognition..." << endl;
	shape_result = image_circle_recognition(color_result.clone());

	// creation of windows to show the images
	namedWindow("image", CV_WINDOW_AUTOSIZE);
	namedWindow("filtering_result", CV_WINDOW_AUTOSIZE);
	namedWindow("color_result", CV_WINDOW_AUTOSIZE);
//	namedWindow("shape_result", CV_WINDOW_AUTOSIZE);

	// show images in the windows
	imshow("image", img);
	imshow("filtering_result", filtering_result);
	imshow("color_result", color_result);
	imshow("shape_result", shape_result);
	waitKey(0);
	destroyAllWindows();

	cout << "fine programma" << endl;
	getchar();
}