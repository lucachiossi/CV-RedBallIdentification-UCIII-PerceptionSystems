#include "opencv\cv.hpp"
#include <iostream>
#include "MeanShift.h"
#include "PearsonCorrelation.h"

#define ESCAPE 27

using namespace cv;
using namespace std;

// take an image in BGR colour space and apply the MeanShift segmentation agorithm
// return the segmented image in BGR colur space
//Mat mean_shift_segmentation_application(Mat input_image) {
//	
//	Mat output_image = input_image;
//
//	// Convert color from RGB to Lab
//	cvtColor(output_image, output_image, CV_HSV2BGR);
//	cvtColor(output_image, output_image, CV_BGR2Lab);
//
//	// Initilize Mean Shift with spatial bandwith and color bandwith
//	MeanShift MSProc(8, 16);
//
//	// Filtering Process
//	MSProc.MSFiltering(output_image);
//
//	// Segmentation Process include Filtering Process (Region Growing)
//	MSProc.MSSegmentation(output_image);
//	
//	// Print the bandwith
//	cout << "the Spatial Bandwith is " << MSProc.hs << endl;
//	cout << "the Color Bandwith is " << MSProc.hr << endl;
//
//	// Convert color from Lab to RGB
//	cvtColor(output_image, output_image, CV_Lab2BGR);
//	cvtColor(output_image, output_image, CV_BGR2HSV);
//
//	return output_image;
//}

// this function takes an image as inputs and apply some morphological transformations
// in this specific case we work with dilation and erosion
Mat mophological_transformation_application(Mat input_image) {
	Mat output_image = input_image;
	Mat kernel;

	// Create kernel of size 3x3. Try different sizes
	kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	//kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
//	cout << "kernel fro morphological transformations: " << endl << kernel << endl;

	// Morphologic transformations
	// in order to completely delete noises we should apply many times erosion and dilation
	// doing this we are supposing to work in a certain environment where the red ball will
	// not be farer than a certain distance
	
	

	// deleting noise inside
	int noise_inside_reduction_iteration = 3;
	for (int i = 0; i < noise_inside_reduction_iteration; i++) {
		dilate(output_image, output_image, kernel);
	}
	for (int i = 0; i < noise_inside_reduction_iteration; i++) {
		erode(output_image, output_image, kernel);
	}

	// deleting noise outside
	int noise_outside_reduction_iteration = 3;
	for (int i = 0; i < noise_outside_reduction_iteration; i++) {
		erode(output_image, output_image, kernel);
	}
	for (int i = 0; i < noise_outside_reduction_iteration; i++) {
		dilate(output_image, output_image, kernel);
	}

	return output_image;
}

bool circle_detection(vector<Point> contour, Point centre, Mat stats, int* max_radious) {
//	cout << "centre received: " << centre << endl;
//	cout << "stats received: " << stats << endl;
//	cout << "contour received: " << contour << endl;

	Mat circle_received(1000, 750, CV_8UC3, Scalar(0, 0, 0));
	Scalar color(rand() & 255, rand() & 255, rand() & 255);
	vector<vector<Point>> contours;
	contours.push_back(contour);
	drawContours(circle_received, contours, 0, color);
	circle(circle_received, centre, 3, color, -1, 8);

//	namedWindow("circle received", CV_WINDOW_AUTOSIZE);
//	imshow("circle received", circle_received);
	waitKey(2000);

	// DETECTING CIRCUMFERENCE
	double radious = 0;
	double radious_sum = 0;
	// Signature
	vector<double> circle_reference;
	vector<double> object_signature;
	for (int i = 0; i < contour.size(); i++) {
		double distance = sqrt(pow((centre.x - contour[i].x), 2) + pow((centre.y - contour[i].y), 2));

		if (distance > *max_radious) {
			*max_radious = distance;
		}
		radious_sum = radious_sum + distance;

		circle_reference.push_back(10);
		object_signature.push_back(distance);
//		cout << "signature: " << object_signature[i] << endl;
	}

	double radious_avarage = radious_sum / contour.size();





	// mean absolute error
	double error_threshold1 = 0.075;
	
	int error_points1 = 0;
	float ratio_threshold1 = 0.4;

	double error_term1 = 0;
	for (int i = 0; i < object_signature.size(); i++) {

		error_term1 = abs((radious_avarage - object_signature[i])/radious_avarage);
//		cout << "error_term1: " << error_term1 << endl;
		if (error_term1 > error_threshold1) {
			error_points1++;
		}
	}
//	cout << "error_points1: " << error_points1 << endl;

	float ratio1 = (float)error_points1 / (float)object_signature.size();
	cout << "ratio1: " << ratio1 << endl;

	if (ratio1 > ratio_threshold1) {
		cout << "ratio: " << ratio1 << " didn't pass as too many error points" << endl;
		return false;
	}

	// mean square error
/*	double error_threshold2 = 0.005625;

	int error_points2 = 0;
	float ratio_threshold2 = 0.3;

	double error_term2 = 0;
	for (int i = 0; i < object_signature.size(); i++) {

		error_term2 = pow(abs((radious_avarage - object_signature[i])/ radious_avarage), 2);
		cout << "error_term2: " << error_term2 << endl;

		if (error_term2 > error_threshold2) {
		error_points2++;
		}
	}
	cout << "error_points2: " << error_points2 << endl;

	float ratio2 = (float)error_points2 / (float)object_signature.size();
	cout << "ratio2: " << ratio2 << endl;

	if (ratio2 > ratio_threshold2) {
		cout << "ratio: " << ratio2 << " didn't pass as too many error points" << endl;
		return false;
	} */



	// Radius too short -> we consider it as a noise object
	// we will not be able to recognise very small circles
	double radious_threshold = 15;
//	cout << "my radious: " << radious_avarage << endl;
	if (radious_avarage <= radious_threshold) {
		cout << "centre: " << centre << "didn't pass radious lenght" << endl;
		return false;
	}

	// DETECTING CIRCLE
	// Area
	double area_proportion = 0.6;
	double my_area = contourArea(contour);
	double area_expected = pow(*max_radious,2) * 3.14159265358979323846;
//	cout << "my area: " << my_area << endl;
//	cout << "area expected: " << area_expected << endl;
	cout << "my_area / area_expected: " << my_area / area_expected << endl;

	if (my_area / area_expected < area_proportion) {
		cout << "centre: " << centre << "didn't pass area proportion" << endl;
		return false;
	}

//	cout << "centre: " << centre << "will be printed" << endl;

	return true;
}

Mat object_detecting_labelling(Mat input_image, Mat to_label_image) {
	Mat output_image = input_image;

	// Labelling objects
//	cout << "-> labelling objects" << endl;

	int number_of_labels;
	Mat labels, stats, centroids;
	int connectivity = 8;

	number_of_labels = connectedComponentsWithStats(output_image, labels, stats, centroids, connectivity);

//	cout << "number_of_labels: " << number_of_labels << endl;
//	cout << "stats:" << endl << stats << endl;
//	cout << "centroids:" << endl << centroids << endl;

	// Find all closed contours in the binary image.
//	cout << "-> finding contours" << endl;

	vector<vector<Point>> contours;
	findContours(output_image, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
//	cout << "Number of countours"  << contours.size() << endl;

	// Find all circles in the image
//	cout << "-> finding circles" << endl;

	RNG rng;

	for (int i = 1; i < number_of_labels; i++) {
		// check shape
		int x_coordinate = (int)centroids.at<double>(i, 0);
		int y_coordinate = (int)centroids.at<double>(i, 1);
		Point centre = Point(x_coordinate, y_coordinate);
		vector<Point> contour = contours[contours.size()-i];
//		cout << "centro: " << labels.at<int>(centre) << endl;
//		cout << "index: " << contours.size() - i << endl;
//		cout << "contorno: " << labels.at<int>(contour[0]);
		int radious = 0;
		if (labels.at<int>(centre) != labels.at<int>(contour[0])) {
			cout << "ERRORE CENTRO E CONTORNO NON CORRISPONDONO" << endl;
//			cout << "centro nel punto: " << centre << endl;
//			waitKey(0);
//			exit(-1);
		}
		else if (circle_detection(contour, centre, stats.row(i), &radious)) {
			cout << "-> labelling circle" << endl;
			// bound object
			Scalar color(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			circle(to_label_image, centre, radious, color, 2);
		}
	}

	return to_label_image;
}

/// take an image and apply some filters in order to have better processing results
/// both input and output images are in BGR colour space
Mat image_pre_filtering(Mat input_image) {
	Mat output_image = input_image;

	// reduce dimension
//	cout << "-> resizing image" << endl;

	Size size(1000, 750);
//	resize(output_image, output_image, size);//resize image
//	namedWindow("cropped_img", CV_WINDOW_AUTOSIZE);
//	imshow("cropped_img", output_image);

	//  reduction of the camera's noise
//	cout << "-> reducing noise" << endl;
//	blur(output_image, output_image, Size(3,3));
//	medianBlur(output_image, output_image, 3);

	int sigma_x = 10;
	int sigma_y = 10;
	GaussianBlur(output_image, output_image, Size(3, 3), sigma_x, sigma_y);
//	namedWindow("noise_reducted", CV_WINDOW_AUTOSIZE);
//	imshow("noise_reducted", output_image);
	

	// segmentation - application of meanshift algorithm
	/*cout << "-> mean shift segmentation" << endl;

	output_image = mean_shift_segmentation_application(output_image);
	namedWindow("mean shift segmentation");
	imshow("mean shift segmentation", output_image);*/

	return output_image;
}

/// take an image and select only objects of the colour in the range we are interested in
/// the input image must be in GBR colour space
/// the output image will be returned as GBR colour space
Mat red_color_filtering(Mat input_image) {

	Mat output_image;

	// convert from RGB to HSV
	cvtColor(input_image, output_image, COLOR_BGR2HSV);
//	cout << "numero canali immagine HSV: " << output_image.channels();

	// range of red colors
//	cout << "-> colour range operation" << endl;

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

/// this function take as input an image and an image we want to print detected circles
/// it reduces the noise with morphological transformations and finally
/// it cheks if there are circles, bounding them eventually
Mat image_circle_recognition(Mat input_image, Mat output_image) {
	Mat circle_detecting_image = input_image;

	// application of morphological transformation
	// chiusura - apertura
//	cout << "-> application of erosion and dilation" << endl;

	circle_detecting_image = mophological_transformation_application(circle_detecting_image.clone());
	namedWindow("morphological transformation");
	imshow("morphological transformation", circle_detecting_image);
//	imwrite("C:/Users/lucac_000/Desktop/morphological.jpg", circle_detecting_image);

	// objects detectios and labelling
//	cout << "-> labelling objects" << endl;

	circle_detecting_image = object_detecting_labelling(circle_detecting_image.clone(), output_image);
//	namedWindow("circle recognized");
//	imshow("circle recognized", circle_detecting_image);

	return output_image;
}

/// this functions defines the different steps of processing in order to achieve our goal
Mat operation_sequence(Mat input_image) {
	Mat  filtering_result, color_result, shape_result;

	// PRE-FILTERING
//	cout << "pre filtering..." << endl;
	filtering_result = image_pre_filtering(input_image.clone());
	if (!filtering_result.data) {
		cout << "errore prefiltering" << endl;
		getchar();
		exit(-1);
	}
//	namedWindow("filtering_result", CV_WINDOW_AUTOSIZE);
//	imshow("filtering_result", filtering_result);

	// RECOGNITION OF RED OBJECTS
//	cout << "red objects detection..." << endl;
	color_result = red_color_filtering(filtering_result.clone());
	if (!color_result.data) {
		cout << "errore lettura color detection" << endl;
		getchar();
		exit(-1);
	}
//	namedWindow("color_result", CV_WINDOW_AUTOSIZE);
//	imshow("color_result", color_result);
//	imwrite("C:/Users/lucac_000/Desktop/color.jpg", color_result);

	// RECOGNITION OF CIRCLE SHAPED OBJECTS
//	cout << "circle object recognition..." << endl;
	shape_result = image_circle_recognition(color_result.clone(), filtering_result.clone());
	if (!shape_result.data) {
		cout << "errore lettura shape detection" << endl;
		getchar();
		exit(-1);
	}
//	namedWindow("shape_result", CV_WINDOW_AUTOSIZE);
//	imshow("shape_result", shape_result);

	return shape_result;
}

int main(int argc, char* argv[]) {
	cout << "inizio programma" << endl;

	// declaration image containers
	Mat img_input, img_output;

	// TESTS FROM IMAGES
//	img_input = imread("C:/Users/lucac_000/source/repos/RedBallRecognising/ImagesDataset/my_dataset2.jpg", CV_LOAD_IMAGE_COLOR);
//	// check reading result
//	if (!img_input.data) {
//		cout << "errore lettura immagine" << endl;
//		getchar();
//		return -1;
//	}
////	namedWindow("image", CV_WINDOW_AUTOSIZE);
////	imshow("image", img_input);
////	imwrite("C:/Users/lucac_000/Desktop/input.jpg", img_input);
//
//	img_output = operation_sequence(img_input);
//
//	namedWindow("img_output", CV_WINDOW_AUTOSIZE);
//	imshow("img_output", img_output);
////	imwrite("C:/Users/lucac_000/Desktop/output.jpg", img_output);
//	waitKey(0);
//	destroyAllWindows();


	// TESTS FROM CAMERA
//	namedWindow("image input", CV_WINDOW_AUTOSIZE);
	namedWindow("image output", CV_WINDOW_AUTOSIZE);
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		cout << "error in VideoCapture, check device!" << endl;
	}
	// keyboard pressed
	char keypressed = 0;
	// check the success for image reading
	bool success;
	while (keypressed != ESCAPE) {
		success = capture.read(img_input);
		if (success == false) {
			cout << "cannot read the frame from file" << endl;
			return 1;
		}

//		imshow("image input", img_input);

		img_output = operation_sequence(img_input);
		imshow("image output", img_output);

		keypressed = waitKey(2);
	}

	cout << "fine programma" << endl;
	waitKey(0);
}