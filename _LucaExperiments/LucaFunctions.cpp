#include "opencv\cv.hpp"
#include <iostream>

using namespace std;
using namespace cv;

// PRE-FILTERING


// RECOGNITION OF RED OBJECTS

/*
This function take an image and gives us back another image containing all the
red objects contained in the previous one in greyscale format
Input: an image as Mat format
Output: an image as Mat format
*/

Mat red_color_filtering(Mat input_image) {
	
	Mat output_image;

	// convert from RGB to HSV
	cvtColor(input_image, output_image, COLOR_BGR2HSV);
	cout << "numero canali immagine HSV: " << output_image.channels();

	// range of red colors
	/*int lower_red = (110, 50, 50);
	int upper_red = (130, 255, 255);

	mask = inRange(output_image, lower_red, upper_red, output_image);*/

	return output_image;
}