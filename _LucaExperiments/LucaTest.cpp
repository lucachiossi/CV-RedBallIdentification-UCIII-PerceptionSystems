#include "opencv\cv.hpp"
#include <iostream>
#include "LucaFunctions.cpp"

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
	cout << "inizio programma" << endl;

	// declaration image container
	Mat img, filtering_result, color_result, shape_result;

	// read image
	img = imread("C:/Users/lucac_000/source/repos/RedBallRecognising/ImagesDataset/redball18.jpg");

	// check reading result
	if (!img.data) {
		cout << "errore lettura immagine mandril.jpg" << endl;
		getchar();
		return -1;
	}

	// PRE-FILTERING


	// RECOGNITION OF RED OBJECTS
	color_result = red_color_filtering(img);

	// RECOGNITION OF CIRCLE SHAPED OBJECTS


	// creation of a window to show the image
	namedWindow("image", CV_WINDOW_AUTOSIZE);

	// show image in the window
	imshow("image", img);
	waitKey(0);
	destroyWindow("original");

	cout << "fine programma" << endl;
	getchar();
}