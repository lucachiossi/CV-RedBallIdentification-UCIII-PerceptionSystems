#include "opencv\cv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
	cout << "inizio programma" << endl;

	// declaration image container
	Mat img17;

	// read image
	img17 = imread("C:/Users/lucac_000/source/repos/RedBallRecognising/ImagesDataset/redball17.jpg");

	// check reading result
	if (!img17.data) {
		cout << "errore lettura immagine mandril.jpg" << endl;
		getchar();
		return -1;
	}

	// recognition of red objects


	// recognition of ball shape objects


	// creation of a window to show the image
	namedWindow("image17", CV_WINDOW_AUTOSIZE);

	// show image in the window
	imshow("image17", img17);
	waitKey(0);
	destroyWindow("original");

	cout << "fine programma" << endl;
	getchar();
}