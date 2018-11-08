#include "opencv\cv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
	cout << "inizio programma" << endl;

	// declaration matrix
	// dichiarazione matrice
	Mat img;

	// read image
	// lettura immagine
	img = imread("C:/Users/lucac_000/Desktop/Luca/UNIVERSITA/__MAGISTRALE__/_II_anno/M1_Sistemas_de_Percepcion/_Laboratory/Lab1/mandril.jpg");

	// check reading result
	// controllo esito lettura
	if (!img.data) {
		cout << "errore lettura immagine mandril.jpg" << endl;
		getchar();
		return -1;
	}

	// create a window to show the image
	// creazione finestra per mostrare la immagine
	namedWindow("original", CV_WINDOW_AUTOSIZE);

	// show image in the window
	// mostrare immagine nella finestra creata
	imshow("original", img);
	waitKey(0);
	destroyWindow("original");

	cout << "fine programma" << endl;
	getchar();
}