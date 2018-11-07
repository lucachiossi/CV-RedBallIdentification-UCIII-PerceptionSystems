#include "opencv\cv.hpp"
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[]) {
	cout << "inizio programma" << endl;

	// dichiarazione matrice
	Mat img;

	// lettura immagine
	img = imread("C:/Users/lucac_000/Desktop/Luca/UNIVERSITA/__MAGISTRALE__/_II_anno/M1_Sistemas_de_Percepcion/_Laboratory/Lab1/mandril.jpg");

	// controllo esito lettura
	if (!img.data) {
		cout << "errore lettura immagine mandril.jpg" << endl;
		getchar();
		return -1;
	}

	// creazione finestra per mostrare la immagine
	namedWindow("original", CV_WINDOW_AUTOSIZE);

	// mostrare immagine nella finestra creata
	imshow("original", img);
	waitKey(0);
	destroyWindow("original");

	cout << "fine programma" << endl;
	getchar();
}