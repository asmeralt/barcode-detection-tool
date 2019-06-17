#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "GradMagnitudeBarcodeDetector.h"

void drawRotatedRect(cv::Mat& img, cv::RotatedRect rotRect, const cv::Scalar& color, int thickness=1) {
	cv::Point2f contour2f [4];
	rotRect.points(contour2f);
	std::vector<cv::Point> contour(4);
	for (int i = 0; i < 4; ++i) {
		contour[i] = contour2f[i];
	}
	cv::polylines(img, contour, true, color, thickness);
}


int main(int argc, char** argv) {
	if (argc < 2)
		return -1;

	cv::namedWindow("BarcodeDetection", cv::WINDOW_AUTOSIZE);
	cv::VideoCapture video(argv[1]);
	cv::Mat frame, grayFrame, gradMag;
	cv::Mat display;
	video >> frame;
	while (true) {
		video >> frame;
		if (frame.empty() || !video.isOpened()) {
			break;
		}
		cv::cvtColor(frame, grayFrame, cv::COLOR_BGR2GRAY);

		findGradMagnitude(grayFrame, gradMag);
		const auto rotRect = detectBarcode(grayFrame);
		drawRotatedRect(frame, rotRect, cv::Scalar(175, 15, 175), 5);

		cv::cvtColor(gradMag, gradMag, cv::COLOR_GRAY2BGR);
		cv::vconcat(frame, gradMag, display);
		cv::resize(display, display, cv::Size(), 0.25, 0.25);
		cv::imshow("BarcodeDetection", display);

		if (cv::waitKey(40) == 27) {
			break;
		}
	}
	video.release();
	return 0;
}
