#include "GradMagnitudeBarcodeDetector.h"

#include<opencv2/imgproc.hpp>

void findGradMagnitude(const cv::Mat& src, cv::Mat& dst) {
	cv::Laplacian(src, dst, CV_32F, 3);
	cv::convertScaleAbs(dst, dst);

	cv::threshold(dst, dst, 0, 255, cv::THRESH_TOZERO + cv::THRESH_OTSU);
	cv::morphologyEx(dst, dst, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(src.cols/30,1)));
	cv::morphologyEx(dst, dst, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, src.rows/50)), cv::Point(-1,-1), 4);
}

cv::RotatedRect detect(const cv::Mat& gradMag) {
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(gradMag, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);
	if (contours.empty()) {
		return cv::RotatedRect();
	}
	std::vector<double> contoursArea(contours.size());
	std::transform(contours.begin(), contours.end(), contoursArea.begin(), [](std::vector<cv::Point>& contour) {
		return cv::contourArea(contour);
	});
	int maxIdx = std::distance(contoursArea.begin(), std::max_element(contoursArea.begin(), contoursArea.end()));
	return cv::minAreaRect(contours[maxIdx]);
}

cv::RotatedRect detectBarcode(const cv::Mat& img) {
	cv::Mat gradMag;
	findGradMagnitude(img, gradMag);
	return detect(gradMag);
}
