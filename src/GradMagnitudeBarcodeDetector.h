#include <opencv2/core.hpp>

void findGradMagnitude(const cv::Mat& src, cv::Mat& dst);

cv::RotatedRect detectBarcode(const cv::Mat& gradMag);
