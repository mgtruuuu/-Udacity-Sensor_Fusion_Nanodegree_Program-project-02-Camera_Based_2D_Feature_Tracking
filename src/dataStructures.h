#ifndef dataStructures_h
#define dataStructures_h

#include <opencv2/core.hpp>

#include <iostream>
#include <string_view>
#include <vector>





struct Result {
	double et_KPsAndDESCs;
	int num_KPs;
	int num_matchs;
	float mean_KPSize;
	float std_KPSize;
};



// Represents the available sensor information at the same time instance.
class DataFrame {
private:
    cv::Mat cameraImg;                      // camera image
    std::vector<cv::KeyPoint> keypoints;    // 2D keypoints within camera image
    cv::Mat descriptors;                    // keypoint descriptors
    std::vector<cv::DMatch> kptMatches;     // keypoint matches between previous and current frame
    Result result;


public:
	const cv::Mat& getCameraImg() const { return cameraImg; }
	const std::vector<cv::KeyPoint>& getKeypoints() const { return keypoints; }
	const cv::Mat& getDescriptors() const { return descriptors; }
	const std::vector<cv::DMatch>& getKPMatches() const  { return kptMatches; }
	const Result& getResult() const { return result; }

	void setCameraImg(cv::Mat cameraImg) { this->cameraImg = cameraImg; }
	void setKeypoints(std::vector<cv::KeyPoint> keypoints) { this->keypoints = keypoints; }
	void setDescriptors(cv::Mat descriptors) { this->descriptors = descriptors; }
	void setKPMatches(std::vector<cv::DMatch> kptMatches) { this->kptMatches = kptMatches; }
	void setResult(Result result) { this->result = result; }

	


	void setResult_et_KPsAndDESCs(double et) { result.et_KPsAndDESCs = et; }

	void setResult_num_KPs() { result.num_KPs = static_cast<int>(keypoints.size()); }

	void setResult_num_matchs() { result.num_matchs = static_cast<int>(kptMatches.size()); }

	void setResult_mean_KPSize() {
		float mean_KPsize{ .0f };
		for (const auto& kp : keypoints)
			mean_KPsize += kp.size;

		mean_KPsize /= keypoints.size();

		result.mean_KPSize = mean_KPsize;
	}

	void setResult_std_KPSize() {
		float std_KPSize{ .0f };

		for (const auto& kp : keypoints)
			std_KPSize += (kp.size - this->getResult().mean_KPSize) 
			* (kp.size - this->getResult().mean_KPSize);

		std_KPSize /= keypoints.size();

		std_KPSize = sqrt(std_KPSize);

		result.std_KPSize = std_KPSize;
	}

	void printResult() {
		std::cout << "2D Feature tracking result about the image : \n";
		std::cout << '\t' << "# Keypoints : " << result.num_KPs << '\n';
		std::cout << '\t' << "Mean of the size of Keypoints : " << result.mean_KPSize << '\n';
		std::cout << '\t' << "Standard Deviation of the size of Keypoints : " << result.std_KPSize << '\n';
		std::cout << '\t' << "Estimated time to get KPs and DESCs : " << result.et_KPsAndDESCs << " ms" << '\n';
		std::cout << '\t' << "# Matches (compared to the image ahead) : " << result.num_matchs << '\n';
		std::cout << "----------------------------------------------------------------\n";
	}

};







#endif /* dataStructures_h */
