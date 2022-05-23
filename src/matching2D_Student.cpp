#include <string>
#include <numeric>
#include "matching2D.hpp"
#include "enums.h"




std::string_view getDetector(Detector detectorType) {
    switch (detectorType) {
    case Detector::SHITOMASI:   return "SHITOMASI";
    case Detector::HARRIS:      return "HARRIS";
    case Detector::FAST:        return "FAST";
    case Detector::BRISK:       return "BRISK";
    case Detector::ORB:         return "ORB";
    case Detector::AKAZE:       return "AKAZE";
    case Detector::SIFT:        return "SIFT";
    default:                    assert(false, "Wrong Detector type\n");
    }
}

std::string_view getMatcher(Matcher matcherType) {
    switch (matcherType) {
    case Matcher::MAT_BF:       return "MAT_BF";
    case Matcher::MAT_FLANN:    return "MAT_FLANN";
    default:                    assert(false, "Wrong Matcher type\n");
    }
}

std::string_view getSelector(Selector selectorType) {
    switch (selectorType) {
    case Selector::SEL_NN:      return "SEL_NN";
    case Selector::SEL_KNN:     return "SEL_KNN";       // for k=2 only
    default:                    assert(false, "Wrong Selector type\n");
    }
}

std::string_view getDescriptor(Descriptor descriptorType) {
    switch (descriptorType) {
    case Descriptor::BRIEF:     return "BRIEF";
    case Descriptor::FREAK:     return "FREAK";
    case Descriptor::BRISK:     return "BRISK";
    case Descriptor::ORB:       return "ORB";
    case Descriptor::AKAZE:     return "AKAZE";
    case Descriptor::SIFT:      return "SIFT";
    default:                    assert(false, "Wrong Descriptor type\n");
    }
}

std::string_view getDescriptorOption(DescriptorOption descriptorOptionType) {
    switch (descriptorOptionType) {
    case DescriptorOption::DES_BINARY:  return "DES_BINARY";
    case DescriptorOption::DES_HOG:     return "DES_HOG";
    default:                            assert(false, "Wrong DescriptorOption type\n");
    }
}







// Remove keypoints outside of the vehicleRect.
void selectKeypointsOnVeh(const bool bFocusOnVehicle, std::vector<cv::KeyPoint>& keypoints) {
    
    if (bFocusOnVehicle == false)       return;


    // car 2D-image box ...
    const cv::Rect& vehicleRect{ 535, 180, 180, 150 };

    std::vector<cv::KeyPoint> vehKPs;
    
    for (const auto& keypoint : keypoints)
        if (vehicleRect.contains(keypoint.pt))
            vehKPs.push_back(keypoint);

    keypoints = vehKPs;
}



void getKeypointsAndDescriptors(
    const Detector detectorType, const Descriptor descriptorType, const cv::Mat& imgGray,
    double& elapsedTime, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {

    const int64 t1{ cv::getTickCount() };

    detectKeypoints(detectorType, imgGray, keypoints);

    const int64 t2{ cv::getTickCount() };


    // Remove keypoints outside of the vehicleRect.
    const cv::Rect& vehicleRect{ 535, 180, 180, 150 };
    auto isKPOutOfBox{ [&vehicleRect](const cv::KeyPoint& kp)-> bool {
            return !vehicleRect.contains(kp.pt);
        }
    };
    keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), isKPOutOfBox), keypoints.end());

    
    const int64 t3{ cv::getTickCount() };

    computeDescriptors(detectorType, descriptorType, imgGray, keypoints, descriptors);
    
    const int64 t4{ cv::getTickCount() };
    elapsedTime = static_cast<double>(t4 - t3 + t2 - t1) * 1000.0 / cv::getTickFrequency();
}





void detectKeypoints(const Detector detectorType, const cv::Mat& imgGray,
    std::vector<cv::KeyPoint>& keypoints) {

    cv::Ptr<cv::FeatureDetector> detector;
    switch (detectorType) {
    case Detector::SHITOMASI:   detKeypointsShiTomasi(imgGray, keypoints);  break;

    case Detector::HARRIS:      detKeypointsHarris(imgGray, keypoints);     break;

    case Detector::FAST:        detKeypointsFAST(imgGray, keypoints);       break;


    case Detector::BRISK:       detector = cv::BRISK::create();
        detector->detect(imgGray, keypoints);       break;

    case Detector::ORB:         detector = cv::ORB::create();
        detector->detect(imgGray, keypoints);       break;

    case Detector::AKAZE:       detector = cv::AKAZE::create();
        detector->detect(imgGray, keypoints);       break;

    case Detector::SIFT:        detector = cv::SIFT::create();
        detector->detect(imgGray, keypoints);       break;

    default:                    assert(false, "Wrong Detector type!\n");
    }
}




void computeDescriptors(const Detector detectorType, const Descriptor descriptorType, const cv::Mat& imgGray,
    std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {

    cv::Ptr<cv::DescriptorExtractor> extractor;

    switch (descriptorType) {

    case Descriptor::BRIEF:     extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();    break;
    case Descriptor::FREAK:     extractor = cv::xfeatures2d::FREAK::create();                       break;

    case Descriptor::BRISK:     extractor = cv::BRISK::create();    break;
    case Descriptor::ORB:       extractor = cv::ORB::create();      break;
    case Descriptor::AKAZE:     extractor = cv::AKAZE::create();    break;
    case Descriptor::SIFT:      extractor = cv::SIFT::create();     break;

    default:                    assert(false, "Wrong Descriptor type!\n");
    }

    //const cv::InputArray& mask{ cv::noArray() };
    //constexpr bool useProvidedKeypoints{ true };   // Detect keypoints and compute descriptor in two stages.
    //extractor->detectAndCompute(imgGray, mask, keypoints, descriptors, useProvidedKeypoints);
    extractor->compute(imgGray, keypoints, descriptors);
}







// Detect keypoints in image using the TRADITIONAL Shi-Thomasi detector.
void detKeypointsShiTomasi(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints) {

    // Shi-Tomasi detector

    constexpr int blockSize{ 4 };       // size of an average block for computing a derivative covariation matrix 
                                        // over each pixel neighborhood
    constexpr double maxOverlap{ 0.0 }; // maximun permissible overlap between two features in %
    constexpr double minDistance{ (1.0 - maxOverlap) * blockSize };
    const int maxCorners{ static_cast<int>(img.rows * img.cols / std::max(1.0, minDistance)) }; // max. num. of keypoints;

    constexpr double qualityLevel{ 0.01 };  // minimal accepted quality of image corners
    constexpr bool useHarrisDetector{ false };
    constexpr double k{ 0.04 };


    // Apply corner detection
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance,
        cv::Mat{}, blockSize, useHarrisDetector, k);


    // Add corners to result vector.

    for (const auto& corner : corners) {
        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f{ corner.x, corner.y };
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
}


void detKeypointsHarris(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints) {

    // detector parameters
    
    constexpr int blockSize{ 2 };       // For every pixel, a blockSize � blockSize neighborhood is considered.
    constexpr int apertureSize{ 3 };    // aperture parameter for Sobel operator (must be odd)
    constexpr int minResponse{ 100 };   // minimum value for a corner in the 8bit scaled response matrix
    constexpr double k{ 0.04 };         // Harris parameter (see equation for details)


    // Detect Harris corners and normalize output

    cv::Mat dst{ cv::Mat::zeros(img.size(), CV_32FC1) };
    cv::Mat dst_norm;
    cv::Mat dst_norm_scaled;

    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    normalize(dst, dst_norm, .0, 255.0, cv::NORM_MINMAX, CV_32FC1, cv::Mat{});
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);


    // Apply non-maximum suppression (NMS)

    constexpr float maxOverlap{ 0.0f };     // maximum permissible overlap between two features in %, 
                                            // used during non-maxima suppression
    for (int j{ 0 }; j < dst_norm.rows; ++j) {
        for (int i{ 0 }; i < dst_norm.cols; ++i) {
            const float response{ dst_norm.at<float>(j, i) };

            // Apply the minimum threshold for Harris cornerness response.

            if (response < minResponse)     continue;

            // Create a tentative new keypoint otherwise.
            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt = static_cast<cv::Point2f>(cv::Point2i{ i, j });
            newKeyPoint.size = static_cast<float>(2 * apertureSize);
            newKeyPoint.response = response;


            // Perform non-maximum suppression (NMS) in local neighbourhood around the new keypoint.

            bool bOverlap{ false };     

            // Loop over all existing keypoints.
            for (auto& keypoint : keypoints) {
                const float kptOverlap{ cv::KeyPoint::overlap(newKeyPoint, keypoint) };

                // Test if overlap exceeds the maximum percentage allowable.
                if (kptOverlap > maxOverlap) {
                    bOverlap = true;

                    // If overlapping, test if new response is the local maximum.
                    if (newKeyPoint.response > keypoint.response) {
                        keypoint = newKeyPoint;     // Replace the old keypoint.
                        break;                      // Exit for loop.
                    }
                }
            }

            // If above response threshold and not overlapping any other keypoint,
            // add to keypoints list.
            if (!bOverlap)  keypoints.push_back(newKeyPoint);
        }
    }
}


void detKeypointsFAST(const cv::Mat& img, std::vector<cv::KeyPoint>& keypoints) {

    constexpr int threshold{ 100 };
    constexpr bool nonmaxSuppresion{ true };
    constexpr cv::FastFeatureDetector::DetectorType type{ cv::FastFeatureDetector::TYPE_9_16 };

    cv::FAST(img, keypoints, threshold, nonmaxSuppresion, type);
}




// Find best matches for keypoints in two camera images based on several matching methods.
void matchDescriptors(const cv::Mat& descSource, const cv::Mat& descRef,
    const Matcher matcherType, const DescriptorOption descriptorOptionType,
    const Selector selectorType, const bool crossCheck, std::vector<cv::DMatch>& matches) {

    cv::Ptr<cv::DescriptorMatcher> matcher;     // configure matcher

    if (matcherType == Matcher::MAT_BF) {

        // for BRISK, BRIEF, ORB, FREAK and AKAZE descriptors
        if (descriptorOptionType == DescriptorOption::DES_BINARY) {
            const int normType{ cv::NORM_HAMMING };
            matcher = cv::BFMatcher::create(normType, crossCheck);
        }

        // for SIFT descriptor
        else if (descriptorOptionType == DescriptorOption::DES_HOG) {
            const int normType{ cv::NORM_L2 };
            matcher = cv::BFMatcher::create(normType, crossCheck);
        }
    }
    else if (matcherType == Matcher::MAT_FLANN) {
        matcher = cv::FlannBasedMatcher::create();
    }


    // Perform matching task.

    if (selectorType == Selector::SEL_NN) {                 // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches);       // Finds the best match for each descriptor in desc1.
    }
    else if (selectorType == Selector::SEL_KNN) {           // k nearest neighbors (k=2)
        assert(crossCheck == false, "The 8th argument of the function matchDescriptors() in main() must be 'false' in order to choose the SEL_KNN Selector Type.\n");
        constexpr int k{ 2 };
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, k);

        constexpr float minDescDistRatio{ 0.8f };
        for (const auto& knn_match : knn_matches)
            if (knn_match[0].distance < minDescDistRatio * knn_match[1].distance)
                matches.push_back(knn_match[0]);
        std::cout << '\t' << knn_matches.size() - matches.size() << " keypoints were removed (K-Nearest-Neighbor approach)." << std::endl;
    }
}




void visualizeKeypoints(const cv::Mat& img, const std::vector<cv::KeyPoint>& keypoints, const bool bVis) {
    
    if (bVis == false)      return;


    cv::Mat matchImg{ img.clone() };
    const cv::Rect& vehicleRect{ 535, 180, 180, 150 };      // the same const variable in fcn selectKeypointsOnVeh()
    cv::rectangle(matchImg, vehicleRect, cv::Scalar{ 255.0, 255.0, 255.0 }, 1, 20, 0);
    cv::Mat img_kps;
    cv::drawKeypoints(matchImg, keypoints, img_kps);

    std::string windowName{ "Matching keypoints between two camera images" };
    cv::namedWindow(windowName, 7);
    cv::imshow(windowName, img_kps);
    cv::waitKey(0);
}

void visualizeMatches(const cv::Mat& imgFront, const cv::Mat& imgBack, 
    const std::vector<cv::KeyPoint>& keypointsFront, const std::vector<cv::KeyPoint>& keypointsBack,
    const std::vector<cv::DMatch>& matches, const bool bVis) {

    if (bVis == false)      return;


    cv::Mat matchImg{ imgBack.clone() };
    cv::drawMatches(
        imgFront, keypointsFront, imgBack, keypointsBack, matches, matchImg,
        cv::Scalar::all(-1), cv::Scalar::all(-1),
        std::vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    std::string windowName{ "Matching keypoints between two camera images" };
    cv::namedWindow(windowName, 7);
    cv::imshow(windowName, matchImg);
    std::cout << "Press key to continue to next image" << std::endl;
    cv::waitKey(0);     // wait for key to be pressed
}


//
void printTable(Detector detectorType, Descriptor descriptorType, const std::vector<Result>& results) {

    std::cout << "Detector" << '\t'
        << "Descriptor" << '\t'
        << "ET_KPsAndDESCs(ms)" << '\t'
        << "#KPs" << '\t'
        << "Mean_KPSize" << '\t'
        << "STD_KPSize" << '\t'
        << "#Matches" << '\n';

    for (const auto& result : results) {
        std::cout << static_cast<std::string>(getDetector(detectorType)) << '\t' << '\t'
            << static_cast<std::string>(getDescriptor(descriptorType)) << '\t' << '\t'
            << result.et_KPsAndDESCs << '\t' << '\t' << '\t'
            << result.num_KPs << '\t'
            << result.mean_KPSize << '\t' << '\t'
            << result.std_KPSize << '\t' << '\t'
            << result.num_matchs << '\n';
    }

    std::cout << "================================================================================" << std::endl;
}


bool writeRecordToFile(std::string file_name,
    Detector detectorType, Descriptor descriptorType, std::vector<Result> results) {

    std::ofstream file;
    file.open(file_name, std::ios_base::app);
    for (const auto& result : results) {
        file << static_cast<std::string>(getDetector(detectorType)) << '\t'
            << static_cast<std::string>(getDescriptor(descriptorType)) << '\t'
            << result.et_KPsAndDESCs << '\t'
            << result.num_KPs << '\t' 
            << result.mean_KPSize << '\t' 
            << result.std_KPSize << '\t' 
            << result.num_matchs << '\n';
    }
    file.close();

    return true;
}
