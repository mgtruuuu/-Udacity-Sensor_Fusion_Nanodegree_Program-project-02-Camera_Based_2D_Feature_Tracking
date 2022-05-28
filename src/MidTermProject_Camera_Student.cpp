/* INCLUDES FOR THIS PROJECT */

#include <cmath>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "options.h"
#include "matching2D.hpp"





/* MAIN PROGRAM */
int main(int argc, const char* argv[]) {



    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    const std::string dataPath{ "../" };

    // camera
    const std::string imgBasePath{ dataPath + "images/" };
    const std::string imgPrefix{ "KITTI/2011_09_26/image_00/data/000000" }; // left camera, color
    const std::string imgFileType{ ".png" };
    constexpr int imgStartIndex{ 0 };   // first file index to load (Assumes Lidar and camera names have identical naming convention.)
    constexpr int imgEndIndex{ 9 };     // last file index to load
    constexpr int imgFillWidth{ 4 };    // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    constexpr int dataBufferSize{ 2 };      // # images which are held in memory (ring buffer) at the same time
    std::deque<DataFrame> dataBuffer;       // list of data frames which are held in memory at the same time
    std::vector<Result> results;


    //
    //// Visualization options
    //
    constexpr bool bVisKPs{ true };
    constexpr bool bVisMatches{ false };

    //
    //// detection/description option
    //
    constexpr Detector detectorType{ Detector::SIFT };      // SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    constexpr Descriptor descriptorType{ Descriptor::SIFT };// FREAK, BRIEF, BRISK, ORB, AKAZE, SIFT
    constexpr Matcher matcherType{ Matcher::MAT_BF };       // MAT_BF, MAT_FLANN
    constexpr DescriptorOption descriptorOptionType{ DescriptorOption::DES_HOG };    // DES_BINARY, DES_HOG
    constexpr Selector selectorType{ Selector::SEL_KNN };   // SEL_NN, SEL_KNN


    std::cout << "Detector   type : " << static_cast<std::string>(getDetector(detectorType)) << '\n';
    std::cout << "Descriptor type : " << static_cast<std::string>(getDescriptor(descriptorType)) << '\n';
    std::cout << "START!\n";


    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex{ 0 }; imgIndex <= imgEndIndex - imgStartIndex; ++imgIndex) {


        /* LOAD IMAGE INTO BUFFER */

        // Assemble filenames for current index.
        std::ostringstream imgNumber;
        imgNumber << std::setfill('0') << std::setw(imgFillWidth) << imgStartIndex + imgIndex;
        const std::string imgFullFilename{ imgBasePath + imgPrefix + imgNumber.str() + imgFileType };

        // Load image from file and convert to grayscale.
        const cv::Mat img{ cv::imread(imgFullFilename) };
        cv::Mat imgGray;
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);


        //// TASK MP.1: Set up the loading procedure for the images using Ring data buffer.

        DataFrame frame;
        frame.setCameraImg(imgGray);
        if (dataBuffer.size() > dataBufferSize)
            dataBuffer.pop_front();

        dataBuffer.push_back(frame);

        std::cout << "#1 : LOAD IMAGE INTO BUFFER done" << std::endl;

        ////





        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        std::vector<cv::DMatch> matches;     // applied from the second image ...

        double elapsedTime{ -1.0 };

        //// TASK MP.2 -> Add the following keypoint detectors in file matching2D.cpp 
        ////              and enable enum-based selection based on detectorType
        //// TASK MP.3 -> Only keep keypoints on the preceding vehicle.
        //// TASK MP.4 -> Add the following descriptors in file matching2D.cpp 
        ////              and enable string-based selection based on descriptorType.


        /* DETECT IMAGE KEYPOINTS */
        /* EXTRACT KEYPOINT DESCRIPTORS */

        const auto it_curr{ dataBuffer.end() - 1 };

        getKeypointsAndDescriptors(detectorType, descriptorType, it_curr->getCameraImg(),
            elapsedTime, keypoints, descriptors);

        it_curr->setKeypoints(keypoints);
        it_curr->setDescriptors(descriptors);

        it_curr->setResult_num_KPs();
        it_curr->setResult_mean_KPSize();
        it_curr->setResult_std_KPSize();
        it_curr->setResult_et_KPsAndDESCs(elapsedTime);

        std::cout << "#2 : DETECT KEYPOINTS done" << std::endl;
        std::cout << "#3 : EXTRACT DESCRIPTORS done." << std::endl;

        ////
        ////
        ////



        /* MATCH KEYPOINT DESCRIPTORS */


        // Wait until at least two images have been processed.
        if (dataBuffer.size() <= 1) {

            it_curr->setResult_num_matchs();

            it_curr->printResult();

            results.push_back(it_curr->getResult());

            // Visualize keypoints.
            visualizeKeypoints(it_curr->getCameraImg(), it_curr->getKeypoints(), bVisKPs);

            continue;
        }


        //// TASK MP.5 -> Add FLANN matching in file matching2D.cpp.
        //// TASK MP.6 -> Add KNN match selection and perform descriptor distance ratio 
        ////              filtering with t=0.8 in file matching2D.cpp.


        const auto it_prev{ it_curr - 1 };
        constexpr bool crossCheck{ false };
        matchDescriptors(it_prev->getDescriptors(), it_curr->getDescriptors(),
            matcherType, descriptorOptionType, selectorType, crossCheck, matches);

        it_curr->setKPMatches(matches);
        it_curr->setResult_num_matchs();

        std::cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << std::endl;

        ////
        ////


        it_curr->printResult();

        results.push_back(it_curr->getResult());


        // Visualize keypoints.
        visualizeKeypoints(it_curr->getCameraImg(), it_curr->getKeypoints(), bVisKPs);

        // Visualize matches between current and previous image.
        visualizeMatches(it_prev->getCameraImg(), it_curr->getCameraImg(),
            it_prev->getKeypoints(), it_curr->getKeypoints(),
            matches, bVisMatches);

    }   // eof loop over all images


    printTable(detectorType, descriptorType, results);

    //if (writeRecordToFile("benchmark.txt", detectorType, descriptorType, results))
    //    assert(false, "Check the fcn. writeRecordToFile()\n");



    return 0;
}


