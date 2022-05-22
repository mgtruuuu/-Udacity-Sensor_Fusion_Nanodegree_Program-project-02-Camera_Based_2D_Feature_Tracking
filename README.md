# Camera-based 2D Feature Tracking

## 1. Project Overview

<img src="images/keypoints.png" width="820" height="248" />


This is the mid-term project about Camera-based 2D feature tracking (to check the final project click [here](https://github.com/mgtruuuu/Udacity-Sensor_Fusion_Nanodegree_Program-project-03-Track_an_Object_in_3D_Space.git)). It covers the following key concepts:

- Ring data buffer (to save memory when loading and processing images)
- Keypoint detection (Shi-Tomasi, HARRIS, FAST, BRISK, ORB, AKAZE and SIFT)
- Descriptor extraction & matching (BRISK, BRIEF, ORB, FREAK, AKAZE and SIFT)
- Performance evaluation (Test and Compare the various algorithms in different combinations)


## 2. Key Implementation

### MP.1 Data Buffer Optimization

```c++
constexpr int dataBufferSize{ 2 };      // # images which stay in memory (ring buffer) at the same time
std::deque<DataFrame> dataBuffer;       // list of data frames which stay in memory

for (size_t imgIndex{ 0 }; imgIndex <= imgEndIndex - imgStartIndex; ++imgIndex) {

    // ...

    DataFrame frame;
    frame.setCameraImg(imgGray);
    if (dataBuffer.size() > dataBufferSize)
        dataBuffer.pop_front();

    dataBuffer.push_back(frame);

    // ...
}
```


### MP.2,3,4 Keypoint Detection/Removal/Descriptors

```c++
getKeypointsAndDescriptors(detectorType, descriptorType, it_curr->getCameraImg(), elapsedTime, keypoints, descriptors);
```

```c++
void getKeypointsAndDescriptors(
    const Detector detectorType, const Descriptor descriptorType, const cv::Mat& imgGray, double& elapsedTime, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {

    detectKeypoints(detectorType, imgGray, keypoints);


    // Remove keypoints outside of the vehicleRect.
    const cv::Rect& vehicleRect{ 535, 180, 180, 150 };
    auto isKPOutOfBox{ [&vehicleRect](const cv::KeyPoint& kp)-> bool {
            return !vehicleRect.contains(kp.pt);
        }
    };
    keypoints.erase(std::remove_if(keypoints.begin(), keypoints.end(), isKPOutOfBox), keypoints.end());


    computeDescriptors(detectorType, descriptorType, imgGray, keypoints, descriptors);
}
```

```c++
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
```

```c++
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

    extractor->compute(imgGray, keypoints, descriptors);
}
```

### MP.5,6 Descriptor Matching/DistanceRatio

```c++
matchDescriptors(it_prev->getDescriptors(), it_curr->getDescriptors(), matcherType, descriptorOptionType, selectorType, crossCheck, matches);
```

```c++
// Find best matches for keypoints in two camera images based on several matching methods.
void matchDescriptors(const std::vector<cv::KeyPoint>& kPtsSource, const std::vector<cv::KeyPoint>& kPtsRef, const cv::Mat& descSource, const cv::Mat& descRef, const Matcher matcherType, const DescriptorOption descriptorOptionType, const Selector selectorType, const bool crossCheck, std::vector<cv::DMatch>& matches) {
    
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
```


## 3. Performance evaluation (MP.7,8,9)

### Performance benchmark (average values of images )


 | DETECTOR |DESCRIPTOR |Elapsed Time (ms) |#Keypoints |Mean_KPSize |STD_KPSize |#Matches |
 |--- | --- | --- | --- | --- | --- | --- | 
 | SHITOMASI | FREAK | 37.91826 | 117.9 | 4 | 0 | 76.6 | 
 | SHITOMASI | BRIEF | 19.7889 | 117.9 | 4 | 0 | 94.4 | 
 | SHITOMASI | BRISK | 41.57278 | 117.9 | 4 | 0 | 76.7 | 
 | SHITOMASI | ORB | 13.68993 | 117.9 | 4 | 0 | 90.7 | 
 | SHITOMASI | SIFT | 24.07728 | 117.9 | 4 | 0 | 92.7 | 
 | HARRIS | FREAK | 36.04401 | 25.7 | 6 | 0 | 14.6 | 
 | HARRIS | BRIEF | 12.19149 | 25.7 | 6 | 0 | 17.7 | 
 | HARRIS | BRISK | 37.69465 | 25.7 | 6 | 0 | 14.2 | 
 | HARRIS | ORB | 13.07088 | 25.7 | 6 | 0 | 16.2 | 
 | HARRIS | SIFT | 22.80195 | 25.7 | 6 | 0 | 16.4 | 
 | FAST | FREAK | 25.7206 | 41.8 | 7 | 0 | 26.7 | 
 | FAST | BRIEF | 1.11028 | 41.8 | 7 | 0 | 31.9 | 
 | FAST | BRISK | 24.54518 | 41.8 | 7 | 0 | 27.3 | 
 | FAST | ORB | 1.23865 | 41.8 | 7 | 0 | 31 | 
 | FAST | SIFT | 12.5775 | 41.8 | 7 | 0 | 29.5 | 
 | BRISK | FREAK | 81.41851 | 256.3 | 19.11561 | 9.859705 | 152.6 | 
 | BRISK | BRIEF | 57.76577 | 276.2 | 21.94223 | 14.58136 | 170.4 | 
 | BRISK | BRISK | 83.11754 | 276.2 | 21.94223 | 14.58136 | 157 | 
 | BRISK | ORB | 62.40447 | 276.2 | 21.94223 | 14.58136 | 151 | 
 | BRISK | SIFT | 80.6068 | 276.2 | 21.94223 | 14.58136 | 164.6 | 
 | ORB | FREAK | 58.65157 | 62 | 37.9248 | 7.380311 | 42.1 | 
 | ORB | BRIEF | 6.58821 | 116.1 | 56.05776 | 25.13533 | 54.5 | 
 | ORB | BRISK | 32.29423 | 107 | 51.94267 | 21.61631 | 75.1 | 
 | ORB | ORB | 9.49425 | 116.1 | 56.05776 | 25.13533 | 76.1 | 
 | ORB | SIFT | 42.06362 | 116.1 | 56.05776 | 25.13533 | 76.3 | 
 | AKAZE | BRIEF | 69.84111 | 167 | 7.693421 | 3.531132 | 126.6 | 
 | AKAZE | BRISK | 91.45076 | 167 | 7.693421 | 3.531132 | 121.5 | 
 | AKAZE | ORB | 70.13651 | 167 | 7.693421 | 3.531132 | 118.6 | 
 | AKAZE | AKAZE | 117.7821 | 167 | 7.693421 | 3.531132 | 125.9 | 
 | AKAZE | SIFT | 81.97281 | 167 | 7.693421 | 3.531132 | 127 | 
 | SIFT | FREAK | 110.0292 | 137.4 | 4.688527 | 4.686256 | 59.6 | 
 | SIFT | BRIEF | 87.61845 | 138.6 | 5.032351 | 5.945847 | 70.2 | 
 | SIFT | BRISK | 111.2881 | 138.5 | 4.99825 | 5.793 | 59.2 | 
 | SIFT | SIFT | 144.4186 | 138.6 | 5.032351 | 5.945847 | 80 | 

Check the file(`./table/benchmark_average.csv`) for more information.


### Top 3 keypoint/descriptor combinations (based on the benchmark above)

1. FAST detectors and BRIEF descriptors
2. ORB detectors and BRIEF descriptors
3. ORB detectors and ORB descriptors

The most important factor that I consider when selecting the good combination is elapsed time. These combinations are much more faster than the others except the combination FAST/ORB. Detector type ORB can detect many keypoints than FAST so FAST/ORB is excluded even though it is a little bit faster than ORB/BRIEF and ORB/ORB.



## 4. Dependencies for Running Locally

1. cmake >= 2.8

- All OSes: [click here for installation instructions](https://cmake.org/install/)


2. make >= 4.1 (Linux, Mac), 3.81 (Windows)

- Linux: make is installed by default on most Linux distros
- Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
- Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)


3. OpenCV >= 4.1

- All OSes: refer to the [official instructions](https://docs.opencv.org/master/df/d65/tutorial_table_of_content_introduction.html)
- This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors. If using [homebrew](https://brew.sh/): `$> brew install --build-from-source opencv` will install required dependencies and compile opencv with the `opencv_contrib` module by default (no need to set `-DOPENCV_ENABLE_NONFREE=ON` manually). 
- The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)


4. gcc/g++ >= 5.4

- Linux: gcc / g++ is installed by default on most Linux distros
- Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
- Windows: recommend using either [MinGW-w64](http://mingw-w64.org/doku.php/start) or [Microsoft's VCPKG, a C++ package manager](https://docs.microsoft.com/en-us/cpp/build/install-vcpkg?view=msvc-160&tabs=windows). VCPKG maintains its own binary distributions of OpenCV and many other packages. To see what packages are available, type `vcpkg search` at the command prompt. For example, once you've _VCPKG_ installed, you can install _OpenCV 4.1_ with the command:

    ```bash
    c:\vcpkg> vcpkg install opencv4[nonfree,contrib]:x64-windows
    ```

    Then, add *C:\vcpkg\installed\x64-windows\bin* and *C:\vcpkg\installed\x64-windows\debug\bin* to your user's _PATH_ variable. Also, set the _CMake Toolchain File_ to *c:\vcpkg\scripts\buildsystems\vcpkg.cmake*.



## 5. Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.
