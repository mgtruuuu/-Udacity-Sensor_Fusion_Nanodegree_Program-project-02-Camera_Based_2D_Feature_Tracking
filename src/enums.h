#ifndef ENUMS_H_
#define ENUMS_H_

#include <string_view>



enum class Detector {
    SHITOMASI,          // corner detection
    HARRIS,             // corner detection
    FAST,               // corner detection

    BRISK,
    ORB,
    AKAZE,
    SIFT,
};

enum class Matcher {
    MAT_BF,
    MAT_FLANN,
};

enum class Selector {
    SEL_NN,
    SEL_KNN             // for k=2 only in this project
};

enum class Descriptor {
    BRIEF,
    FREAK,

    BRISK,
    ORB,
    AKAZE,
    SIFT,
};

enum class DescriptorOption {
    DES_BINARY,
    DES_HOG,
};


std::string_view getDetector(Detector detectorType);
std::string_view getMatcher(Matcher matcherType);
std::string_view getSelector(Selector selectorType);
std::string_view getDescriptor(Descriptor descriptorType);
std::string_view getDescriptorOption(DescriptorOption descriptorOptionType);


#endif