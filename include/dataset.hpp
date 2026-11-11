// rayane

#pragma once

#include <map>
#include <string>
#include <vector>

struct Sample {
    std::string name;
    int label;
    int index;  // sample number within its class
    std::map<std::string, std::vector<double>> features;
};

const std::vector<std::string>& methodNames();

// reject empty data, missing descriptors and incorrect dimensions
std::vector<Sample> loadDataset(const std::string& root, int maxClasses = 18);
