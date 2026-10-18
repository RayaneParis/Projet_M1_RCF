#pragma once 
#include <map>
#include <string>
#include <vector>

struct Sample {
    std::string name;
    int label;                                                  // class label
    int index;                                                  // index of the sample in the dataset
    std::map<std::string, std::vector<double>> features;        // method -> vector of features
};

// Get the names of all methods in the dataset
const std::vector<std::string>& methodNames();

// Load dataset 
std::vector<Sample> loadDataset(const std::string& root, int maxClasses = 18);