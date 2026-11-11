// rayane

#pragma once
#include <map>
#include <string>
#include <vector>

struct Sample
{
    std::string name;
    int label;                                           // Class label
    int index;                                           // Index of the sample in the dataset
    std::map<std::string, std::vector<double>> features; // method -> vector of features
};

// Get the names of all methods in the dataset
const std::vector<std::string> &methodNames();

// Load dataset
std::vector<Sample> loadDataset(const std::string &root, int maxClasses = 18);