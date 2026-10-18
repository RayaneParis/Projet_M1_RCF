#include "dataset.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace fs = std::filesystem;

// Get the names of all methods in the dataset
const std::vector<std::string>& methodNames() {
    static const std::vector<std::string> names = {"ART", "GFD", "SA", "Yang", "Zernike7"};
    return names;
}

// Read a vector of doubles from a text file
static std::vector<double> readVector(const fs::path& file) {
    std::ifstream in(file);
    std::vector<double> v;
    double x;
    while (in >> x) v.push_back(x);
    return v;
}

// Check if a sample name is valid 
static bool validName(const std::string& s) {
    if (s.size() != 7 || s[0] != 's' || s[3] != 'n') return false;
    for (int i : {1, 2, 4, 5, 6})
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    return true;
}

// Load dataset from a directory
std::vector<Sample> loadDataset(const std::string& root, int maxClasses) {
    std::map<std::string, Sample> byName; // map from sample name to Sample object

    for (const auto& method : methodNames()) {
        fs::path dir = fs::path(root) / "Signatures" / method;
        if (!fs::exists(dir)) {
            std::cerr << "Warning: directory not found: " << dir << "\n";
            continue;
        }   
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;

            std::string stem = entry.path().filename().string().substr(0, 7);
            std::transform(stem.begin(), stem.end(), stem.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (!validName(stem)) continue;

            int label = std::stoi(stem.substr(1, 2));
            if (label > maxClasses) continue;

            Sample& s = byName[stem];                           
            s.name = stem;                      
            s.label = label;
            s.index = std::stoi(stem.substr(4, 3));
            s.features[method] = readVector(entry.path());
        }
    }

    std::vector<Sample> samples;
    for (auto& [name, s] : byName) samples.push_back(std::move(s));

    // Check for missing features
    for (const auto& s : samples) 
        for (const auto& m : methodNames())
            if (!s.features.count(m) || s.features.at(m).empty())
                std::cerr << "Warning: " << s.name << " without descriptor " << m << "\n";

    return samples;
}

