// rayane
#include "dataset.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;

// Get the names of all methods in the dataset
const std::vector<std::string>& methodNames() {
    static const std::vector<std::string> names = {"ART", "GFD", "SA", "Yang", "Zernike7"};
    return names;
}

// read vector
static std::vector<double> readVector(const fs::path& file) {
    std::ifstream input(file);
    if (!input) {
        throw std::runtime_error("cannot open " + file.string());
    }
    std::vector<double> values;
    double number;
    while (input >> number) {
        if (!std::isfinite(number)) {
            throw std::runtime_error("invalid number in " + file.string());
        }
        values.push_back(number);
    }
    if (!input.eof() || values.empty()) {
        throw std::runtime_error("invalid descriptor " + file.string());
    }
    return values;
}

// check if a sample name is valid
static bool validName(const std::string& name) {
    if (name.size() != 7) return false;
    if (name[0] != 's' || name[3] != 'n') return false;

    for (int i = 0; i < 7; i++) {
        if (i != 0 && i != 3) {
            if (name[i] < '0' || name[i] > '9') return false;
        }
    }
    return true;
}

// Load dataset from a directory
std::vector<Sample> loadDataset(const std::string& root, int maxClasses) {
    std::map<std::string, Sample> byName;  // map from sample name to Sample object

    // try the normal layout first; otherwise use '=' for every dataset folder
    fs::path datasetRoot = root;
    std::string prefix;
    if (!fs::is_directory(datasetRoot)) {
        prefix = "=";
        datasetRoot = datasetRoot.parent_path() / (prefix + datasetRoot.filename().string());
    }

    for (const auto& method : methodNames()) {
        fs::path dir = datasetRoot / (prefix + "Signatures") / (prefix + method);
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

    // check for missing features
    for (const auto& s : samples)
        for (const auto& m : methodNames())
            if (!s.features.count(m) || s.features.at(m).empty())
                std::cerr << "Warning: " << s.name << " without descriptor " << m << "\n";

    return samples;
}
