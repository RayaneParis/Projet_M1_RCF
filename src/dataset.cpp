// rayane

#include "dataset.hpp"

#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

const std::vector<std::string>& methodNames() {
    static const std::vector<std::string> names = {"ART", "GFD", "SA", "Yang", "Zernike7"};
    return names;
}

static std::vector<double> readVector(const fs::path& file) {
    std::ifstream input(file);
    if (!input) {
        throw std::runtime_error("cannot open " + file.string());
    }
    std::vector<double> values;
    while (true) {
        input >> std::ws;
        if (input.eof()) break;

        double number;
        if (!(input >> number) || !std::isfinite(number)) {
            throw std::runtime_error("invalid number in " + file.string());
        }
        values.push_back(number);
    }
    if (values.empty()) {
        throw std::runtime_error("invalid descriptor " + file.string());
    }
    return values;
}

static bool validName(const std::string& name) {
    if (name.size() != 7) return false;
    if (name[0] != 's' || name[3] != 'n') return false;

    for (int i = 0; i < 7; ++i) {
        if (i != 0 && i != 3) {
            if (name[i] < '0' || name[i] > '9') return false;
        }
    }
    return true;
}

std::vector<Sample> loadDataset(const std::string& root, int maxClasses) {
    const std::map<std::string, std::size_t> dimensions = {
        {"ART", 36}, {"GFD", 100}, {"SA", 20}, {"Yang", 29}, {"Zernike7", 18}};
    std::map<std::string, Sample> byName;

    // try the normal layout first; otherwise use '=' for every dataset folder
    fs::path datasetRoot = root;
    std::string prefix;
    if (!fs::is_directory(datasetRoot)) {
        prefix = "=";
        datasetRoot = datasetRoot.parent_path() / (prefix + datasetRoot.filename().string());
    }

    for (const auto& method : methodNames()) {
        fs::path dir = datasetRoot / (prefix + "Signatures") / (prefix + method);
        if (!fs::is_directory(dir)) {
            throw std::runtime_error("directory not found: " + dir.string());
        }
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (!entry.is_regular_file()) continue;

            std::string name = entry.path().filename().string().substr(0, 7);
            for (char& c : name) {
                c = std::tolower(static_cast<unsigned char>(c));
            }
            if (!validName(name)) continue;

            int label = std::stoi(name.substr(1, 2));
            if (label > maxClasses) continue;

            Sample& sample = byName[name];
            sample.name = name;
            sample.label = label;
            sample.index = std::stoi(name.substr(4, 3));
            sample.features[method] = readVector(entry.path());
        }
    }

    std::vector<Sample> samples;
    for (const auto& entry : byName) {
        const Sample& sample = entry.second;
        for (const auto& method : methodNames()) {
            auto feature = sample.features.find(method);
            if (feature == sample.features.end()) {
                throw std::runtime_error(sample.name + ": missing descriptor " + method);
            }
            if (feature->second.size() != dimensions.at(method)) {
                throw std::runtime_error(
                    sample.name + ": " + method + " has " + std::to_string(feature->second.size()) +
                    " values; expected " + std::to_string(dimensions.at(method)));
            }
        }
        samples.push_back(sample);
    }
    if (samples.empty()) {
        throw std::runtime_error("no samples found in " + datasetRoot.string());
    }

    return samples;
}
