// Mehdi and Rayane

#include <cstddef>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "dataset.hpp"
#include "knn.hpp"

int main() {
    auto data = loadDataset("data/SharvitB2", 12);
    std::cout << data.size() << " samples loaded\n";

    if (data.empty()) {std::cerr << "no samples found!!!\n"; return 1; } // check if the dataset is empty and print an error message

    const std::map<std::string, std::size_t> expectedDimensions = {
        {"ART", 36},
        {"GFD", 100},
        {"SA", 20},
        {"Yang", 29},
        {"Zernike7", 18}
    };

    bool valid = true;

    // check every sample
    for (const auto& sample : data) {
        for (const auto& [method, expected] : expectedDimensions) {
            auto it = sample.features.find(method);

            if (it == sample.features.end()) {
                std::cerr << sample.name << ": missing " << method << "\n";
                valid = false;
            } else if (it->second.size() != expected) {
                std::cerr << sample.name << ": " << method
                          << " has " << it->second.size()
                          << " values; expected " << expected << "\n";
                valid = false;
            }
        }
    }

    if (!valid) return 1;

    std::cout << "all samples have the expected descriptor dimensions\n";

    if (data.size() < 6) {
        std::cerr << "leave-one-out evaluation with k = 5 requires at least six samples\n";
        return 1;
    }

    try {
        std::cout << "leave-one-out evaluation: k = 1, 3, 5\n";

        for (const auto& method : methodNames()) {
            for (int k : {1, 3, 5}) {
                std::size_t correct = 0;

                for (std::size_t i = 0; i < data.size(); ++i) {
                    //hold out sample i; all other samples form the training set.
                    std::vector<Sample> training;
                    for (std::size_t j = 0; j < data.size(); ++j) {
                        if (j != i) training.push_back(data[j]);
                    }

                    int prediction = predictKNN(training, data[i].features.at(method), method, k);
                    if (prediction == data[i].label) ++correct;
                }

                double rate = 100.0 * correct / data.size();
                std::cout << method << " | k: " << k
                          << " | correct: " << correct << "/" << data.size()
                          << " | recognition rate: " << rate << "%\n";
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "prediction failed: " << error.what() << "\n";
        return 1;
    }
}
