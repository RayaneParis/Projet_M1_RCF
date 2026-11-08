// Mehdi and Rayane
#include <algorithm>
#include <cstddef>
#include <exception>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include "dataset.hpp"
#include "knn.hpp"
#include "kmeans.hpp"
#include "vote.hpp"

static void printSection(const std::string& title) {
    std::cout << "\n======================= " << title << " =======================\n";
}

int main() {
    // Load the dataset
    auto data = loadDataset("data/SharvitB2", 6);
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
    // 2.a. k-NN classification (leave-one-out)
    try {
        printSection("2.a. k-NN classification (leave-one-out)");

        for (const auto& method : methodNames()) {
            for (DistanceType type : {DistanceType::Euclidean, DistanceType::Manhattan}) {
                for (int k : {1, 3, 5, 7, 9}) {
                    std::size_t correct = 0;

                    for (std::size_t i = 0; i < data.size(); ++i) {
                        //hold out sample i; all other samples form the training set.
                        std::vector<Sample> training;
                        for (std::size_t j = 0; j < data.size(); ++j) {
                            if (j != i) training.push_back(data[j]);
                        }

                        int prediction = predictKNN(training, data[i].features.at(method), method, k, type);
                        if (prediction == data[i].label) ++correct;
                    }

                    double rate = 100.0 * correct / data.size();
                    std::cout << method 
                            << " | " << (type == DistanceType::Euclidean ? "L2" : "L1")
                            << " | k: " << k
                            << " | correct: " << correct << "/" << data.size()
                            << " | recognition rate: " << rate << "%\n";
                }
            }
        }
    } catch (const std::exception& error) {
        std::cerr << "prediction failed: " << error.what() << "\n";
        return 1;
    }

    // 2.b. K-means clustering 
    try {
        printSection("2.b. K-means clustering (K = 6, best SSE over 20 initializations)");

        for (const auto& method : methodNames()) {
            KMeansResult best = kmeans(data, method, 6, 0);
            for (unsigned seed = 1; seed < 20; ++seed) {
                KMeansResult r = kmeans(data, method, 6, seed);
                if (r.sse < best.sse) best = r;
            }
            std::cout << method
                      << " | SSE: " <<best.sse
                      << " | iterators: " << best.iterators
                      << " | recognition rate: " << clusterRecognitionRate(data, best, 6) << "%\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "k-means failed: " <<error.what() << "\n";
        return 1;
    }

    try {
        // Test 1 
        printSection("2.b. K-means clustering stability (K = 6, 20 seeds)");
        
        for (const auto& method : methodNames()) {
            double minRate = 100.0, maxRate = 0.0, sumRate = 0.0;
            double minSse = 1e300, maxSse = 0.0;
            for (unsigned seed = 0; seed < 20; ++seed) {
                KMeansResult r = kmeans(data, method, 6, seed);
                double rate = clusterRecognitionRate(data, r, 6);
                minRate = std::min(minRate, rate);
                maxRate = std::max(maxRate, rate);
                sumRate += rate;
                minSse = std::min(minSse, r.sse);
                maxSse = std::max(maxSse, r.sse);
            }
            std::cout << method
                     << " | rate min/mean/max: " << minRate << " / " << sumRate / 20
                     << " / " << maxRate
                     << " | SSE min/max: " << minSse << " / " << maxSse << "\n";
        } 
        // Test 2
        KMeansResult one = kmeans(data, "ART", 1, 0);
        std::cout << "K = 1 -> " << clusterRecognitionRate(data, one, 1)
                                 << "% (should be 16.6667%)\n";
        KMeansResult all = kmeans(data, "ART", 72, 0);
        std::cout << "K = 72 -> " << clusterRecognitionRate(data, all, 72)
                                 << "% (should be 100%)\n";

    } catch (const std::exception& error) {
        std::cerr << "k-means stability test failed: " << error.what() << "\n";
        return 1;
    }
    // 2.c. Majority vote 
    try {
        printSection("2.c. Majority vote (leave-one-out)");

        const std::vector<std::string> priority = {"ART", "GFD", "Zernike7", "Yang", "SA"};

        for (DistanceType type : {DistanceType::Euclidean, DistanceType::Manhattan})
            for (int k : {1, 3, 5, 7, 9})
                std::cout << "vote | " 
                          << (type == DistanceType::Euclidean ? "L2" : "L1")
                          << " | k: " << k
                          << " | recognition rate: "
                          << leaveOneOutVote(data, priority, k, type) << "%\n";
    } catch (const std::exception& error) {
        std::cerr << "majority vote failed: " << error.what() << "\n";
        return 1;
    }
}
