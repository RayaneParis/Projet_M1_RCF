// mehdi & rayane

#include <algorithm>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dataset.hpp"
#include "kmeans.hpp"
#include "knn.hpp"
#include "svm.hpp"
#include "vote.hpp"

const std::vector<int> neighborCounts = {1, 3, 5, 7, 9};

static void printSection(const std::string& title) {
    std::cout << "\n======================= " << title << " =======================\n";
}

static void runKNN(const std::vector<Sample>& data) {
    printSection("2.a. k-NN classification (leave-one-out)");

    for (const auto& method : methodNames()) {
        for (DistanceType type : {DistanceType::Euclidean, DistanceType::Manhattan}) {
            for (int k : neighborCounts) {
                std::size_t correct = 0;
                for (std::size_t i = 0; i < data.size(); ++i) {
                    // remove only the held-out sample and preserve the training order
                    std::vector<Sample> training = data;
                    training.erase(training.begin() + i);

                    int prediction =
                        predictKNN(training, data[i].features.at(method), method, k, type);
                    if (prediction == data[i].label) ++correct;
                }

                double rate = 100.0 * correct / data.size();
                std::cout << method << " | " << (type == DistanceType::Euclidean ? "L2" : "L1")
                          << " | k: " << k << " | correct: " << correct << "/" << data.size()
                          << " | recognition rate: " << rate << "%\n";
            }
        }
    }
}

static void runKMeans(const std::vector<Sample>& data) {
    const int clusterCount = 6;
    const unsigned seedCount = 20;
    printSection("2.b. K-means clustering (K = 6, 20 initializations)");

    for (const auto& method : methodNames()) {
        KMeansResult best = kmeans(data, method, clusterCount, 0);
        double minRate = clusterRecognitionRate(data, best, clusterCount);
        double maxRate = minRate;
        double sumRate = minRate;
        double maxSse = best.sse;

        // use the same runs for the best result and the stability summary
        for (unsigned seed = 1; seed < seedCount; ++seed) {
            KMeansResult result = kmeans(data, method, clusterCount, seed);
            double rate = clusterRecognitionRate(data, result, clusterCount);
            minRate = std::min(minRate, rate);
            maxRate = std::max(maxRate, rate);
            sumRate += rate;
            maxSse = std::max(maxSse, result.sse);
            if (result.sse < best.sse) best = result;
        }

        std::cout << method << " | SSE: " << best.sse << " | iterators: " << best.iterators
                  << " | recognition rate: " << clusterRecognitionRate(data, best, clusterCount)
                  << "%\n";
        std::cout << method << " | rate min/mean/max: " << minRate << " / " << sumRate / seedCount
                  << " / " << maxRate << " | SSE min/max: " << best.sse << " / " << maxSse << "\n";
    }

    KMeansResult one = kmeans(data, "ART", 1, 0);
    std::cout << "K = 1 -> " << clusterRecognitionRate(data, one, 1)
              << "% (should be 16.6667%)\n";
    KMeansResult all = kmeans(data, "ART", 72, 0);
    std::cout << "K = 72 -> " << clusterRecognitionRate(data, all, 72)
              << "% (should be 100%)\n";
}

static void runVoting(const std::vector<Sample>& data) {
    printSection("2.c. Majority vote (leave-one-out)");
    const std::vector<std::string> priority = {"ART", "GFD", "Zernike7", "Yang", "SA"};

    for (DistanceType type : {DistanceType::Euclidean, DistanceType::Manhattan}) {
        for (int k : neighborCounts) {
            double rate = leaveOneOutVote(data, priority, k, type);
            std::cout << "vote | " << (type == DistanceType::Euclidean ? "L2" : "L1")
                      << " | k: " << k << " | recognition rate: " << rate << "%\n";
        }
    }
}

static void runSVM(const std::vector<Sample>& data) {
    printSection("2.d. SVM classification (leave-one-out)");

    for (const auto& method : methodNames()) {
        std::size_t correct = 0;
        for (std::size_t i = 0; i < data.size(); ++i) {
            // train on all samples except the one being tested
            std::vector<Sample> training = data;
            training.erase(training.begin() + i);

            int prediction = predictSVM(training, data[i].features.at(method), method);
            if (prediction == data[i].label) ++correct;
        }

        double rate = 100.0 * correct / data.size();
        std::cout << method << " | SVM RBF | correct: " << correct << "/" << data.size()
                  << " | recognition rate: " << rate << "%\n";
    }
}

int main() {
    try {
        std::vector<Sample> data = loadDataset("data/SharvitB2", 6);
        std::cout << data.size() << " samples loaded\n";
        std::cout << "all samples have the expected descriptor dimensions\n";

        if (data.size() <= static_cast<std::size_t>(neighborCounts.back())) {
            throw std::runtime_error("leave-one-out with k = 9 requires at least ten samples");
        }

        runKNN(data);
        runKMeans(data);
        runVoting(data);
        runSVM(data);
    } catch (const std::exception& error) {
        std::cerr << "experiment failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
