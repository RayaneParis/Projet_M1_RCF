// rayane

#pragma once

#include <string>
#include <vector>

#include "dataset.hpp"

struct KMeansResult {
    std::vector<int> assignment;  // cluster assignment for each sample
    std::vector<std::vector<double>> centroids;
    double sse = 0.0;  // sum of squared errors
    int iterators = 0;  // number of iterations
};

KMeansResult kmeans(const std::vector<Sample>& data, const std::string& method, int K,
                    unsigned seed, int maxIter = 100);
