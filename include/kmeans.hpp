// Rayane K-means
#pragma once
#include <string>
#include <vector>
#include "dataset.hpp"

struct KMeansResult
{
    std::vector<int> assignment; // cluster assignment for each sample
    std::vector<std::vector<double>> centroids;
    double sse = 0.0; // sum of squared errors
    int iterators = 0;
};

KMeansResult kmeans(const std::vector<Sample> &data, const std::string &method,
                    int k, unsigned seed, int maxIter = 100);

double clusterRecognitionRate(const std::vector<Sample> &data, const KMeansResult &result, int K);