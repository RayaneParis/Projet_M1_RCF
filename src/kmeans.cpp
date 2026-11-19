// rayane and mehdi

#include "kmeans.hpp"

#include <algorithm>
#include <numeric>
#include <random>
#include <stdexcept>

#include "distance.hpp"

KMeansResult kmeans(const std::vector<Sample>& data, const std::string& method, int K,
                    unsigned seed, int maxIter) {
    if (data.empty() || K < 1 || static_cast<std::size_t>(K) > data.size()) {
        throw std::invalid_argument("k-means requires data and a valid number of clusters");
    }
    if (maxIter < 1) {
        throw std::invalid_argument("k-means requires at least one iteration");
    }

    const std::size_t n = data.size();
    const std::size_t dim = data[0].features.at(method).size();
    if (dim == 0) {
        throw std::invalid_argument("k-means requires nonempty descriptors");
    }

    // shuffle sample indices so the same seed gives the same initial centers
    std::vector<std::size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(indices.begin(), indices.end(), rng);

    KMeansResult result;
    result.assignment.assign(n, -1);
    for (int c = 0; c < K; ++c) {
        result.centroids.push_back(data[indices[c]].features.at(method));
    }

    for (int it = 0; it < maxIter; ++it) {
        result.iterators = it + 1;
        bool changed = false;

        // assign each sample to its nearest center
        for (std::size_t i = 0; i < n; ++i) {
            const auto& x = data[i].features.at(method);
            int closest = 0;
            double bestDistance = euclideanDistance(x, result.centroids[0]);

            for (int c = 1; c < K; ++c) {
                double distance = euclideanDistance(x, result.centroids[c]);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    closest = c;
                }
            }

            if (result.assignment[i] != closest) {
                result.assignment[i] = closest;
                changed = true;
            }
        }

        if (!changed) break;

        // sum the feature values in each cluster
        std::vector<std::vector<double>> sum(K, std::vector<double>(dim, 0.0));
        std::vector<int> count(K, 0);

        for (std::size_t i = 0; i < n; ++i) {
            const auto& x = data[i].features.at(method);
            int c = result.assignment[i];
            for (std::size_t d = 0; d < dim; ++d) {
                sum[c][d] += x[d];
            }
            ++count[c];
        }

        // use the mean as the new center; keep empty clusters unchanged
        for (int c = 0; c < K; ++c) {
            if (count[c] == 0) continue;
            for (std::size_t d = 0; d < dim; ++d) {
                result.centroids[c][d] = sum[c][d] / count[c];
            }
        }
    }

    // add the squared distances to the assigned centers
    for (std::size_t i = 0; i < n; ++i) {
        double distance =
            euclideanDistance(data[i].features.at(method), result.centroids[result.assignment[i]]);
        result.sse += distance * distance;
    }
    return result;
}
