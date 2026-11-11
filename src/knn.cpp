// mehdi - knn

#include "knn.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <utility>

static bool closer(const std::pair<double, int>& a, const std::pair<double, int>& b) {
    return a.first < b.first;
}

int predictKNN(const std::vector<Sample>& training, const std::vector<double>& query,
               const std::string& method, int k, DistanceType type) {
    if (training.empty() || query.empty()) {
        throw std::invalid_argument("k-NN requires training samples and a nonempty query");
    }
    if (k < 1 || static_cast<std::size_t>(k) > training.size()) {
        throw std::invalid_argument("k must be between 1 and the number of training samples");
    }

    // store the distance and class label of each sample
    std::vector<std::pair<double, int>> neighbors;
    neighbors.reserve(training.size());

    for (const auto& sample : training) {
        auto feature = sample.features.find(method);
        if (feature == sample.features.end()) {
            throw std::invalid_argument(sample.name + ": missing descriptor " + method);
        }

        double distance = computeDistance(query, feature->second, type);
        if (!std::isfinite(distance)) {
            throw std::invalid_argument("k-NN requires finite distances");
        }
        neighbors.push_back({distance, sample.label});
    }

    // preserve training order when distances are equal
    std::stable_sort(neighbors.begin(), neighbors.end(), closer);

    std::map<int, int> votes;
    for (int i = 0; i < k; ++i) {
        ++votes[neighbors[i].second];
    }

    // the first tied class is the one with the closest neighbor
    int winner = neighbors.front().second;
    for (int i = 0; i < k; ++i) {
        int label = neighbors[i].second;
        if (votes[label] > votes[winner]) {
            winner = label;
        }
    }
    return winner;
}
