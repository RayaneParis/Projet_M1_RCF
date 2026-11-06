// mehdi - knn

#include "knn.hpp"
#include "distance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <utility>

int predictKNN(const std::vector<Sample>& training,
               const std::vector<double>& query,
               const std::string& method,
               int k) {
    if (training.empty() || query.empty()) {
        throw std::invalid_argument("k-NN requires training samples and a nonempty query");
    }
    if (k < 1 || static_cast<std::size_t>(k) > training.size()) {
        throw std::invalid_argument("k must be between 1 and the number of training samples");
    }

    // store the distance and class label of each sample
    std::vector<std::pair<double, int>> neighbors;

    for (const auto& sample : training) {
        auto it = sample.features.find(method);
        if (it == sample.features.end()) {
            throw std::invalid_argument(sample.name + ": missing descriptor " + method);
        }

        double distance = euclideanDistance(query, it->second);
        if (!std::isfinite(distance)) {
            throw std::invalid_argument("k-NN requires finite distances");
        }
        neighbors.push_back({distance, sample.label});
    }

    // sort from closest to farthest and keep the original order if distances are equal
    std::stable_sort(neighbors.begin(), neighbors.end(),
                     [](const auto& a, const auto& b) { return a.first < b.first; });

    std::map<int, int> votes;
    for (int i = 0; i < k; ++i) {
        ++votes[neighbors[i].second];
    }

    // choose the class with the most votes
    // if votes are equal, choose the class with the closest sample
    int predictedLabel = neighbors.front().second;
    int bestVotes = 0;
    for (int i = 0; i < k; ++i) {
        int label = neighbors[i].second;
        if (votes[label] > bestVotes) {
            bestVotes = votes[label];
            predictedLabel = label;
        }
    }
    return predictedLabel;
}
