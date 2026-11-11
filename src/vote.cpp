// Rayane vote majoritaire
// refactored and commented by Mehdi

// majority voting decision: each descriptor has already predicted a class using knn.
// the votes map stores how many times each class was predicted.
// then we choose the class with the highest number of votes and if two classes have the same number
// of votes, we keep the class that appears first in predictions.

#include "vote.hpp"

#include <map>
#include <stdexcept>

#include "knn.hpp"

int predictVote(const std::vector<Sample>& training, const Sample& query,
                const std::vector<std::string>& methods, int k, DistanceType type) {
    if (methods.empty()) throw std::invalid_argument("no voting methods");

    std::vector<int> predictions;
    std::map<int, int> votes;
    for (const std::string& method : methods) {
        int label = predictKNN(training, query.features.at(method), method, k, type);
        predictions.push_back(label);
        ++votes[label];
    }

    // keep the earlier method's prediction when vote counts are equal
    int winner = predictions.front();
    for (int label : predictions) {
        if (votes[label] > votes[winner]) {
            winner = label;
        }
    }
    return winner;
}

double leaveOneOutVote(const std::vector<Sample>& data, const std::vector<std::string>& methods,
                       int k, DistanceType type) {
    if (data.size() < 2) {
        throw std::invalid_argument("leave-one-out requires at least two samples");
    }

    std::size_t correct = 0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        // remove only the held-out sample and preserve the training order
        std::vector<Sample> training = data;
        training.erase(training.begin() + i);

        if (predictVote(training, data[i], methods, k, type) == data[i].label) ++correct;
    }
    return 100.0 * correct / data.size();
}
