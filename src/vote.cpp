// Rayane vote majoritaire
// refactored and commented by Mehdi

// majority voting decision: each descriptor has already predicted a class using knn.
// the votes map stores how many times each class was predicted.
// then we choose the class with the highest number of votes and if two classes have the same number
// of votes, we keep the class that appears first in predictions.

#include "vote.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

#include "knn.hpp"

// obtain the five predictions and count them, then choose the winner -> predicts the class of one
// image
int predictVote(const std::vector<Sample>& training, const Sample& query,
                const std::vector<std::string>& methods, int k, DistanceType type) {
    if (methods.empty()) throw std::invalid_argument("no voting methods");

    std::vector<int> predictions;
    std::map<int, int> votes;
    for (const std::string& method : methods) {
        int label = predictKNN(training, query.features.at(method), method, k, type);
        predictions.push_back(label);
        votes[label]++;
    }

    int winner = predictions[0];
    int mostVotes = 0;
    for (int label : predictions) {
        if (votes[label] > mostVotes) {
            mostVotes = votes[label];
            winner = label;
        }
    }
    return winner;
}

// leaveOneOut evaluates how accurately predictVote() works on the whole dataset
double leaveOneOutVote(const std::vector<Sample>& data, const std::vector<std::string>& methods,
                       int k, DistanceType type) {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < data.size(); ++i) {
        std::vector<Sample> training;
        for (std::size_t j = 0; j < data.size(); ++j)
            if (j != i) training.push_back(data[j]);

        if (predictVote(training, data[i], methods, k, type) == data[i].label) ++correct;
    }
    return 100.0 * correct / data.size();
}