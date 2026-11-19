// Rayane vote majoritaire
// refactored and commented by Mehdi

#include "vote.hpp"

#include <map>
#include <stdexcept>

int majorityVote(const std::vector<int>& predictions) {
    if (predictions.empty()) throw std::invalid_argument("majority vote requires at least one label");

    std::map<int, int> votes;
    for (int label : predictions) ++votes[label];

    // Input order breaks ties for both neighbors and descriptor predictions.
    int winner = predictions.front();
    for (int label : predictions) {
        if (votes[label] > votes[winner]) {
            winner = label;
        }
    }
    return winner;
}
