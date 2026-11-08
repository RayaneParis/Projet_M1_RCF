// Rayane 2.c vote majoritaire
#include "vote.hpp"
#include "knn.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

int predictVote(const std::vector<Sample>& training, const Sample& query,
                const std::vector<std::string>& methods, int k, DistanceType type) {
            if (methods.empty()) 
                throw std::invalid_argument("vote requires at least one method");

            std::vector<int> predictions;
            std::map<int, int> votes;
            for (const auto& m : methods) {
                int p = predictKNN(training, query.features.at(m), m, k, type);
                predictions.push_back(p);
                ++votes[p];
            }

            int maxVotes = 0;
            for (const auto& [label, v] : votes) maxVotes = std::max(maxVotes, v);

            for (int p : predictions)
                if (votes[p] == maxVotes) return p;
            return predictions.front(); // fallback 
}

double leaveOneOutVote(const std::vector<Sample>& data,
                        const std::vector<std::string>& methods, 
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