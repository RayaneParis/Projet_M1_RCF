// mehdi aghaei & rayane - knn

#pragma once

#include "dataset.hpp"
#include "distance.hpp"

// predict the class of a sample using its knn
//so the class with the most votes wins
// if classes have the same number of votes(for example 2 votes cat and two votes dog), the class with the closest neighbor wins(distance). if distances are also equal, the order in which samples appear in training vector, will be used.
// throws std::invalid_argument if k is invalid or the data is empty or has incompatible dimensions
int predictKNN(const std::vector<Sample>& training,
               const std::vector<double>& query,
               const std::string& method,
               int k,
               DistanceType type = DistanceType::Euclidean);
