// mehdi aghaei & rayane - knn

#pragma once

#include "dataset.hpp"
#include "distance.hpp"

// Labels of the k closest neighbors, nearest first. Equal distances preserve training order.
std::vector<int> nearestNeighborLabels(const std::vector<Sample>& training,
                                       const std::vector<double>& query,
                                       const std::string& method, int k, DistanceType type);

// ties use the closest neighbor, then the original training order
int predictKNN(const std::vector<Sample>& training, const std::vector<double>& query,
               const std::string& method, int k, DistanceType type = DistanceType::Euclidean);
