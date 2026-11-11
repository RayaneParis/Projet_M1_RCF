// mehdi aghaei & rayane - knn

#pragma once

#include "dataset.hpp"
#include "distance.hpp"

// ties use the closest neighbor, then the original training order
int predictKNN(const std::vector<Sample>& training, const std::vector<double>& query,
               const std::string& method, int k, DistanceType type = DistanceType::Euclidean);
