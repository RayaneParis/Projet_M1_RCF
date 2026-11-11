// rayane

#pragma once

#include <string>
#include <vector>

#include "dataset.hpp"
#include "distance.hpp"

// the method order decides ties between classes
int predictVote(const std::vector<Sample>& training, const Sample& query,
                const std::vector<std::string>& methods, int k, DistanceType type);

double leaveOneOutVote(const std::vector<Sample>& data, const std::vector<std::string>& methods,
                       int k, DistanceType type);
