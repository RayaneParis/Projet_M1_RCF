// Rayane 2.c vote majoritaire

#pragma once 
#include <string>
#include <vector>
#include "dataset.hpp"
#include "distance.hpp"

int predictVote(const std ::vector<Sample>& training, const Sample& query,
                const std::vector<std::string>& methods, int k, DistanceType type);

double leaveOneOutVote(const std::vector<Sample>& data, 
                       const std::vector<std::string>& methods,
                       int k, DistanceType type);


                