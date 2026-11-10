// Mehdi & Rayane - euclidean/manhattan distance

#pragma once

#include <vector>

enum class DistanceType { Euclidean, Manhattan };

// throws std::invalid_argument if their lengths differ.
double euclideanDistance(const std::vector<double>& a,
                         const std::vector<double>& b);

double manhattanDistance(const std::vector<double>& a,
                         const std::vector<double>& b);

double computeDistance(const std::vector<double>& a,
                         const std::vector<double>& b,
                        DistanceType type);
