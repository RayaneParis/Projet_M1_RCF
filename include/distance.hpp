// mehdi & rayane

#pragma once

#include <vector>

enum class DistanceType {
    Euclidean,
    Manhattan
};

// both distances require vectors of equal length
double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b);

double manhattanDistance(const std::vector<double>& a, const std::vector<double>& b);

double computeDistance(const std::vector<double>& a, const std::vector<double>& b,
                       DistanceType type);
