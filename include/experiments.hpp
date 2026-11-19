#pragma once

#include "dataset.hpp"

// experiment 1: take the first 6 classes only
inline constexpr int experimentClassCount = 6;

void runExperiments(const std::vector<Sample>& data);
