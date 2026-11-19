#pragma once

#include "dataset.hpp"

// The main experiments use the first six classes of SharvitB2.
inline constexpr int experimentClassCount = 6;

void runExperiments(const std::vector<Sample>& data);
