// mehdi

#pragma once

#include "dataset.hpp"

int predictSVM(const std::vector<Sample>& training, const std::vector<double>& query,
               const std::string& method, double C = 1.0);

