// Mehdi - euclidean distance

#include "distance.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

double euclideanDistance(const std::vector<double>& a,
                         const std::vector<double>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("euclidean distance requires equal length vectors");
    }

    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        double difference = a[i] - b[i];
        sum += difference * difference;
    }

    return std::sqrt(sum);
}
