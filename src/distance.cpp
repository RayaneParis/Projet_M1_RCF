// Mehdi & Rayane - euclidean/manhattan distance

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

double manhattanDistance(const std::vector<double>& a,
                         const std::vector<double>& b) {
    if (a.size() != b.size()) 
        throw std::invalid_argument("manhattan distance requires equal length vectors");
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) sum += std::abs(a[i] - b[i]);
    return sum;
}

double computeDistance(const std::vector<double>& a,
                        const std::vector<double>& b,
                        DistanceType type) {
    return type == DistanceType::Manhattan ? manhattanDistance(a, b)
                                         : euclideanDistance(a, b);
}
        

