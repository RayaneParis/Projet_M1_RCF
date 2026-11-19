// mehdi & rayane
#include <exception>
#include <iostream>

#include "dataset.hpp"
#include "experiments.hpp"

int main() {
    try {
        const auto data = loadDataset("data/SharvitB2", experimentClassCount);
        runExperiments(data);
    } catch (const std::exception& error) {
        std::cerr << "experiment failed: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
