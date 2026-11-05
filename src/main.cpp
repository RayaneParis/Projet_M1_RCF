#include <iostream>
#include "dataset.hpp"

int main() {
    auto data = loadDataset("data/SharvitB2", 18);  // Load the dataset from the specified directory with a maximum of 18 classes
    std::cout << data.size() << " samples loaded\n";

    if (data.empty()) {std::cerr << "No samples found in the dataset.\n"; return 1; } // Check if the dataset is empty and print an error message

    for (const auto& m : methodNames()) 
        std::cout << " : dimension " << data[0].features.at(m).size() << "\n";  // print the dimension of the first sample's features for each method
}
