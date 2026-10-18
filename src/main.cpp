#include <iostream>
#include "dataset.hpp"

int main() {
    auto data = loadDataset("data/SharvitB2", 18);
    std::cout << data.size() << " samples loaded\n";
    for (const auto& m : methodNames()) 
        std::cout << " : dimension " << data[0].features.at(m).size() << "\n";  // print the dimension of the first sample's features for each method
}
