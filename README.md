# Pattern Recognition Project

M1 VMI — Université Paris Cité
Mehdi AGHAEI and Rayane Khatim

We compare different shape recognition methods using the **SharvitB2 dataset**, which contains 18 classes with 12 samples each.

The project is implemented in **C++17** and includes:
- k-Nearest Neighbors (k-NN)
- k-Means Clustering
- Majority Voting across five feature descriptors
- Support Vector Machine (SVM)

## Dataset

Each image is represented by five precomputed feature descriptors: **ART, GFD, SA, Yang, and Zernike7**.

Place the dataset in:

```text
data/SharvitB2/
```

The program also supports `data/=SharvitB2/` with the corresponding prefixed subfolders.

## Build and Run

From the repository root:

```bash
cmake -S . -B build
cmake --build build
./build/projet_M1_RCF
```

Requires a C++17-compatible compiler and CMake.

## Project Structure

```text
include/       C++ header files
src/           C++ source files
data/          Local dataset
docs/          Report, slides
CMakeLists.txt Build configuration
```

## Authors

Mehdi Aghaei & Rayane Khatim
