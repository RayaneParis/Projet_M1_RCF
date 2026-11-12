# Pattern Recognition Project

M1 VMI — Université Paris Cité
Mehdi AGHAEI and Rayane Khatim

We compare different shape recognition methods using the **SharvitB2 dataset**, which contains 18 classes with 12 samples each.

The project is implemented in **C++17** and includes:
- euclidean (L2) distance - complete implementation
- manhattan (L1) distance - complete implementation
- k-Nearest Neighbors (k-NN) - complete implementation
- k-Means Clustering - complete implementation
- Majority Voting across five feature descriptors - complete implementation
- Support Vector Machine (SVM, RBF kernel) | OpenCV (`cv::ml::SVM`) 

## Dataset

Each image is represented by five precomputed feature descriptors: **ART, GFD, SA, Yang, and Zernike7**.

Place the dataset in:

```text
data/SharvitB2/
```

The program also supports `data/=SharvitB2/` with the corresponding prefixed subfolders.

## Requirements
 
- A C++17 compiler (GCC, Clang or MSVC)
- CMake 3.16 or newer
- OpenCV 4 (modules `core` and `ml`), used only for the SVM
```bash
# macOS
brew install cmake opencv
 
# Ubuntu / Debian
sudo apt install build-essential cmake libopencv-dev
```
 
## Build and Run
 
From the repository root:
 
```bash
cmake -S . -B build
cmake --build build
./build/projet_M1_RCF
```

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
