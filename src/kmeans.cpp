// rayane
// commented by mehdi

#include "kmeans.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>

#include "distance.hpp"

// k-means groups samples into k clusters using their feature vectors
KMeansResult kmeans(const std::vector<Sample>& data, const std::string& method, int K,
                    unsigned seed, int maxIter) {

    // check that the dataset is not empty and k is valid
    if (data.empty() || K < 1 || static_cast<std::size_t>(K) > data.size())
        throw std::invalid_argument("k-means requires data and a valid number of clusters");

    // n is the number of samples, dim is the number of values in each descriptor
    const std::size_t n = data.size();
    const std::size_t dim = data[0].features.at(method).size();

    // create sample indexes from 0 to n-1, then shuffle them
    // the seed allows us to reproduce the same random initialization
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);

    KMeansResult r;

    // -1 means the sample has not been assigned to a cluster yet
    r.assignment.assign(n, -1);

    // choose k random samples as the initial cluster centers
    for (int c = 0; c < K; ++c) r.centroids.push_back(data[idx[c]].features.at(method));

    // repeat the assignment and update steps until convergence or maxIter
    for (int it = 0; it < maxIter; ++it) {
        r.iterators = it + 1;
        bool changed = false;

        // step 1: assign each sample to its nearest center
        for (std::size_t i = 0; i < n; ++i) {
            const auto& x = data[i].features.at(method);
            int best = 0;
            double bestD = euclideanDistance(x, r.centroids[0]);

            // compare the sample with all the other centers
            for (int c = 1; c < K; ++c) {
                double d = euclideanDistance(x, r.centroids[c]);
                if (d < bestD) {
                    bestD = d;
                    best = c;
                }
            }

            // update the assignment if the sample has changed clusters
            if (r.assignment[i] != best) {
                r.assignment[i] = best;
                changed = true;
            }
        }

        // stop if no sample changed clusters during this iteration
        if (!changed) break;

        // step 2: prepare the sums and number of samples for each cluster
        // sum[c][d] stores the sum of feature d in cluster c
        // count[c] stores how many samples belong to cluster c
        std::vector<std::vector<double>> sum(K, std::vector<double>(dim, 0.0));
        std::vector<int> count(K, 0);

        // add each sample's feature values to its assigned cluster
        for (std::size_t i = 0; i < n; ++i) {
            const auto& x = data[i].features.at(method);
            int c = r.assignment[i];
            for (std::size_t d = 0; d < dim; ++d) sum[c][d] += x[d];
            ++count[c];
        }

        // step 3: calculate new centers by taking the mean of each cluster
        // if a cluster is empty, keep its previous center
        for (int c = 0; c < K; ++c)
            if (count[c] > 0)
                for (std::size_t d = 0; d < dim; ++d) r.centroids[c][d] = sum[c][d] / count[c];
    }

    // calculate sse: sum of squared distances between samples and their centers
    // a smaller sse means samples are closer to their assigned centers
    for (std::size_t i = 0; i < n; ++i) {
        double d = euclideanDistance(data[i].features.at(method), r.centroids[r.assignment[i]]);
        r.sse += d * d;
    }
    return r;
}


// calculate cluster purity after k-means has finished
// in contrast to k-means, this function uses the real class labels
// it is not the same as classification accuracy on unseen samples
double clusterRecognitionRate(const std::vector<Sample>& data, const KMeansResult& r, int K) {

    // count how many samples of each real class are inside each cluster
    std::vector<std::map<int, int>> counts(K);
    for (std::size_t i = 0; i < data.size(); ++i) {
        ++counts[r.assignment[i]][data[i].label];
    }

    std::size_t correct = 0;

    // for each cluster, find the class that appears most often
    // add that number to the total
    for (const auto& m : counts) {
        int best = 0;
        for (const auto& [label, c] : m) best = std::max(best, c);
        correct += best;
    }

    // cluster purity = total majority-class samples / total samples
    return 100.0 * correct / data.size();
}
