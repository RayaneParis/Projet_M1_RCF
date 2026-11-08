// Rayane 2.b K-means
#include "kmeans.hpp"
#include "distance.hpp"

#include <algorithm>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>

KMeansResult kmeans(const std::vector<Sample>& data, const std::string& method,
                int K, unsigned seed, int maxIter) {
    if (data.empty() || K < 1 || static_cast<std::size_t> (K) > data.size())
        throw std::invalid_argument("k-means requires data and a valid number of clusters");

        const std::size_t n = data.size();
        const std::size_t dim = data[0].features.at(method).size();

        std::vector<std::size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(idx.begin(), idx.end(), rng);

        KMeansResult r;
        r.assignment.assign(n, -1);
        for (int c = 0; c < K; ++c)
            r.centroids.push_back(data[idx[c]].features.at(method));

        for (int it =  0; it < maxIter; ++it) {
            r.iterators = it +1;
            bool changed = false;

            for (std::size_t i = 0; i < n; ++i) {
                const auto& x = data[i].features.at(method);
                int best = 0;
                double bestD = euclideanDistance(x, r.centroids[0]);
                for (int c = 1; c < K; ++c) {
                    double d = euclideanDistance(x, r.centroids[c]);
                    if (d < bestD) {
                        bestD = d;
                        best = c;
                    }
                }
                if (r.assignment[i] != best) {
                    r.assignment[i] = best;
                    changed = true;
                }
            }
            if (!changed) break;

                std::vector<std::vector<double>> sum(K, std::vector<double>(dim, 0.0));
                std::vector<int> count(K, 0);
                for (std::size_t i = 0; i < n; ++i) {
                    const auto& x = data[i].features.at(method);
                    int c = r.assignment[i];
                    for (std::size_t d = 0; d < dim; ++d) sum[c][d] += x[d];
                    ++count[c];
                }
                for (int c = 0; c < K; ++c)
                    if (count[c] > 0)
                        for (std::size_t d = 0; d < dim; ++d)
                            r.centroids[c][d] = sum[c][d] / count[c];
            }

            for (std::size_t i = 0; i < n; ++i) {
                double d = euclideanDistance(data[i].features.at(method), r.centroids[r.assignment[i]]);
                r.sse += d * d;
            }
            return r;
        }

        double clusterRecognitionRate(const std::vector<Sample>& data, const KMeansResult& r, int K) {
            std::vector<std::map<int, int>> counts(K);
            for (std::size_t i = 0; i < data.size(); ++i) {
                ++counts[r.assignment[i]][data[i].label];
            }

            std::size_t correct = 0;
            for (const auto& m : counts) {
                int best = 0;
                for (const auto& [label, c] : m) best = std::max(best, c);
                correct += best;
            }
            return 100.0 * correct / data.size();
        }
        


