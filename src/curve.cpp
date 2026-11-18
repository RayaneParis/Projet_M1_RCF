// Rayane 3.c Curve precision/recall
#include "curve.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace {

PRCurve buildCurve(const std::vector<Sample>& data,
                   const std::vector<std::vector<int>>& ranking, int K) {
    const std::size_t n = data.size();
    PRCurve curve;
    curve.points.resize(K);

    for (int r = 1; r <= K; ++r) {
        double sumPrecision = 0.0, sumRecall = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            int relevantTotal = 0;
            for (int label : ranking[i])
                if (label == data[i].label) ++relevantTotal;

            int found = 0;
            for (int pos = 0; pos < r; ++pos)
                if (ranking[i][pos] == data[i].label) ++found;

            sumPrecision += static_cast<double>(found) / r;
            sumRecall += relevantTotal > 0 ? static_cast<double>(found) / relevantTotal : 0.0;
        }
        curve.points[r - 1] = {r, sumRecall / n, sumPrecision / n};
    }

    double auc = 0.0, prevR = 0.0, prevP = curve.points[0].precision;
    for (const auto& p : curve.points) {
        auc += (p.recall - prevR) * (p.precision + prevP) / 2.0;
        prevR = p.recall;
        prevP = p.precision;
    }
    curve.auc = auc;
    return curve;
}

void checkK(const std::vector<Sample>& data, int K) {
    if (K < 1 || static_cast<std::size_t>(K) > data.size() - 1)
    throw std::invalid_argument("precision/recall: K must be between 1 and n - 1");
}

}

PRCurve precisionRecallCurve(const std::vector<Sample>& data, const std::string& method,
                             DistanceType type, int K) {
    checkK(data, K);
    const std::size_t n = data.size();
    std::vector<std::vector<int>> ranking(n);

    for (std::size_t i = 0; i < n; ++i) {
        std::vector<std::pair<double, int>> d; // (distance. label)
        for (std::size_t j = 0; j < n; ++j) {
            if (j == i) continue;
            d.push_back({computeDistance(data[i].features.at(method),
                                         data[j].features.at(method), type),
                                         data[j].label});
        }
        std::stable_sort(d.begin(), d.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& p : d) ranking[i].push_back(p.second);
    }
    return buildCurve(data, ranking, K);
}

PRCurve precisionRecallCurveVote(const std::vector<Sample>& data,
                                 const std::vector<std::string>& methods,
                                 DistanceType type, int K) {
    checkK(data, K);
    const std::size_t n = data.size();
    std::vector<std::vector<int>> ranking(n);

    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> avgRank(n, 0.0);

        for (const auto& m : methods) {
            std::vector<std::pair<double, std::size_t>> d;
            for (std::size_t j = 0; j < n; ++j) {
                if (j == i) continue;
                d.push_back({computeDistance(data[i].features.at(m),
                                             data[j].features.at(m), type), j});
            }
            std::stable_sort(d.begin(), d.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            for (std::size_t pos = 0; pos < d.size(); ++pos)
                avgRank[d[pos].second] += static_cast<double>(pos + 1) / methods.size();
        }

        std::vector<std::size_t> others;
        for (std::size_t j = 0; j < n; ++j)
            if (j != i) others.push_back(j);
        std::stable_sort(others.begin(), others.end(),
                         [&](std::size_t a, std::size_t b) { return avgRank[a] < avgRank[b]; });
        for (std::size_t j : others) ranking[i].push_back(data[j].label);
    }
    return buildCurve(data, ranking, K);
}