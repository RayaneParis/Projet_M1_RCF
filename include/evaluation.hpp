// Rayane 3.b evaluation.hpp
#pragma once
#include <cstddef>
#include <functional>
#include <vector>

#include "dataset.hpp"
#include "kmeans.hpp"

struct Predictions {
    std::vector<int> truth;
    std::vector<int> predicted;
};

using SamplePredictor = std::function<int(const std::vector<Sample>& training, const Sample& query)>;
using FoldVisitor = std::function<void(const std::vector<Sample>& training, const Sample& query)>;

// Visit each held-out sample once, keeping training samples in their original order.
void forEachLeaveOneOutFold(const std::vector<Sample>& data, const FoldVisitor& visit);

// Both label vectors follow the original sample order.
Predictions leaveOneOut(const std::vector<Sample>& data, const SamplePredictor& predict);

struct Accuracy {
    std::size_t correct;
    double percent;
};

Accuracy classificationAccuracy(const std::vector<int>& truth, const std::vector<int>& predicted);

struct ClassMetrics {
    double precision = 0.0;
    double recall = 0.0;
    double f1 = 0.0;
};

struct EvaluationReport {
    std::vector<std::vector<int>> confusion ;
    std::vector<ClassMetrics> perClass;
    double accuracy = 0.0;
    double macroPrecision = 0.0;
    double macroRecall = 0.0;
    double macroF1 = 0.0;
};

// labels range from 1 to numClasses
EvaluationReport evaluate(const std::vector<int>& truth,
                          const std::vector<int>& predicted,
                          int numClasses);

double clusterPurity(const std::vector<Sample>& data, const KMeansResult& result);

struct ClusteringSummary {
    std::size_t bestRun = 0;  // First run with minimum SSE.
    double bestRunPurity = 0.0;
    double minPurity = 100.0;
    double meanPurity = 0.0;
    double maxPurity = 0.0;
    double maxSse = 0.0;
};

// Purities are percentages. True labels are used only after clustering.
ClusteringSummary summarizeClustering(const std::vector<Sample>& data,
                                      const std::vector<KMeansResult>& runs);
