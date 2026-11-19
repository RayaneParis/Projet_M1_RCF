// Rayane 3.b evaluation.cpp

#include "evaluation.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>

void forEachLeaveOneOutFold(const std::vector<Sample>& data, const FoldVisitor& visit) {
    if (data.size() < 2) {
        throw std::invalid_argument("leave-one-out requires at least two samples");
    }

    std::vector<Sample> training(data.begin() + 1, data.end());
    for (std::size_t i = 0; i < data.size(); ++i) {
        visit(training, data[i]);
        // Restore this sample in place of the next held-out sample, preserving order.
        if (i < training.size()) training[i] = data[i];
    }
}

Predictions leaveOneOut(const std::vector<Sample>& data, const SamplePredictor& predict) {
    Predictions result;
    result.truth.reserve(data.size());
    result.predicted.reserve(data.size());
    forEachLeaveOneOutFold(data, [&](const auto& training, const Sample& query) {
        result.truth.push_back(query.label);
        result.predicted.push_back(predict(training, query));
    });
    return result;
}

Accuracy classificationAccuracy(const std::vector<int>& truth, const std::vector<int>& predicted) {
    if (truth.empty() || truth.size() != predicted.size()) {
        throw std::invalid_argument("evaluate: truth and predicted must have the same nonzero size");
    }
    std::size_t correct = 0;
    for (std::size_t i = 0; i < truth.size(); ++i) {
        if (truth[i] == predicted[i]) ++correct;
    }
    return {correct, 100.0 * correct / truth.size()};
}

// computes confusion matrix, accuracy and precision/recall/f1 for each class
// labels go from 1 to numClasses
EvaluationReport evaluate(const std::vector<int>& truth, const std::vector<int>& predicted,
                          int numClasses) {
    // check the inputs
    const Accuracy accuracy = classificationAccuracy(truth, predicted);
    if (numClasses <= 0) {
        throw std::invalid_argument("evaluate: numClasses must be positive");
    }

    const int C = numClasses;
    EvaluationReport r;
    // confusion matrix full of zeros
    r.confusion.assign(C, std::vector<int>(C, 0));

    // fill the confusion matrix
    for (std::size_t i = 0; i < truth.size(); ++i) {
        // labels start at 1 but indexes start at 0, so we remove 1
        int t = truth[i] - 1, p = predicted[i] - 1;
        if (t < 0 || t >= C || p < 0 || p >= C)
            throw std::invalid_argument("evaluate: label out of range");
        ++r.confusion[t][p];
    }
    // accuracy in percent
    r.accuracy = accuracy.percent;

    // metrics for each class
    r.perClass.resize(C);
    for (int c = 0; c < C; ++c) {
        // tp = well classified samples of class c (the diagonal)
        int tp = r.confusion[c][c];

        // column sum = everything predicted as c
        // row sum = everything that is really c
        int predictedAsC = 0, actuallyC = 0;
        for (int k = 0; k < C; ++k) {
            predictedAsC += r.confusion[k][c];
            actuallyC += r.confusion[c][k];
        }

        ClassMetrics& m = r.perClass[c];
        // precision = tp / (tp + fp), we put 0 if nothing is predicted as c
        m.precision = predictedAsC > 0 ? double(tp) / predictedAsC : 0.0;
        // recall = tp / (tp + fn), we put 0 if there is no sample of c
        m.recall = actuallyC > 0 ? double(tp) / actuallyC : 0.0;
        // f1 = mix of precision and recall, 0 if both are 0 (to avoid dividing by 0)
        m.f1 = (m.precision + m.recall) > 0
                   ? 2.0 * m.precision * m.recall / (m.precision + m.recall)
                   : 0.0;

        // macro average = mean over the classes (we add 1/C of each class)
        // the macro values must start at 0 in the struct
        r.macroPrecision += m.precision / C;
        r.macroRecall += m.recall / C;
        r.macroF1 += m.f1 / C;
    }
    return r;
}

double clusterPurity(const std::vector<Sample>& data, const KMeansResult& result) {
    if (data.empty() || result.centroids.empty() || result.assignment.size() != data.size()) {
        throw std::invalid_argument(
            "cluster purity requires data, clusters and one assignment per sample");
    }

    std::vector<std::map<int, int>> counts(result.centroids.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
        int cluster = result.assignment[i];
        if (cluster < 0 || static_cast<std::size_t>(cluster) >= counts.size()) {
            throw std::invalid_argument("cluster assignment is out of range");
        }
        ++counts[cluster][data[i].label];
    }

    std::size_t correct = 0;

    // count the majority class in each cluster
    for (const auto& cluster : counts) {
        int best = 0;
        for (const auto& entry : cluster) {
            best = std::max(best, entry.second);
        }
        correct += best;
    }

    return 100.0 * correct / data.size();
}

ClusteringSummary summarizeClustering(const std::vector<Sample>& data,
                                      const std::vector<KMeansResult>& runs) {
    if (runs.empty()) throw std::invalid_argument("clustering summary requires at least one run");

    ClusteringSummary summary;
    for (std::size_t i = 0; i < runs.size(); ++i) {
        const double purity = clusterPurity(data, runs[i]);
        summary.minPurity = std::min(summary.minPurity, purity);
        summary.maxPurity = std::max(summary.maxPurity, purity);
        summary.meanPurity += purity;
        summary.maxSse = std::max(summary.maxSse, runs[i].sse);
        if (i == 0 || runs[i].sse < runs[summary.bestRun].sse) {
            summary.bestRun = i;
            summary.bestRunPurity = purity;
        }
    }
    summary.meanPurity /= runs.size();
    return summary;
}
