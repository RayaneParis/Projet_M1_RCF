// mehdi & rayane
#include "experiments.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "curve.hpp"
#include "evaluation.hpp"
#include "kmeans.hpp"
#include "knn.hpp"
#include "svm.hpp"
#include "vote.hpp"

namespace {

constexpr int curveNeighborCount = 12;
const std::vector<int> neighborCounts = {1, 3, 5, 7, 9};
const std::vector<std::string> votePriority = {"ART", "GFD", "Zernike7", "Yang", "SA"};

using PredictionsBySetting = std::map<std::pair<DistanceType, int>, Predictions>;
using KnnPredictions = std::map<std::string, PredictionsBySetting>;

void printSection(const std::string& title) {
    std::cout << "\n======================= " << title << " =======================\n";
}
// testing all 5 descriptors with L1 and L2 and k= 1,3,5,7,9
// it uses leaveOneOut() for the predictions and classificationAccuracy() to calculate the
// recognition rate
KnnPredictions runKNN(const std::vector<Sample>& data) {
    printSection("2.a. k-NN classification (leave-one-out)");

    KnnPredictions results;

    std::filesystem::create_directories("results");

    std::ofstream csv("results/experiment1_knn_6classes.csv");

    if (!csv) {
        throw std::runtime_error("Cannot create experiment CSV");
    }

    csv << "descriptor,distance,k,correct,total,recognition_rate_percent\n";

    for (const auto& method : methodNames()) {
        for (DistanceType type : {DistanceType::Euclidean, DistanceType::Manhattan}) {
            for (int k : neighborCounts) {
                Predictions& result = results[method][{type, k}];
                result = leaveOneOut(data, [&](const auto& training, const Sample& query) {
                    return predictKNN(training, query.features.at(method), method, k, type);
                });
                const Accuracy accuracy = classificationAccuracy(result.truth, result.predicted);

                csv << method << "," << (type == DistanceType::Euclidean ? "L2" : "L1") << "," << k
                    << "," << accuracy.correct << "," << data.size() << "," << std::fixed
                    << std::setprecision(4) << accuracy.percent << "\n";

                std::cout << method << " | " << (type == DistanceType::Euclidean ? "L2" : "L1")
                          << " | k: " << k << " | correct: " << accuracy.correct << "/"
                          << data.size() << " | recognition rate: " << accuracy.percent << "%\n";
            }
        }
    }
    return results;
}

void runKMeans(const std::vector<Sample>& data) {
    const int clusterCount = 6;
    const unsigned seedCount = 20;
    printSection("2.b. K-means clustering (K = 6, 20 initializations)");

    for (const auto& method : methodNames()) {
        std::vector<KMeansResult> runs;
        for (unsigned seed = 0; seed < seedCount; ++seed) {
            runs.push_back(kmeans(data, method, clusterCount, seed));
        }
        const ClusteringSummary summary = summarizeClustering(data, runs);
        const KMeansResult& best = runs[summary.bestRun];
        std::cout << method << " | SSE: " << best.sse << " | iterations: " << best.iterators
                  << " | purity: " << summary.bestRunPurity << "%\n";
        std::cout << method << " | purity min/mean/max (%): " << summary.minPurity << " / "
                  << summary.meanPurity << " / " << summary.maxPurity
                  << " | SSE min/max: " << best.sse << " / " << summary.maxSse << "\n";
    }
}

PredictionsBySetting runVoting(const KnnPredictions& knn) {
    printSection("2.c. Majority vote (leave-one-out)");

    PredictionsBySetting results;

    std::filesystem::create_directories("results");
    std::ofstream csv("results/experiment2_vote_6classes.csv");
    if (!csv) {
        throw std::runtime_error("Cannot create voting CSV");
    }
    csv << "distance,k,correct,total,recognition_rate_percent\n";

    for (DistanceType type : {DistanceType::Euclidean, DistanceType::Manhattan}) {
        for (int k : neighborCounts) {
            const auto setting = std::make_pair(type, k);
            Predictions& result = results[setting];
            result.truth = knn.at(votePriority.front()).at(setting).truth;
            std::vector<int> votes;
            for (std::size_t i = 0; i < result.truth.size(); ++i) {
                votes.clear();
                for (const auto& method : votePriority) {
                    votes.push_back(knn.at(method).at(setting).predicted.at(i));
                }
                result.predicted.push_back(majorityVote(votes));
            }

            const Accuracy accuracy = classificationAccuracy(result.truth, result.predicted);

            csv << (type == DistanceType::Euclidean ? "L2" : "L1") << "," << k << ","
                << accuracy.correct << "," << result.truth.size() << "," << std::fixed
                << std::setprecision(4) << accuracy.percent << "\n";

            std::cout << "vote | " << (type == DistanceType::Euclidean ? "L2" : "L1")
                      << " | k: " << k << " | recognition rate: " << accuracy.percent << "%\n";
        }
    }
    return results;
}

void runSVM(const std::vector<Sample>& data) {
    printSection("2.d. SVM classification (leave-one-out)");

    for (const auto& method : methodNames()) {
        const Predictions result =
            leaveOneOut(data, [&](const auto& training, const Sample& query) {
                return predictSVM(training, query.features.at(method), method);
            });
        const Accuracy accuracy = classificationAccuracy(result.truth, result.predicted);
        std::cout << method << " | SVM RBF | correct: " << accuracy.correct << "/" << data.size()
                  << " | recognition rate: " << accuracy.percent << "%\n";
    }
}

// prints the report: confusion matrix, table per class and the global results
void printReport(const EvaluationReport& r) {
    const int C = static_cast<int>(r.confusion.size());
    const auto flags = std::cout.flags();
    const auto precision = std::cout.precision();

    // confusion matrix
    std::cout << "Confusion matrix (rows = true class, columns = predicted)\n";
    // first line with the predicted class numbers
    std::cout << "       ";
    for (int p = 1; p <= C; ++p) std::cout << std::setw(4) << p;
    std::cout << "\n";
    // one line per true class
    for (int t = 0; t < C; ++t) {
        std::cout << std::setw(4) << t + 1 << " |";
        for (int p = 0; p < C; ++p) std::cout << std::setw(4) << r.confusion[t][p];
        std::cout << "\n";
    }

    // table per class, 3 digits after the point
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "class | precision | recall | F1\n";
    for (int c = 0; c < C; ++c)
        std::cout << std::setw(5) << c + 1 << " | " << std::setw(9) << r.perClass[c].precision
                  << " | " << std::setw(6) << r.perClass[c].recall << " | " << std::setw(5)
                  << r.perClass[c].f1 << "\n";

    // global results
    std::cout << "accuracy: " << r.accuracy << "%"
              << " | macro precision: " << r.macroPrecision << " | macro recall: " << r.macroRecall
              << " | macro F1: " << r.macroF1 << "\n";
    std::cout.flags(flags);
    std::cout.precision(precision);
}

void printEvaluation(const std::string& name, const PredictionsBySetting& results, int k) {
    const Predictions& result = results.at({DistanceType::Euclidean, k});
    std::cout << "\n--- " << name << " | L2 | k = " << k << " ---\n";
    printReport(evaluate(result.truth, result.predicted, experimentClassCount));
}

void runEvaluation(const KnnPredictions& knn, const PredictionsBySetting& voting) {
    printSection("3.b. Evaluation metrics: confusion matrix, precision, recall, F1-score");

    printEvaluation("k-NN | SA", knn.at("SA"), 1);
    printEvaluation("k-NN | ART", knn.at("ART"), 5);
    printEvaluation("Majority vote", voting, 9);
}

void exportClassCurves(const std::string& approach, const std::vector<int>& truth,
                       const std::vector<std::vector<int>>& votes, std::ostream& csv) {
    double sumAuc = 0.0;
    for (int label = 1; label <= experimentClassCount; ++label) {
        const PRCurve curve = precisionRecallCurve(truth, votes, label);
        sumAuc += curve.auc;
        std::cout << approach << " | class " << label << " | PR-AUC: " << curve.auc << "\n";
        for (const PRPoint& point : curve.points) {
            csv << approach << ",L2," << curveNeighborCount << ',' << label << ',';
            if (point.threshold) csv << *point.threshold;
            csv << ',' << point.recall << ',' << point.precision << ',' << curve.auc << '\n';
        }
    }
    std::cout << approach << " | macro PR-AUC: " << sumAuc / experimentClassCount << "\n";
}

void runPrecisionRecall(const std::vector<Sample>& data) {
    printSection("3.c. Classifier precision/recall (12 neighbors, L2, one-vs-rest)");

    std::vector<int> truth;
    std::map<std::string, std::vector<std::vector<int>>> neighborVotes;
    std::vector<std::vector<int>> descriptorVotes;
    forEachLeaveOneOutFold(data, [&](const auto& training, const Sample& query) {
        truth.push_back(query.label);
        std::vector<int> decisions;
        for (const auto& method : votePriority) {
            auto neighbors = nearestNeighborLabels(training, query.features.at(method), method,
                                                   curveNeighborCount, DistanceType::Euclidean);
            decisions.push_back(majorityVote(neighbors));
            neighborVotes[method].push_back(std::move(neighbors));
        }
        descriptorVotes.push_back(std::move(decisions));
    });

    std::ostringstream csv;
    csv << std::setprecision(16) << "approach,distance,k,class,threshold,recall,precision,auc\n";
    for (const auto& method : methodNames()) {
        exportClassCurves("k-NN " + method, truth, neighborVotes.at(method), csv);
    }
    exportClassCurves("Majority vote", truth, descriptorVotes, csv);

    std::filesystem::create_directories("results");
    std::ofstream file("results/curves.csv");
    file << csv.str();
    file.close();
    if (!file) throw std::runtime_error("cannot write results/curves.csv");
}

}  // namespace

void runExperiments(const std::vector<Sample>& data) {
    if (data.size() <= static_cast<std::size_t>(curveNeighborCount)) {
        throw std::invalid_argument("leave-one-out with 12 neighbors requires at least 13 samples");
    }
    std::cout << data.size() << " samples loaded\n";
    std::cout << "all samples have the expected descriptor dimensions\n";

    // const KnnPredictions knn = runKNN(data);
    runKMeans(data);
    // const PredictionsBySetting voting = runVoting(knn);
    // runSVM(data);
    // runEvaluation(knn, voting);
    // runPrecisionRecall(data);
}
