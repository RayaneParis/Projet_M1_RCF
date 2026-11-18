// Rayane 3.b evaluation.cpp

#include "evaluation.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>

EvaluationReport evaluate(const std::vector<int>& truth,
                          const std::vector<int>& predicted,
                          int numClasses) {
    if (truth.empty() || predicted.size() != predicted.size())
        throw std::invalid_argument("evaluate: truth and predicted must have the same nonzero size");

        const int C = numClasses;
        EvaluationReport r;
        r.confusion.assign(C, std::vector<int>(C, 0));

        std::size_t correct = 0;
        for (std::size_t i = 0; i < truth.size(); ++i) {
            int t = truth[i] - 1, p = predicted[i] - 1;
            if (t < 0 || t>=C || p < 0 || p>= C)
                throw std::invalid_argument("evaluate: label out of range");
            ++r.confusion[t][p];
            if (t == p) ++correct;
        }
        r.accuracy = 100.0 * correct / truth.size();

        r.perClass.resize(C);
        for (int c = 0; c < C; ++c) {
            int tp = r.confusion[c][c];
            int predictedAsC = 0, actuallyC = 0;
            for(int k = 0; k < C; ++k) {
                predictedAsC += r.confusion[k][c];
                actuallyC += r.confusion[c][k];
            }
            ClassMetrics& m = r.perClass[c];
            m.precision = predictedAsC > 0 ? double(tp) / predictedAsC : 0.0;
            m.recall = actuallyC > 0 ? double(tp) / actuallyC : 0.0;
            m.f1 = (m.precision + m.recall) > 0
                ? 2.0 * m.precision * m.recall / (m.precision + m.recall) : 0.0;
            r.macroPrecision += m.precision / C;
            r.macroRecall += m.recall / C;
            r.macroF1 += m.f1 / C;
        }
        return r;  
}

void printReport(const EvaluationReport& r) {
    const int C = static_cast<int>(r.confusion.size());

    std::cout << "Confusion matrix (rows = true class, columns = predicted)\n";
    std::cout << "       ";
    for (int p = 1; p <= C; ++p) std::cout << std::setw(4) << p;
    std::cout << "\n";
    for (int t = 0; t < C; ++t) {
        std::cout << std::setw(4) << t + 1 << " |";
        for (int p = 0; p < C; ++p) std::cout << std::setw(4) << r.confusion[t][p];
        std::cout << "\n";
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "class | precision | recall | F1\n";
    for (int c = 0; c < C; ++c) 
        std::cout << std::setw(5) << c + 1 << " | " << std::setw(9) << r.perClass[c].precision
                    << " | " << std::setw(6) << r.perClass[c].recall
                    << " | " << std::setw(5) << r.perClass[c].f1 << "\n";

    std::cout << "accuracy: " << r.accuracy << "%"
              << " | macro precision: " << r.macroPrecision
              << " | macro recall: " << r.macroRecall
              << " | macro F1: " << r.macroF1 << "\n";
    std::cout.unsetf(std::ios::floatfield);
}

