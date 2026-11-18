// Rayane 3.b evaluation.hpp
#pragma once
#include <vector>

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

// the labels are assumed to be in the range [0, numClasses)
EvaluationReport evaluate(const std::vector<int>& truth,
                          const std::vector<int>& predicted,
                          int numClasses);

void printReport(const EvaluationReport& report);