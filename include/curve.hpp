// Rayane — classification precision/recall
#pragma once

#include <optional>
#include <vector>

struct PRPoint {
    std::optional<double> threshold;  // Empty at the starting point (no positive predictions).
    double recall;
    double precision;
};

struct PRCurve {
    std::vector<PRPoint> points;
    double auc = 0.0;
};

// One-vs-rest curve. Each held-out image scores a class by the fraction of its votes for it.
PRCurve precisionRecallCurve(const std::vector<int>& truth,
                            const std::vector<std::vector<int>>& votes, int positiveClass);
